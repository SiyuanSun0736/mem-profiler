# 交叉熵之后的优化备选

> 这份文档回答“在已经接入三分类交叉熵辅助头后，还能用哪些方法继续优化模型和单程序评分”。

## 1. 当前基线

当前主模型已经使用：

1. pairwise log-ratio 回归头。
2. 三分类交叉熵辅助头，默认 `aux_class_lambda=0.05`。
3. tie / near-tie 回归降权。
4. O0/O2/O3 多锚点评分。
5. score-first tuned scoring 参数。

当前结果的形态很清楚：

1. proxy 单程序评分已经较稳：`corr_score_log = 0.9072`，`mae_score_log = 0.2723`，`band_acc = 0.8048`。
2. strict time 外部验证仍偏弱：默认 score-first 口径 `corr_model_time = 0.3980`，`band_acc_model = 0.6150`。
3. `O2-O3` 仍是最难 pair，主要因为 tie 密集且真实差异很小。

所以下一步优化不能只继续加模型复杂度，而应优先解决三个问题：

1. 训练目标和最终 time 指标不完全一致。
2. near-tie 区间标签噪声高。
3. 单程序评分依赖锚点聚合，锚点质量和不确定性还可以更精细。

## 2. 优先级 A：低成本、当前数据可直接做

### A1. Time-aware scoring fine tune

当前训练仍主要对齐 proxy label，最终验收却要看 strict time。可以先不改 backbone，只重新调评分层参数，让锚点聚合更偏向 time 指标。

状态：已完成首轮 focused retune。新增 `time-aware` tuned 选择口径，并在 [08-score-selection-objective-comparison.md](08-score-selection-objective-comparison.md) 中与 `score-first` / `time-first` 并排回放。

做法：

1. 继续使用当前 PairTransformer。
2. 在 [scripts/tune_score_program_fine.py](../../scripts/tune_score_program_fine.py) 中增加多目标选择函数。
3. 目标函数不要只看 `score_corr`，而是组合：
   1. proxy `corr_score_log`
   2. strict time `corr_model_time`
   3. strict time `band_acc_model`
   4. repeat-backed 子集 `corr_model_time`
4. 对 `tie_gate_threshold`、`tie_shrink_power`、`tie_margin_weight_alpha`、`min_anchor_quality` 和 outlier 参数重新搜索。

推荐目标：

$$
J =
0.35 \cdot r_{proxy}
+ 0.35 \cdot r_{time}
+ 0.20 \cdot acc^{time}_{band}
+ 0.10 \cdot r_{repeat}
- 0.10 \cdot MAE_{time}
$$

首轮验收标准：

1. strict time `corr_model_time` 高于默认 score-first 的 `0.3980`。
2. repeat-backed 子集不低于默认 score-first 的 `0.7881`。
3. proxy `corr_score_log` 不明显跌破 `0.90`。

首轮结果：

1. 默认 `score-first` 仍保留为主线：proxy `corr_score_log = 0.9072`，`mae_score_log = 0.2723`。
2. 可选 `time-aware` 口径达到：proxy `corr_score_log = 0.9059`，strict time `corr_model_time = 0.3993`，`mae_model_time = 1.0878`。
3. repeat-backed 子集从默认 `0.7881` 提升到 `0.7945`。
4. 收益方向正确但幅度很小，因此不建议替换默认口径；当实验更看重真实时间外部一致性时，可用 `--tuned-selection-objective time-aware` 显式切换。

这是当前最值得先做的优化，因为它直接针对“proxy 好、time 一般”的主要问题。首轮已经说明评分层可以轻微改善 time 外部一致性，但主要瓶颈仍在标签强度和 near-tie 样本。

### A2. Per-pair calibration

当前所有 variant pair 共用同一个回归输出尺度，但 `O0-O3` 和 `O2-O3` 的误差结构明显不同。可以在模型输出后加轻量校准层，不动主模型。

做法：

1. 对每个 variant pair 学一个校准函数：

$$
\hat{y}' = a_{p}\hat{y} + b_{p}
$$

其中 $p \in \{O0\text{-}O1, O0\text{-}O2, O0\text{-}O3, O1\text{-}O2, O1\text{-}O3, O2\text{-}O3\}$。

2. `a_p` 和 `b_p` 只在 validation split 上拟合。
3. 对样本少或不稳定的 pair 回退到全局校准。
4. 推理时根据 query-anchor 的 variant pair 应用校准，再进入锚点评分。

验收标准：

1. pairwise test MAE 下降。
2. 单程序 `mae_score_log` 下降。
3. strict time 指标不下降。

这一步成本低，适合先作为后处理脚本实现。

### A3. Uncertainty-aware anchor weighting

当前锚点聚合已经使用质量、距离和分类置信度，但还没有显式使用模型不确定性。可以用轻量 ensemble 或 MC dropout 给每个 query-anchor pair 一个方差估计。

做法：

1. 训练 3 到 5 个不同 seed 的 PairTransformer，或推理时打开 dropout 做 MC dropout。
2. 对每个 anchor estimate 计算均值和方差。
3. 锚点权重增加一项：

$$
w' = \frac{w}{\epsilon + \sigma^2}
$$

4. 方差过大的 anchor estimate 直接降权或剔除。

验收标准：

1. `mae_score_log` 下降。
2. strict time `band_acc_model` 提升。
3. 锚点间冲突大的样本能输出更低 confidence。

这一步尤其适合单程序输出，因为它能把“不确定”从隐藏误差变成显式信号。

### A4. Tie threshold tuning by pair type

当前三分类阈值是全局 `0.05`。但 `O2-O3` 的真实差异更小，全局阈值可能不适合所有 pair。

做法：

1. 为不同 variant pair 搜索不同 tie 阈值，例如：
   1. `O0-O2` / `O0-O3` 使用较小 tie 阈值。
   2. `O1-O2` / `O2-O3` 使用较大 tie 阈值。
2. pair 表保留 `label_class_global` 和 `label_class_pairwise` 两套标签。
3. 分类头可以训练 pair-aware label，回归头仍训练连续 log-ratio。

验收标准：

1. `O2-O3 aux_tie_recall` 提升。
2. `O2-O3 acc_3cls` 提升。
3. 不明显损害 `O0-O2`、`O0-O3` 这类远距离 pair。

这一步直接针对 near-tie 难点，成本中等。

## 3. 优先级 B：需要少量代码改动，但可能收益较大

### B1. Ordinal objective

O0/O1/O2/O3 本身有顺序结构，当前三分类只看 pair 的胜负和平局，没有显式建模“优化等级差距”。可以加入 ordinal 辅助任务。

做法：

1. 对每个 pair 计算 `abs(variant_rank_diff)`。
2. 新增一个 ordinal head 预测距离档：
   1. adjacent：`O0-O1`、`O1-O2`、`O2-O3`
   2. medium：`O0-O2`、`O1-O3`
   3. far：`O0-O3`
3. 或者直接预测 `|log_ratio|` 的离散强度：`tie / small / medium / large`。
4. 总损失变成：

$$
\mathcal{L} =
\mathcal{L}_{reg}
+ \lambda_{cls}\mathcal{L}_{cls}
+ \lambda_{ord}\mathcal{L}_{ord}
$$

验收标准：

1. 远距离 pair 不退化。
2. 近距离 pair 的方向和 tie 更稳。
3. 单程序评分的 band accuracy 不下降。

### B2. Pairwise ranking loss

当前回归头逐 pair 拟合 log-ratio，但单程序评分本质上还关心同一 program 内 O0/O1/O2/O3 的排序。可以加入排序一致性约束。

做法：

1. 对同一 program 的多个 variant score 形成小列表。
2. 用模型 pair 输出构造 latent score。
3. 增加 margin ranking loss 或 ListNet/ListMLE loss。
4. 只在同一 program 内计算，不跨 program 混排。

验收标准：

1. program 内排序准确率提升。
2. `band_accuracy` 提升。
3. strict time 方向一致率不下降。

这比纯 CE 更贴近最终“单程序评分”接口。

### B3. Time distillation head

当前 strict time 样本少，但不是没有。可以把 `score_time` 作为弱监督蒸馏头，而不是直接替代 proxy 训练目标。

做法：

1. Pairwise 主训练仍用 `cycles_per_iter`。
2. 对有 strict time 的 anchor 或 run，新增 time score head。
3. time head 只在有 `score_time` 的样本上计算 loss。
4. 使用较小权重，避免少量 noisy time label 冲掉主任务。

推荐损失：

$$
\mathcal{L} =
\mathcal{L}_{pair}
+ \lambda_{time}\mathcal{L}_{time}
$$

验收标准：

1. strict time `corr_model_time` 提升。
2. repeat-backed 子集提升或持平。
3. proxy 指标不明显下降。

这一步是把最终验收信号提前放进训练环，但需要小心数据少和标签混合来源问题。

### B4. Quality-weighted training

当前质量信号主要用于过滤和锚点质量，还没有完整进入训练权重。可以把 run 质量映射到 pair 权重。

做法：

1. 为每个 run 计算质量分：
   1. `active_window_ratio`
   2. `active_pid_count`
   3. `window_count`
   4. 是否有 repeat timing
2. pair 权重取两端 run 质量的几何均值。
3. 回归 loss 和 CE loss 都乘以这个质量权重。
4. 低质量但未过滤的样本不直接删除，而是降权。

验收标准：

1. validation / test 指标更稳。
2. 对 strict time 的相关性提升。
3. 不显著降低覆盖率。

## 4. 优先级 C：更偏研究探索

### C1. Siamese contrastive pretraining

可以先用无监督或弱监督方式学习运行表示，再做 pairwise fine tune。

可用正负样本：

1. 同一 program 的不同 variant 是 hard positive / ordered positive。
2. 不同 program 是 negative。
3. 同一 variant rank 的不同 program 可作为 weak positive。

风险是当前数据规模小，contrastive pretraining 未必稳定。因此只适合作为后续探索。

### C2. Window-token lightweight model

当前运行级摘要可能丢掉 burst 信息。可以不直接上大时序 Transformer，而是先做轻量窗口 token：

1. 每个 run 采样固定数量窗口 token。
2. 加一个小 pooling encoder。
3. 输出 run embedding 后仍进入现有 pairwise 框架。

推进条件：

1. 运行级特征扩展和质量权重都做完。
2. hard pair 分析证明错误来自阶段性 burst，而不是标签接近。

### C3. Model ensemble for production scoring

如果最终更看重单程序评分稳定性，可以不追求单模型最优，而用小 ensemble：

1. 3 个 seed 的 PairTransformer。
2. 1 个线性或树模型 baseline。
3. 评分层对多个模型输出做加权平均。

优点是稳定，缺点是推理和维护成本更高。

## 5. 推荐执行顺序

下一轮建议按下面顺序推进：

1. A2：先做 per-pair calibration，低成本修正不同 pair 的尺度偏差。
2. A3：接入 uncertainty-aware anchor weighting，让单程序输出有 confidence。
3. A4：尝试 pair-specific tie threshold，专攻 `O2-O3`。
4. A1：在新增 repeat timing 或扩展 strict time 样本后，再扩大 time-aware 网格重跑。
5. B4：把质量权重接入训练，减少低质量样本对边界的干扰。
6. B3：只有当 strict time 样本继续增加后，再做 time distillation head。
7. B1/B2：如果排序和 band 仍不稳，再加入 ordinal / ranking 目标。
8. C 类方案只在上述方法收益耗尽后再做。

当前最不建议马上做的是直接堆更大的 Transformer。原因是现在主要瓶颈不在 backbone 容量，而在 time 目标对齐、near-tie 标签、锚点聚合和样本质量。
