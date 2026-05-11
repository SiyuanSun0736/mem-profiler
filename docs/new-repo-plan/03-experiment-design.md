# 实验设计方案

> 这份文档回答“基于当前这批 O0-O3 数据，接下来最应该做哪些实验，哪些暂时不要做”。

## 1. 总体原则

实验设计必须先服务当前数据，而不是服务未来想做的数据集。当前实验的第一目标不是证明布局优化，而是证明这批 non-time 摘要特征在程序级 held-out 划分上确实有恢复优化方向的能力。

因此，第一阶段只保留下面四条链路：

1. pairwise non-time 回归
2. pairwise non-time 三分类
3. 单程序锚点评分
4. 评分结果绑定弱诊断证据

## 2. 当前实验矩阵

### 2.1 数据维度

当前真实可用的数据维度只有：

1. `program`
2. `variant ∈ {O0,O1,O2,O3}`
3. `window_id`
4. `machine_id` 固定为单机

没有 `repeat_id`，没有多机维度，也没有布局类别标签。

### 2.2 当前最重要的划分方式

1. train / val / test 必须按 `program` 划分。
2. 所有 pair 只能在同一个 `program` 内部构造。
3. 所有结论默认只对“未见过程序上的 O0-O3 代理任务”成立。

### 2.3 当前最重要的任务形式

1. pairwise log-ratio 回归
2. pairwise 三分类：`i_better / tie / j_better`
3. 单程序 anchor-based scoring
4. 证据绑定：热点窗口 / 热点实体 / 瓶颈类别

## 3. 当前推荐的基线顺序

### Baseline 0. 朴素名义排序基线

直接利用 `O0 < O1 < O2 < O3` 的名义顺序给出方向预测。这不是目标模型，而是必须存在的 sanity check，用来防止把“优化级名称先验”误写成“non-time 学到了东西”。

### Baseline 1. 线性或树模型

建议至少包含：

1. Ridge / Linear Regression 做 log-ratio 回归
2. Logistic Regression 做三分类或方向判断
3. 可选 XGBoost 作为更强的非深度基线

### Baseline 2. MLP 拼接模型

用当前 53 维 non-time 主输入做 `[x_i; x_j; x_i - x_j]` 输入，验证非线性组合是否有效。

### Baseline 3. PairTransformer

把每个运行摘要当成一个 token，通过共享投影和显式差分做比较。这是当前最合理的“稍强结构”而不是最终结构。

## 4. 当前阶段不建议优先做的实验

1. 不优先做 full 视图，因为当前 `wall_time_sec` 不是真正有区分度的主标签输入。
2. 不优先做 time-only 学习，因为 `total_cycles` 本身已经是标签代理，直接喂入只会变成近似泄漏上界。
3. 不优先做窗口级时序 Transformer，因为当前运行级特征还没有把原始字段用尽。
4. 不优先做 repeat 稳定性实验，因为现有数据不支持。
5. 不优先做跨机器泛化，因为现有数据不支持。

## 5. 当前应使用的评价指标

### 5.1 pairwise 指标

1. MAE
2. RMSE
3. R²
4. `dir_acc`
5. `acc_3cls`
6. `aux_acc_3cls`
7. `aux_tie_recall`
8. `val_reg_loss`
9. `val_aux_class_loss`

其中 `acc_3cls` 来自回归头输出按阈值离散后的结果，`aux_acc_3cls` 和 `aux_tie_recall` 来自三分类辅助头。二者必须分开报告：前者衡量连续 log-ratio 头是否天然学到方向边界，后者衡量交叉熵辅助目标是否真的改善 `i_better / tie / j_better` 判别。

### 5.2 单程序评分指标

1. `mae_score_log`
2. Pearson 或 Spearman 相关系数
3. `dir_accuracy`
4. `band_accuracy`

### 5.3 解释输出指标

当前没有人工标注真值，因此解释层只应评估“覆盖率”和“可回指性”，不应假装有强监督指标。

建议至少记录：

1. 有多少评分结果附带热点窗口证据
2. 有多少评分结果附带一级瓶颈类别
3. 支持特征是否能回指到原始窗口或热点实体

## 6. 当前必须完成的实验

### E1. 程序级 held-out pairwise 实验

目的：回答 non-time 摘要特征是否能在未见程序上恢复方向。

输出：

1. MLP 和 PairTransformer 的 train/val/test 指标
2. 与朴素 O-rank 基线的对比
3. 回归-only 与“回归 + 三分类交叉熵辅助头”的目标函数消融

当前推荐把“回归 + 三分类交叉熵”作为默认训练目标，而不是只作为附加实验。消融的目的不是重新证明要不要分类头，而是确认当前 `aux_class_lambda`、类别权重和 tie 阈值是否仍然合适。

建议至少比较：

1. `L_reg`：只训练 log-ratio 回归头。
2. `L_reg + λ_cls L_CE`：回归头加三分类交叉熵辅助头。
3. 可选 `L_reg + λ_cls L_CE + λ_dir L_BCE`：只在二分类方向 BCE 有额外收益时保留。

验收时不要只看 `MAE` 或 `R²`。如果交叉熵辅助头有效，它应至少带来下面一种收益：

1. `aux_acc_3cls` 或 `aux_tie_recall` 提升。
2. `O1-O2` / `O2-O3` 近邻 pair 的 `acc_3cls` 或 tie 召回提升。
3. 单程序评分阶段的 gating 更稳，表现为 `band_accuracy` 或 strict time 指标不下降。

当前已完成首轮消融，汇总见 [train_set/objective_ablation.md](../../train_set/objective_ablation.md)。在 `aux_class_lambda ∈ {0, 0.05, 0.10, 0.20, 0.30}` 中，`0.05` 的综合选择分最高：test `aux_acc_3cls = 0.8417`、`aux_tie_recall = 0.7222`、`O2-O3 aux_acc_3cls = 0.6000`，同时 test `MAE = 0.5677`、`R² = 0.8031`。当前主模型已经按 `aux_class_lambda=0.05` 重训，并已同步重跑评分层 gating 和 strict time 评估。

### E2. 各变体对细分实验

目的：确认哪些 pair 容易，哪些 pair 难。

输出：

1. `O0-O1`
2. `O0-O2`
3. `O0-O3`
4. `O1-O2`
5. `O1-O3`
6. `O2-O3`

这一步尤其重要，因为当前结果已经显示 `O2-O3` 这种近邻 pair 更难。

### E3. 单程序锚点评分实验

目的：验证成对模型能否被锚点法稳定转成单程序分数。

输出：

1. 当前默认 `O0/O2/O3` 加权多锚点结果
2. 与 `O0/O3` 双锚点对照结果
3. 分数与 `score_gt` 的相关性和档位准确率

### E4. 特征扩展实验

目的：验证当前已经接入的第一轮扩展特征是否真正带来增益，并识别还值得继续补的剩余字段。

输出：

1. 当前 53 维主输入整体结果
2. 去掉 fault subtype 子集后的对照
3. 去掉 mm syscall 子集后的对照
4. 去掉 warmup/steady-state 子集后的对照
5. 若继续补列，再增加“当前 53 维 + 剩余字段”的增量实验

### E5. 诊断绑定实验

目的：让单程序评分不只是分数。

输出：

1. score + band
2. Top bottleneck
3. 对应热点窗口或热点实体证据
4. 对应支持特征摘要

### E6. 交叉熵之后的单程序优化实验

目的：在当前 CE 辅助头已经有效的基础上，继续优化单程序评分，尤其是 strict time 外部验证。

第一批实验应优先覆盖：

1. time-aware scoring fine tune
2. per-pair calibration
3. uncertainty-aware anchor weighting
4. pair-specific tie threshold

这些方案的详细定义、优先级和验收标准见 [09-optimization-ideas-after-ce.md](09-optimization-ideas-after-ce.md)。

## 7. 当前结果对实验设计的启示

已有结果已经给出三条非常具体的设计约束：

1. Transformer 在方向判断上明显优于 MLP，因此当前实验不该只盯回归误差。
2. 单程序评分对 proxy 真值已经较稳，但对 strict 时间真值仍只有中等相关，因此锚点设计和证据绑定必须成为主实验，而不是附录。
3. `O2-O3` 和 `O1-O2` 这类近邻 pair 更难，因此下一轮应优先做难样本分析，而不是继续追求更大的 backbone。

## 8. 报告组织建议

建议按下面顺序组织实验报告：

1. 先描述真实数据边界
2. 再给 held-out pairwise 结果
3. 再给各变体对细分结果
4. 再给单程序锚点评分结果
5. 最后给热点窗口和瓶颈证据案例

这个顺序的核心目的是防止再次出现“方案写得像最终研究，结果其实只是代理任务”的错位。
