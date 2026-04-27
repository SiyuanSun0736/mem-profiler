# 数据问题、优化方向与优先级

> 这份文档回答两个问题：当前这批数据最主要的问题是什么，以及下一步最值得做的优化是什么。

## 1. 结论先行

当前最该优先做的不是继续加模型，而是先做语义过滤。

原因很直接：这批数据里存在一批“文件存在，但语义上无效”的 run。它们会直接污染 `cycles_per_iter`、pair 标签和 anchor 真值；如果不先清理，后面不管换更大的 Transformer、调更复杂的损失，都会先被这批坏样本拖住。

本轮调整后的落点是 [scripts/build_run_features.py](../../scripts/build_run_features.py)，而不是 [scripts/freeze_curated_manifest.py](../../scripts/freeze_curated_manifest.py)。原因是如果直接在 freeze 层按语义过滤剔除坏 run，当前四变体完整 program 数会从 145 掉到 122，训练集边界会被整体改写；放在 feature 层过滤更稳，既保留 raw/curated manifest 作为采集账本，又能阻止坏 run 进入 `pairs` 和 `anchor_set`。

## 2. 当前主要问题

### 2.1 存在语义无效 run

过滤后的摘要在 [train_set/run_feature_filter_summary.json](../../train_set/run_feature_filter_summary.json)：

1. 总运行数 580。
2. 被语义过滤剔除 71 条。
3. 其中 71 条 `active_pid_count < 5`。
4. 其中 57 条 `cycles_per_iter <= 0`。

按 variant 分布如下：

1. O0: 145 → 128
2. O1: 145 → 127
3. O2: 145 → 128
4. O3: 145 → 126

这说明问题不是集中在单一 variant，而是四个 variant 都有一批 run 实际上没有形成足够稳定的 fixed-work 语义。

### 2.2 代理标签和真实时间仍然不够一致

过滤后重新计算的外部时间验证在 [train_set/score_time_eval.json](../../train_set/score_time_eval.json)：

1. loose 对照下，proxy vs `score_time`: Pearson r = 0.3510，Spearman ρ = 0.3626。
2. loose 对照下，model vs `score_time`: Pearson r = 0.3671，Spearman ρ = 0.3431。
3. strict 时间真值下，proxy vs `score_time`: Pearson r = 0.5025，Spearman ρ = 0.6739。
4. strict 时间真值下，model vs `score_time`: Pearson r = 0.5312，Spearman ρ = 0.6026。
5. strict 过滤从 loose 的 250 行里剔除了 51 行，全部来自 `low_active_window_ratio`。

这说明时间真值过滤本身非常重要：只要把低活跃窗口占比的 run 排除掉，外部一致性会明显改善。但即便在 strict 口径下，当前 `cycles_per_iter` 和模型分数与真实时间也还只是中等相关，不是强时间监督。

### 2.3 中间 variant 仍然最难分

过滤后重训的结果在 [train_set/model_transformer_eval.json](../../train_set/model_transformer_eval.json)：

1. 整体 test: R² = 0.8060，`dir_acc = 0.8725`，`acc_3cls = 0.7542`。
2. 但 `O2-O3` 仍然最难：`dir_acc = 0.6667`，`acc_3cls = 0.35`，R² 仍为负。

这说明清掉坏样本以后，主问题不再是“全局训练不稳”，而是“相近优化级别之间的差异本来就很小，且 tie 密集”。

### 2.4 过滤后样本规模明显收缩

语义过滤后的下游产物为：

1. [train_set/run_features.parquet](../../train_set/run_features.parquet): 509 runs。
2. [train_set/pairs.parquet](../../train_set/pairs.parquet): 1494 pairs，129 个程序。
3. [train_set/anchor_set.parquet](../../train_set/anchor_set.parquet): 250 anchors，128 个拥有 O0 基线的程序。

这说明过滤是必要的，但也带来了新的现实：当前可用于稳定建模的数据比“145 x 4”想象中更少。

### 2.5 仍然存在死特征或弱特征

本轮特征构建仍然报告零方差列 `minor_fault_ratio`。这一列现在仍保留在 [train_set/run_features.parquet](../../train_set/run_features.parquet) 和 [train_set/run_features_zscore.parquet](../../train_set/run_features_zscore.parquet) 中用于账本和兼容性，但已经从 pair / anchor / model / score 的实际输入列里剔除，不再参与训练和推理。

## 3. 最值得做的优化

### P1. 语义过滤，不是加模型

这是当前第一优先级，且本轮已经实施。

过滤规则默认接在 [scripts/build_run_features.py](../../scripts/build_run_features.py) 中：

1. `active_pid_count < 5` 的 run 直接剔除。
2. `cycles_per_iter <= 0` 的 run 直接剔除。
3. 过滤摘要写入 [train_set/run_feature_filter_summary.json](../../train_set/run_feature_filter_summary.json)。

这一步带来的直接收益是：

1. Transformer test R² 从之前的 0.5869 提升到 0.8060。
2. Transformer test `dir_acc` 从 0.7929 提升到 0.8725。
3. 单程序评分 Pearson r 从 0.8567 提升到 0.9160。
4. 单程序评分方向准确率从 0.5966 提升到 0.7320。

所以当前最强结论不是“模型该换”，而是“坏 run 会显著拖坏整个链路”。

### P2. 补真实时间口径，而不是继续完全依赖 proxy

这一项本轮已经先完成了第一步：

1. 在 [scripts/build_time_score_table.py](../../scripts/build_time_score_table.py) 里加入 strict 时间真值过滤。
2. 默认要求 `active_window_ratio >= 0.10`，并输出 [train_set/time_score_filter_summary.json](../../train_set/time_score_filter_summary.json)。
3. 在 [scripts/evaluate_score_vs_time.py](../../scripts/evaluate_score_vs_time.py) 里同时保留 strict 主统计和 loose 对照统计。

但第二优先级还没有做完，下一步仍然应该是时间真值增强：

1. 对关键程序补 fixed-work repeat timing。
2. 用中位数 wall time 构建更稳的 `score_time`。
3. 用它做更严格的外部验证，而不是只看 proxy 内部相关性。

### P3. 对接近 tie 的 pair 做专门处理

这一项本轮已经做了第二步实现，落点同时在 [scripts/train_transformer.py](../../scripts/train_transformer.py) 和 [scripts/score_program.py](../../scripts/score_program.py)：

1. 对 `|log_ratio|` 做三档分桶：`tie`、`near_tie`、`far`。
2. 对回归头引入 tie-aware weighting：默认 `tie=0.35`、`near_tie=0.65`、`far=1.0`。
3. 将原来的“单头回归 + 可选方向 BCE”改成“回归头 + 3 类辅助头（i_better / tie / j_better）”。
4. 在单程序评分阶段，不再直接使用回归头裸输出，而是让辅助分类头参与近 tie 解码和 gating：高 `p_tie` 的 pair 会被压缩到接近 0，非 tie pair 的方向也由辅助头参与约束。

当前这版模型的结果在 [train_set/model_transformer_eval.json](../../train_set/model_transformer_eval.json)：

1. test 集回归主头：`dir_acc = 0.8775`，`acc_3cls = 0.7833`。
2. test 集辅助分类头：`aux_acc_3cls = 0.8292`，`aux_tie_recall = 0.5833`。
3. `O2-O3` 上，回归主头 `acc_3cls = 0.450`，辅助分类头 `aux_acc_3cls = 0.600`。
4. `O1-O2` / `O1-O3` 上，辅助分类头也继续高于回归主头。

但这一步目前还不是“直接替换主模型”的终点，因为它带来了一个新的权衡：

1. 辅助分类头用于 gating 后，单程序评分已经明显回升，不再像第一版 P3 那样直接拖坏下游评分。
2. 但 strict 时间外部验证在 [train_set/score_time_eval.json](../../train_set/score_time_eval.json) 中仍只有 `corr_model_time = 0.4310`，说明“近 tie 解码变稳”还没有完全等价于“更接近真实时间”。

所以当前更准确的判断不是“P3 已经完成”，而是：

P3 的方向是对的，辅助分类头已经不仅在训练里学到了近 tie pair，也开始参与下游解码；但还需要继续调权重、阈值或推理解码，才能把这部分收益稳定迁移到真实时间口径。

### P4. 改锚点策略，而不是只用 O0/O3 平均

这一项本轮已经做了首版实现，落点在 [scripts/build_anchor_set.py](../../scripts/build_anchor_set.py) 和 [scripts/score_program.py](../../scripts/score_program.py)：

1. 默认锚点从 `O0/O3` 改成 `O0/O2/O3`。
2. 在 anchor set 中加入 `active_window_ratio` 和 `anchor_quality`，用于质量加权。
3. 单程序评分从简单平均改成“质量权重 × variant 距离权重 × 分类置信度”的加权聚合。
4. 对同一程序的多个 anchor estimate 做中位数离群过滤，明显偏离的 anchor 不参与最终聚合。

当前锚点统计在 [train_set/anchor_set.stats.json](../../train_set/anchor_set.stats.json)：

1. 锚点总数从 250 增加到 374。
2. 当前锚点集合为 O0=128、O2=124、O3=122。
3. `anchor_quality_mean = 0.5043`。

这一步带来的直接结果在 [train_set/score_eval.json](../../train_set/score_eval.json)：

1. `n_with_gt = 374`，覆盖面明显高于只用 O0/O3 时的 250。
2. `mae_score_log = 0.3165`。
3. `corr_score_log = 0.8999`。
4. `band_accuracy = 0.8075`。

所以 P4 的判断已经可以更明确一些：

只用 O0/O3 平均确实过于脆弱；把 O2 拉进来，并且按质量和离群情况做加权，是比简单平均更稳的默认策略。

### P5. 清理死特征并引入样本质量权重

这一项本轮已经做完第一步：

1. 新增 [scripts/feature_columns.py](../../scripts/feature_columns.py) 作为共享输入特征列表。
2. `minor_fault_ratio` 由于在当前快照里 `std = 0`，已经从 pair / anchor / model / score 的输入列里统一剔除。
3. [train_set/pairs_stats.json](../../train_set/pairs_stats.json) 现在显示 `feature_dim = 53`、`input_dim = 159`，说明输入维度已经同步收缩。

这一项还没有做完的部分是：

1. 更系统地识别“长期弱信号特征”，而不只处理当前已知的零方差列。
2. 把 `active_pid_count`、`active_window_ratio`、`window_count` 这类质量信号更完整地接入训练采样权重，而不只用于时间真值过滤和锚点质量加权。

## 4. 当前优先级顺序

建议的执行顺序如下：

1. 先保留并稳定语义过滤。
2. 再重建 run_features / pairs / anchor_set / model / scores 全链路。
3. 然后补时间真值与 score_time 外部验证。
4. 再对 O2-O3 这类近邻 pair 做 tie-aware 训练改造。
5. 最后再考虑锚点策略和特征裁剪。

## 5. 当前判断

到这一步，问题已经不是“这套流程能不能跑通”，而是“哪些 run 值得信、哪些 pair 天然难、哪些目标和真实时间还有偏差”。

所以当前最值钱的优化方向非常明确：

先让数据语义正确，再讨论模型复杂度。