# 数据问题、优化方向与优先级

> 这份文档回答两个问题：当前这批数据最主要的问题是什么，以及下一步最值得做的优化是什么。

完整问题样本清单、缺失 strict baseline 的 run，以及 O2/O3 难例分流建议，见 [current-data-quality-audit.md](current-data-quality-audit.md)。

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

完整 71 条 run 名单见 [current-data-quality-audit.md](current-data-quality-audit.md) 和 [train_set/data_quality_audit.json](../../train_set/data_quality_audit.json)。

### 2.2 代理标签和真实时间仍然不够一致

过滤后重新计算的外部时间验证在 [train_set/score_time_eval.json](../../train_set/score_time_eval.json)：

1. 未做 strict 过滤时，proxy vs `score_time`: Pearson r = 0.2993，Spearman ρ = 0.2756。
2. 未做 strict 过滤时，model vs `score_time`: Pearson r = 0.2799，Spearman ρ = 0.2701。
3. 当前 strict 主统计为 361 行，其中 `proxy_strict = 286`、`repeat_timing = 75`。
4. strict 时间真值下，proxy vs `score_time`: Pearson r = 0.4415，Spearman ρ = 0.5635。
5. strict 时间真值下，model vs `score_time`: Pearson r = 0.3982，Spearman ρ = 0.5219。
6. `repeat_backed_only` 子集 75 行上，proxy vs `score_time`: Pearson r = 0.8275，model vs `score_time`: Pearson r = 0.7879。
7. strict 过滤从 loose 的 374 行里只剔除了 13 行，全部来自 `low_active_window_ratio`；缺 strict O0 baseline 的问题已被 repeat timing 覆盖到 0。

这说明现在的 P2 已经不只是“把低活跃窗口 run 过滤掉”，而是开始把一部分 `score_time` 升级成更强的 fixed-work repeat timing 真值。新的 repeat-backed 子集信号明显更强，但整体 strict 主口径里仍有 286 行依赖 `proxy_strict`，所以当前整体口径依然只是“中等强度时间监督”，还不是纯 wall-time 真值。

### 2.3 中间 variant 仍然最难分

过滤后重训的结果在 [train_set/model_transformer_eval.json](../../train_set/model_transformer_eval.json)：

1. 整体 test: `R² = 0.8069`，`dir_acc = 0.9020`，`acc_3cls = 0.7958`，`aux_acc_3cls = 0.8417`。
2. 但 `O2-O3` 仍然最难：`R² = -1.0912`，`acc_3cls = 0.4500`，`aux_tie_recall = 0.4545`。
3. 从全量 pair 分布看，`O2-O3` 的 tie rate 已达 `0.4524`，中位 `|log_ratio|` 只有 `0.0587`，说明这里本来就是近似平局最密集的区间。

这说明清掉坏样本以后，主问题不再是“全局训练不稳”，而是“相近优化级别之间的差异本来就很小，且 tie 密集”。

### 2.4 过滤后样本规模明显收缩

语义过滤后的下游产物为：

1. [train_set/run_features.parquet](../../train_set/run_features.parquet): 509 runs。
2. [train_set/pairs.parquet](../../train_set/pairs.parquet): 1494 pairs，129 个程序。
3. [train_set/anchor_set.parquet](../../train_set/anchor_set.parquet): 374 anchors，128 个拥有 O0 基线的程序。
4. 过滤后的 run_features 实际只保留了 122 个完整四变体程序，另有 10 个程序缺至少一个 variant。

这说明过滤是必要的，但也带来了新的现实：当前可用于稳定建模的数据比“145 x 4”想象中更少。

### 2.5 仍然存在死特征或弱特征

本轮特征构建仍然报告零方差列 `minor_fault_ratio`。这一列现在仍保留在 [train_set/run_features.parquet](../../train_set/run_features.parquet) 和 [train_set/run_features_zscore.parquet](../../train_set/run_features_zscore.parquet) 中用于账本和兼容性，但已经从 pair / anchor / model / score 的实际输入列里剔除，不再参与训练和推理。

## 3. 已完成的优化

### 3.1 已完成：P1 语义过滤

这一项已经落在 [scripts/build_run_features.py](../../scripts/build_run_features.py) 并成为默认行为：

1. `active_pid_count < 5` 的 run 直接剔除。
2. `cycles_per_iter <= 0` 的 run 直接剔除。
3. 过滤摘要写入 [train_set/run_feature_filter_summary.json](../../train_set/run_feature_filter_summary.json)。

当前收益已经比较稳定：

1. Transformer test 结果稳定在 `R² = 0.8069`、`dir_acc = 0.9020`、`acc_3cls = 0.7958`。
2. 单程序评分结果稳定在 `corr_score_log = 0.9072`、`band_accuracy = 0.8021`。
3. 当前已经可以明确判断：坏 run 会直接污染训练和评分链路，先做语义过滤是对的。

所以 P1 在“把坏样本挡在训练链路外”这个目标上已经完成；它不再是当前主变量，除非后续数据采集口径发生变化。

### 3.2 已完成：P2 第二阶段首版，repeat timing 已接入

这一项现在已经不只是 strict 输入过滤，而是推进到了第二阶段首版，落点在 [scripts/collect_repeat_timing.py](../../scripts/collect_repeat_timing.py)、[scripts/build_time_score_table.py](../../scripts/build_time_score_table.py) 和 [scripts/evaluate_score_vs_time.py](../../scripts/evaluate_score_vs_time.py)：

1. strict 时间真值过滤仍然保留 `active_window_ratio >= 0.10`。
2. [scripts/collect_repeat_timing.py](../../scripts/collect_repeat_timing.py) 已能为指定 run 目录写出 `repeat_timing.json`。
3. [scripts/build_time_score_table.py](../../scripts/build_time_score_table.py) 现在会优先使用 `repeat_timing.json` 中的中位数 wall time；缺失时再回退到原有 proxy strict 口径。
4. [scripts/evaluate_score_vs_time.py](../../scripts/evaluate_score_vs_time.py) 现在会额外输出 `repeat_backed_only` 子集统计。

当前这一步带来的直接结果在 [train_set/time_score_filter_summary.json](../../train_set/time_score_filter_summary.json) 和 [train_set/score_time_eval.json](../../train_set/score_time_eval.json)：

1. `time_scores` 现在有 `n_valid_strict = 481`，其中 `n_preferred_repeat = 100`，`n_rescued_by_repeat_timing = 91`。
2. 当前 strict 主统计为 361 行，其中 `proxy_strict = 286`、`repeat_timing = 75`。
3. strict 主统计上，proxy vs `score_time` 为 Pearson `0.4415`、Spearman `0.5635`；model vs `score_time` 为 Pearson `0.3982`、Spearman `0.5219`。
4. `repeat_backed_only` 子集 75 行上，proxy vs `score_time` 提升到 Pearson `0.8275`，model vs `score_time` 提升到 Pearson `0.7879`。

所以 P2 现在应当视为“第二阶段已经开始落地”：repeat timing 已经实打实补进来了，而且成功把 91 行从原来的 strict 缺口里救回来。但它还没有完全完成，因为 strict 主口径里仍有大量样本依赖 `proxy_strict`，剩余 13 行也还没有处理完。

### 3.3 已完成：P3 当前阶段，tie-aware 训练与评分层默认值回退

这一项当前已经不是“只有想法”，而是已经完成了一个可工作的阶段版本，落点在 [scripts/train_transformer.py](../../scripts/train_transformer.py)、[scripts/score_program.py](../../scripts/score_program.py) 和 [scripts/tune_score_program_fine.py](../../scripts/tune_score_program_fine.py)：

1. `|log_ratio|` 已分成 `tie`、`near_tie`、`far` 三档。
2. 回归头已接入 tie-aware weighting。
3. 模型已从“单头回归 + 可选方向 BCE”推进到“回归头 + 3 类辅助头（i_better / tie / j_better）”。
4. 单程序评分阶段已接入辅助分类头 gating。
5. 锚点投票阶段已接入 `tie_margin_weight_alpha`，下调近 tie directional pair 的影响。
6. [scripts/tune_score_program_fine.py](../../scripts/tune_score_program_fine.py) 已能按 query variant 精调 `gate / shrink / alpha / min_anchor_quality / outlier`。
7. [scripts/score_program.py](../../scripts/score_program.py) 已实现 tuned 可靠性回退：`CLI 显式传参 > 当前 variant tuned best > ALL tuned best > 代码硬编码默认值`。

当前产物已经说明这一阶段确实跑通：

1. [train_set/model_transformer_eval.json](../../train_set/model_transformer_eval.json) 中，test 集主头 `dir_acc = 0.9020`、`acc_3cls = 0.7958`，辅助头 `aux_acc_3cls = 0.8417`。
2. `O2-O3` 上，辅助分类头 `aux_acc_3cls = 0.5000`，仍高于或持平于回归主头 `0.4500`。
3. [train_set/score_eval.json](../../train_set/score_eval.json) 中，默认 score-first 结果为 `mae_score_log = 0.2726`、`corr_score_log = 0.9072`、`band_accuracy = 0.8021`。
4. [08-score-selection-objective-comparison.md](08-score-selection-objective-comparison.md) 已确认当前默认建议口径为 score-first，time-first 作为可切换选项保留。

所以 P3 当前应当视为“阶段性完成”：训练端 tie-aware、评分层 gating、variant-local tuned 和可靠性回退都已经落地。

### 3.4 已完成：P4 O0/O2/O3 加权锚点底座

这一项已经完成首版实现，落点在 [scripts/build_anchor_set.py](../../scripts/build_anchor_set.py) 和 [scripts/score_program.py](../../scripts/score_program.py)：

1. 默认锚点已从 `O0/O3` 改成 `O0/O2/O3`。
2. anchor set 已加入 `active_window_ratio` 和 `anchor_quality`。
3. 单程序评分已从简单平均改成“质量权重 × variant 距离权重 × 分类置信度”的加权聚合。
4. 同一程序的多个 anchor estimate 已接入中位数离群过滤。

当前统计在 [train_set/anchor_set.stats.json](../../train_set/anchor_set.stats.json)：

1. 锚点总数从 250 增加到 374。
2. 当前锚点集合为 O0=128、O2=124、O3=122。
3. `anchor_quality_mean = 0.5043`。

对应评分结果在 [train_set/score_eval.json](../../train_set/score_eval.json)：

1. `n_with_gt = 374`，覆盖面明显高于只用 O0/O3 时的 250。
2. `mae_score_log = 0.2726`。
3. `corr_score_log = 0.9072`。
4. `band_accuracy = 0.8021`。

所以 P4 当前也应视为已完成的基础设施，而不是待验证想法。后续即使继续推进 P3，P4 这层 O0/O2/O3 加权锚点仍然是当前单程序评分成立的底座。

### 3.5 已完成：P5 第一阶段，死特征清理已经接上

这一项的第一阶段已经完成：

1. 新增 [scripts/feature_columns.py](../../scripts/feature_columns.py) 作为共享输入特征列表。
2. `minor_fault_ratio` 已从 pair / anchor / model / score 的实际输入列里统一剔除。
3. [train_set/pairs_stats.json](../../train_set/pairs_stats.json) 当前显示 `feature_dim = 53`、`input_dim = 159`。

所以 P5 的“先清掉已知死特征”这一层已经做完，当前不需要再把它表述成待开始事项。

## 4. 尚未完成的优化

### 4.1 未完成：P2 第二阶段，补更强的时间真值

P2 还没有完成的部分，已经不再是“有没有 repeat timing”，而是“剩下哪些样本还值得继续补、哪些应该继续排除”：

1. 当前 strict 主统计还剩 13 行未纳入，全部是 `low_active_window_ratio`。
2. 这 13 行分布在 8 个程序上，不是同一类问题。
3. 其中 [train_set/time_scores.parquet](../../train_set/time_scores.parquet) 显示，`BitBench_uudecode` 的 `O1/O2` 仍未补 repeat timing，但已有 strict baseline，属于仍可被定向救回的样本。
4. `FreeBench_pcompress2` 和 `Prolangs-C_gnugo` 已尝试补 repeat timing，但当前 sidecar 结果是 `11/11` 失败或只在单一 variant 上成功，说明问题已经不是“时间不稳”，而是 `run_cmd` 需要先单独排查。
5. 其余程序如 `BitBench_uuencode`、`MiBench_security-sha`、`Prolangs-C++_city`、`mediabench_gsm_toast`、`mediabench_mpeg2_mpeg2dec` 目前既没有有效 repeat timing，也缺完整 strict baseline，更适合继续排除在 strict 主口径之外。

所以 P2 当前更合理的未完成项不是“大范围继续补采”，而是：

1. 只对像 `BitBench_uudecode O1/O2` 这类已有 strict baseline、且补采后可直接救回的样本继续做定向 repeat timing。
2. 对 `FreeBench_pcompress2`、`Prolangs-C_gnugo` 这类 `11/11` 命令失败的程序，先查 `run_cmd` 和运行环境，再决定是否重试。
3. 对其余缺基线且无有效 sidecar 的样本，当前继续从 strict 主口径排除。

### 4.2 未完成：P3 收敛阶段，只继续攻 O2/O3

P3 当前虽然已完成一个可工作阶段，但还没有完全收敛：

1. `O2` / `O3` 的 variant-local tuned best 已经有稳定信号。
2. `O0` / `O1` 仍然只能依赖 `ALL` 回退，局部 tuned 不可靠。
3. 当前最难的仍是 `O2-O3`，tie rate 高、`|log_ratio|` 小，天然就是最密集的 near-tie 区间。

所以 P3 未完成的重点不是“继续普遍调参”，而是继续补 `O2/O3` 的 near-tie 数据与 fixed-work timing，只在真正有信号的区间继续收敛。

### 4.3 未完成：P5 第二阶段，把质量信号正式接入训练

P5 还没做完的部分主要有两块：

1. 更系统地识别长期弱信号特征，而不只处理当前已知的零方差列。
2. 把 `active_pid_count`、`active_window_ratio`、`window_count` 这类质量信号更完整地接入训练采样权重，而不只用于时间真值过滤和锚点质量加权。

这部分目前还停留在方向明确、实现未落地的状态。

## 5. 改进策略

下一轮更合理的改进策略，不是同时重开 P1 到 P5，而是按“先稳底座，再补真值，最后再做模型侧细化”的顺序推进：

1. 先把 P1 和 P4 当成当前稳定底座，不要反复改动语义过滤规则和默认锚点策略，除非数据采集口径发生变化。
2. 把 P2 当成当前主线任务，优先补关键程序的 fixed-work repeat timing 和中位数 wall time，提升 `score_time` 的真实性。
这一步现在应收缩成定向补强：优先补仍可救回的少量样本，而不是继续大范围盲目 repeat。
3. 把 P3 的后续工作严格收缩到 `O2/O3` 这类 near-tie 高密度区间，不再把 `O0/O1` 的局部 tuned 最优当成主要优化目标。
4. 把 P5 的后续工作做成受控增量：先做特征审计，再引入质量权重，避免一次性同时改模型、改输入、改采样。
5. 每次改动都以 `run_features / pairs / anchor_set / model / scores / score_time_eval` 这条链路可重建为前提，并同时回看 proxy 与 strict time 两套指标，避免只靠单一指标决策。

## 6. 当前优先级顺序

建议的执行顺序如下：

1. 先保持 P1 语义过滤和 P4 锚点策略稳定，确保整条重建链路可重复。
2. 然后补强 P2：对关键程序补 fixed-work repeat timing，用中位数 wall time 做更强的 `score_time` 真值。
这里的“关键程序”现在更明确地指向两类：一类是 `BitBench_uudecode` 这种已有 strict baseline、补采即可救回的样本；另一类是 `FreeBench_pcompress2` / `Prolangs-C_gnugo` 这类需要先修 `run_cmd` 的失败样本。
3. 再继续推进 P3，但只聚焦 `O2` / `O3` 的 near-tie 区间，结合新增时间真值做局部收敛。
4. 在数据和评分底座更稳之后，再推进 P5 第二阶段，把质量信号接入训练采样权重。
5. 最后再做更系统的弱特征审计，而不是一开始就继续加模型复杂度。

## 7. 当前判断

到这一步，问题已经不是“这套流程能不能跑通”，而是“哪些部分已经可以当默认底座、哪些部分还值得继续投入”。

当前更准确的判断是：

1. P1、P4 和 P5 第一阶段已经属于已完成基础设施。
2. P2 第一阶段和 P3 当前阶段也已经落地，但都还有一个明确的后半程要补。
3. P2 已经从“strict 过滤”推进到“repeat-backed 时间真值”，但剩余 13 行不该再一概而论：少数样本值得继续补采，其余则应继续排除在 strict 主口径之外。
4. 下一轮最值钱的投入不是继续扩模型，而是把 repeat-backed 时间真值继续做实，并只在 `O2/O3` 这种真正困难且有信号的区间继续收敛。

所以当前最合理的主线非常明确：

先稳住已经完成的底座，再把时间真值和 near-tie 难例做实，然后再讨论进一步的模型复杂度。
