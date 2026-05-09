# 数据现状与样本设计

> 这份文档回答“这批真实数据到底长什么样，现阶段应该按什么样本单位和标签来建模”。

## 1. 当前数据快照

现有数据不是从零设计的新数据集，而是已经落盘的 llvm-test-suite 采集结果。但这里的“当前数据”实际上有两个层次：原始采集快照和现有训练快照。文档如果不把两者拆开，就会把 raw 层的重复采样误写成已经冻结好的训练视图。

### 1.1 原始采集快照

基于 `manifest_bcc_*.jsonl`、目录结构和样本 JSONL，可以确认下面这些事实：

1. 原始数据根目录是 `data/llvm_test_suite/bcc/O0~O3`。
2. 当前 `manifest_bcc_O0~O3.jsonl` 已在采集收尾自动去重后收敛为 `145/145/145/145`。
3. 当前采集脚本会在单 variant 收尾阶段自动调用 `dedup_dataset_variant.py`，按 `valid -> total_samples -> window_lines -> completion_count` 规则去重，并重写单 variant manifest。
4. 如需进一步冻结一版更严格的训练快照，可以再由 `freeze_curated_manifest.py` 生成 `manifest_curated_O0~O3.jsonl`；当前 full/curated 账本都应视为 `145 x 4` clean ledger，而不再继续沿用更早一轮重复 raw manifest 的旧口径。
5. 单次 raw run 目录至少包含两类文件：`run_metadata.jsonl` 和 `window_metrics.jsonl`。从真实样本可以看到，`run_metadata.jsonl` 已记录 `enabled_probes`、`host_info`、`aggregation_scope` 以及 `collection_backend=hybrid_perf_event_open_bcc`。
6. 当前 raw run 仍然是单机、约 60 秒观测、1.0 秒窗口、按 PID 聚合，没有 repeat 和多机维度。

### 1.2 冻结训练快照

基于 `train_set/pairs_stats.json`、`train_set/anchor_set.stats.json` 和 `train_set/feature_scaler.json`，可以确认当前训练闭环使用的是另一个冻结视图：

1. full/curated 账本覆盖 145 个 program。
2. full/curated 账本共有 580 条运行级记录，也就是每个 program 保留 4 个 variant。
3. `build_run_features.py` 默认会在特征构建阶段做语义过滤，当前保留 509 条运行级样本。
4. 当前 `pairs.parquet` 有 1494 条 pair。
5. 当前 `anchor_set.parquet` 有 374 条 anchor。
6. 现有模型评估、pair 统计和单程序评分结果，默认都基于这个“580 条账本 -> 509 条训练子集”的过滤后闭环，而不是旧版重复 manifest 或未过滤全量账本。
7. 当前 `run_features` 与 `output_dir` 口径已经和现有 curated manifests 对齐；账本层与训练层的差异，主要来自语义过滤而不是“更早一轮 snapshot 混用”。

### 1.3 这对建模意味着什么

1. 第一阶段建模对象应明确为“过滤后训练子集上的单次运行、单机、程序内相对比较”，而不是未过滤账本全量样本。
2. 如果要从当前 manifests 重建 `train_set`，默认链路应是 `manifest_curated -> build_run_features 语义过滤 -> pairs / anchor_set`，而不再是假设 manifests 自带重复样本和缺失目录。
3. 因此本阶段的主要变量应继续放在特征增益验证、质量过滤和时间外部验证，而不是继续讨论账本层是否天然等价于 `145 x 4` clean dataset。

## 2. 原始文件层

每次运行至少包含两类文件：

1. `run_metadata.jsonl`：运行起止时间、目标进程、窗口大小、采样率、后端信息，以及 probe 开关和主机信息。
2. `window_metrics.jsonl`：按 `window_id × pid` 聚合后的窗口级指标。

当前真实数据里，`window_metrics.jsonl` 已经提供了比当前训练特征更多的字段，因此后续扩展特征时不应先重采数据，而应先吃透现有原始字段。

## 3. 当前可直接使用的原始字段

从现有 `window_metrics.jsonl` 看，已经存在下面几组原始量：

### 3.1 PMU 计数量

1. `cycles`
2. `instructions`
3. `llc_loads`
4. `llc_load_misses`
5. `llc_stores`
6. `llc_store_misses`
7. `dtlb_loads`
8. `dtlb_load_misses`
9. `dtlb_stores`
10. `dtlb_store_misses`
11. `dtlb_misses`
12. `itlb_load_misses`

### 3.2 page fault 及 fault 子类型

1. `minor_faults`
2. `major_faults`
3. `anon_faults`
4. `file_faults`
5. `shared_faults`
6. `private_faults`
7. `write_faults`
8. `instruction_faults`

### 3.3 内存相关系统调用统计

1. `mmap_calls`
2. `munmap_calls`
3. `mprotect_calls`
4. `brk_calls`
5. `mmap_bytes`
6. `munmap_bytes`
7. `mprotect_bytes`
8. `brk_growth_bytes`
9. `brk_shrink_bytes`

### 3.4 其他辅助量

1. `samples`
2. `lbr_samples`
3. `lbr_entries`

需要强调的是：当前训练链路已经消费了其中相当一部分字段，包括 fault subtype 子集、mm syscall 密度和 warmup / steady-state 阶段特征；但还没有吃尽所有原始字段，因此“可扩展空间”仍主要在特征工程，不在数据协议或采集层。

## 4. 建模样本单位

基于现有数据，建议把建模样本显式拆成三层，而不是继续抽象出更多理论层级。

### L0. 单次运行原始样本

一条样本对应：

1. 一个 `program`
2. 一个 `variant`
3. 一次 60 秒观测

这个层级的原始来源是 `run_metadata.jsonl + window_metrics.jsonl`。

### L1. 运行级摘要样本

一条样本对应：

1. 一个 `program`
2. 一个 `variant`
3. 一条聚合后的 `run_features`

这层是当前所有模型的主输入。

### L2. 成对样本

一条样本对应：

1. 同一 `program` 下的两个变体 `variant_i` 和 `variant_j`
2. 一组 `[x_i; x_j; x_i - x_j]` 输入
3. 一个方向或倍率标签

这层是当前 pairwise 训练的核心。

### L3. 锚点样本

一条样本对应：

1. 一个 `program`
2. 一个锚点变体，当前默认是 `O0`、`O2` 和 `O3`
3. 一个相对基线的分数 `score_gt`

这层专门服务单程序评分。

## 5. 当前标签设计

这批数据最重要的现实约束是：`wall_time_sec` 基本固定在观测窗口长度附近，不适合作为当前主标签。

因此，现阶段标签应继续沿用已经落地且和数据一致的定义：

1. 用 `cycles_per_iter = total_cycles / iter_count` 作为当前固定工作量代理时间，其中 `iter_count` 优先取真实 `completion_count`，缺失时退化为 `active_pid_count`。
2. pairwise 回归标签定义为 `log(cycles_per_iter_j / cycles_per_iter_i)`。
3. pairwise 三分类标签定义为：
   1. `> +0.05`：`i_better`
   2. `< -0.05`：`j_better`
   3. 其余：`tie`
4. 单程序评分标签定义为相对 `O0` 的对数分数：`log(cycles_per_iter_O0 / cycles_per_iter_k)`。

这一定义是训练期代理标签，而不是最终对外结论本身。原因是：当前数据集的 60s while-true 采集方式更适合恢复“单次迭代的相对代价”，但不能直接替代真正的时间评分。

### 5.1 最终评分必须回到时间验证

模型最终输出的是单程序优化分数，因此最后验收不能只看它是否拟合 `cycles_per_iter`，还必须看它是否和真实时间评分一致。

建议把时间侧真值单独定义为：

1. 为每个 program/variant 运行固定工作量基准。
2. 记录每次完整执行的 `wall_time`，做至少 3 到 5 次 repeat。
3. 用中位数时间定义时间真值分数：`score_time(k) = log(time_O0 / time_k)`。
4. 用模型输出的 `score_model(k)` 与 `score_time(k)` 做相关性、MAE 和档位一致率验证。

如果暂时还没有独立 fixed-work timing 数据，则可以用 `wall_time_sec / completion_count` 形成一个临时 `time_per_iter` 对照分数；但这只能作为过渡检查，不能替代最终时间验证。

## 6. 当前运行级特征设计

当前 `build_run_features.py` 会生成一份更宽的运行级账本，而当前模型实际输入由 `feature_columns.py` 统一定义为 53 维 non-time 特征。需要注意的是：`minor_fault_ratio` 仍保留在 `run_features` 账本里，但因零方差已从模型输入中剔除。

当前 53 维主输入主要分成五类：

1. 效率指标：`ipc`、`cpi`
2. LLC / dTLB / iTLB miss 率与 MPKI
3. page fault 强度、fault subtype 子集与 mm syscall 密度
4. 窗口分布统计：均值、标准差、P95、峰值份额、最小值
5. warmup / steady-state 阶段特征与阶段比值

这套设计的优点是：

1. 不依赖程序名和变体名。
2. 可以直接喂给 MLP 和 Transformer。
3. 对单程序评分和一级瓶颈归因都可复用。

它的局限也很清楚：

1. fault 结构还没完全吃尽，例如 `private_fault_ratio`、`shared_fault_ratio` 仍未进入主输入。
2. mm syscall 字节类特征仍然较薄，目前只保留了聚合密度和 `mmap_bytes_per_ms` 这一类代表量。
3. 阶段特征目前仍以 warmup / steady-state 比值为主，还没有显式纳入峰值窗口位置、热点窗口占比和 PID 集中度等更细粒度阶段信号。

## 7. 不重采数据也能继续扩展的特征

基于现有原始 JSONL，下一轮最值得补的不是更大的 backbone，而是剩余字段和分组消融。

### 7.1 fault 结构特征

1. `private_fault_ratio`
2. `shared_fault_ratio`
3. 更细的 write / instruction / file / anon 组合特征或差值特征

### 7.2 内存系统调用强度特征

1. 显式 `mprotect_per_ms`
2. 更多 `*_bytes_per_ms` 或 `*_bytes_per_ki`
3. `brk_growth_bytes` / `brk_shrink_bytes` 的归一化版本

### 7.3 阶段性窗口特征

1. 峰值窗口位置
2. 热点窗口占比
3. 活跃 PID 集中度
4. 前后固定窗口均值或更细的 phase bucket 特征

这些特征都可以在不改采集代码的情况下直接从现有数据中派生出来。

## 8. 当前数据明确不支持什么

为了避免方案继续漂移，下面这些内容必须明确写成“不支持”而不是“待实现”：

1. 不支持真正的布局 family 结论，因为数据并不是 AoS/SoA 或 blocking 版本。
2. 不支持 repeat 稳定性结论，因为当前没有重复运行。
3. 不支持跨机器泛化结论，因为只有单机数据。
4. 不支持 LBR 建模，因为当前数据里的 `lbr_samples` 基本不构成有效输入。
5. 不支持把 `wall_time_sec` 当作当前主标签，因为观测时长近似固定。

## 9. 推荐的数据策略

现阶段最合理的策略不是新建更多抽象层，而是把下面这条最小数据链路做扎实：

1. 冻结后的 run manifest / curated run list
2. 原始 `window_metrics.jsonl`
3. 运行级 `run_features.parquet`
4. 成对 `pairs.parquet`
5. 锚点 `anchor_set.parquet`
6. 评分输出 `scores.parquet`

在当前这批数据上，这第一步不是形式主义。因为最新 `O0` raw manifest 仍然包含重复采样和缺失目录，不先冻结 run list，后续统计就不具备可复现性。

在当前这批数据上，这一步依然不是形式主义。虽然 `manifest_bcc_*` 已在采集收尾自动去重、`manifest_curated_*` 也可以进一步冻结账本，但训练结果默认还要经过语义过滤；如果不把“full/curated 账本”和“过滤后训练子集”这两层分开写清楚，后续统计同样不具备可复现性。

窗口级原始数据不再承担“协议层”的职责，而只承担两类工作：

1. 给运行级特征提供原料。
2. 给诊断报告提供证据。