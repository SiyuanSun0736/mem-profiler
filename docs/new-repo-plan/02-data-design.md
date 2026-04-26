# 数据现状与样本设计

> 这份文档回答“这批真实数据到底长什么样，现阶段应该按什么样本单位和标签来建模”。

## 1. 当前数据快照

现有数据不是从零设计的新数据集，而是已经落盘的 llvm-test-suite BCC 采集结果。基于 manifest、原始 JSONL 和训练产物，可以确认下面这些事实：

1. 数据根目录是 `data/llvm_test_suite/bcc/O0~O3`。
2. 每个程序有 4 个变体：`O0`、`O1`、`O2`、`O3`。
3. 当前共覆盖 145 个程序，因此总运行数是 580。
4. 每次运行默认观测 60 秒，聚合窗口是 1.0 秒。
5. 当前数据来自同一台机器，没有多机采样维度。
6. 当前没有 repeat 维度，本质上是“每个 program × variant 一次运行”。

这意味着第一阶段任务必须是“单次运行、单机、程序内相对比较”的问题，而不是“重复稳定性”和“跨平台泛化”的问题。

## 2. 原始文件层

每次运行至少包含两类文件：

1. `run_metadata.jsonl`：运行起止时间、目标进程、窗口大小、采样率、后端信息。
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

需要强调的是：当前训练脚本只消费了其中一部分字段，说明“可扩展空间”主要在特征工程，不在数据协议或采集层。

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
2. 一个锚点变体，目前是 `O0` 和 `O3`
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

当前 `build_run_features.py` 已经构造了 38 个 non-time 特征，主要分成四类：

1. 效率指标：`ipc`、`cpi`
2. LLC / dTLB / iTLB miss 率与 MPKI
3. page fault 强度与样本密度
4. 窗口分布统计：均值、标准差、P95、峰值份额、最小值

这套设计的优点是：

1. 不依赖程序名和变体名。
2. 可以直接喂给 MLP 和 Transformer。
3. 对单程序评分和一级瓶颈归因都可复用。

它的局限也很清楚：

1. 还没有使用 fault 子类型字段。
2. 还没有使用 mmap / munmap / brk 等系统调用字段。
3. 还没有把“冷启动窗口”和“稳态窗口”拆开建特征。

## 7. 不重采数据也能立刻扩展的特征

基于现有原始 JSONL，下一轮最值得补的特征不是更多模型，而是更多派生量。

### 7.1 fault 结构特征

1. `anon_fault_ratio`
2. `file_fault_ratio`
3. `write_fault_ratio`
4. `instruction_fault_ratio`
5. `private_fault_ratio`

### 7.2 内存系统调用强度特征

1. `mmap_calls_per_ms`
2. `munmap_calls_per_ms`
3. `mprotect_calls_per_ms`
4. `brk_calls_per_ms`
5. 各类 `*_bytes_per_ki`

### 7.3 阶段性窗口特征

1. 前 5 个窗口均值
2. 后 5 个窗口均值
3. 峰值窗口位置
4. 热点窗口占比
5. 活跃 PID 集中度

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

1. 原始 `window_metrics.jsonl`
2. 运行级 `run_features.parquet`
3. 成对 `pairs.parquet`
4. 锚点 `anchor_set.parquet`
5. 评分输出 `scores.parquet`

窗口级原始数据不再承担“协议层”的职责，而只承担两类工作：

1. 给运行级特征提供原料。
2. 给诊断报告提供证据。