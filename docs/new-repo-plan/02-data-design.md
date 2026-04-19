# 数据设计方案

> 这份文档回答“样本长什么样，标签怎么来，文件怎么落”。

## 1. 数据分层

建议把新仓库的数据显式拆成四层，而不是只保留一个训练文件。

### L0. 原始运行记录层

记录一次真实运行的直接输出，包括：

1. 程序与版本元数据。
2. 时间真值。
3. PMU 计数。
4. page fault 与系统事件。
5. 采集环境元信息。

### L1. 运行级样本层

将 L0 清洗成“单次运行一个样本”的建模输入。

### L2. 成对样本层

在同一程序家族内，把两个布局版本组合成一条 pair，生成优劣标签和倍率标签。

### L3. 评估视图层

从同一份 L1/L2 数据构造三种特征视图：

1. time-only
2. non-time
3. full

## 2. 样本单位

第一阶段建议把样本单位定义为：

“某个 program family 下，某个 layout variant 在某个输入条件、某次重复运行中的一次观测。”

对应最小主键可以是：

1. program_id
2. layout_variant
3. input_id
4. repeat_id

## 3. L0 原始运行记录字段

建议新仓库第一版使用 CSV 或 JSONL 都可以，但字段应固定下来。

### 3.1 元数据字段

1. program_id
2. program_family
3. layout_variant
4. layout_group
5. input_id
6. input_size
7. repeat_id
8. compiler_id
9. compiler_flags
10. machine_id
11. run_ts

### 3.2 时间真值字段

1. wall_time_sec
2. cpu_time_sec
3. fixed_work_time_sec
4. throughput
5. baseline_layout_variant

如果程序天然是固定工作量任务，优先使用 fixed_work_time；如果是吞吐型任务，则把 throughput 一并保留用于解释，但仍建议落地成可比较的时间或速率标签。

### 3.3 非时间特征字段

以下字段均通过 eBPF perf_event 采样或 kprobe 直接采集，非降级代理。

#### 基础硬件计数器（PERF_TYPE_HARDWARE）

1. cycles — `PERF_COUNT_HW_CPU_CYCLES`，差分累加
2. instructions — `PERF_COUNT_HW_INSTRUCTIONS`，差分累加

#### LLC 组（HW_CACHE ll.read/write）

1. llc_loads — LLC load access 采样计数
2. llc_load_misses — LLC load miss 采样计数
3. llc_stores — LLC store access 采样计数（部分 CPU 降级为 cache-references 代理）
4. llc_store_misses — LLC store miss 采样计数（部分 CPU 降级为 cache-misses 代理）

#### dTLB 组（HW_CACHE dtlb.read/write）

1. dtlb_loads — dTLB load access 采样计数
2. dtlb_load_misses — dTLB load miss 采样计数
3. dtlb_stores — dTLB store access 采样计数
4. dtlb_store_misses — dTLB store miss 采样计数
5. dtlb_misses — dtlb_load_misses + dtlb_store_misses 的聚合值（冗余但便于查询）

#### iTLB 组（HW_CACHE itlb.read）

1. itlb_load_misses — iTLB load miss 采样计数（`itlb.read.access` 在此处理器不可用，仅采集 miss）

#### page fault（kprobe/handle_mm_fault）

1. minor_faults — minor page fault 次数
2. major_faults — major page fault 次数

#### 采集辅助

1. samples — eBPF handler 触发总次数（用于估算采样覆盖率）

> **不在此列的字段说明**

- `cache_references` / `cache_misses`：仅作为 LLC store 不支持时的降级代理，不作为独立采集项。
- `page_faults`：= `minor_faults + major_faults`，在派生字段层计算，不单独采集。
- `context_switches` / `cpu_migrations`：当前 eBPF 程序不采集此类 SW 事件，第二阶段如有需要可补充 `PERF_COUNT_SW_CONTEXT_SWITCHES` / `PERF_COUNT_SW_CPU_MIGRATIONS`。
- `lbr_samples` / `lbr_entries`：可选，仅在 `--lbr` 启用时输出，不属于第一版基线字段。

### 3.4 归一化衍生字段

不要只存原始计数，建议同步生成一组派生特征：

1. ipc = instructions / cycles
2. llc_load_miss_rate = llc_load_misses / llc_loads
3. llc_mpki = llc_load_misses / instructions × 1000
4. dtlb_miss_rate = dtlb_misses / (dtlb_loads + dtlb_stores)
5. dtlb_mpki = dtlb_load_misses / instructions × 1000
6. page_faults = minor_faults + major_faults
7. fault_per_ms = page_faults / wall_time_ms

这样做的原因是，不同输入规模下原始计数不可直接比较，而比率特征更容易跨样本稳定。

## 4. L1 运行级样本文件建议

建议建立一个统一的运行级样本表：

1. runs.csv 或 runs.parquet
2. run_features.parquet

推荐目录：

```text
data/
  raw/
    program_family/layout_variant/input_id/repeat_id/
  processed/
    runs.parquet
    run_windows.parquet
    run_features.parquet
    runs_non_time.parquet
    runs_full.parquet
```

其中：

1. raw 保留原始可追溯信息。
2. run_windows.parquet 保留“每个窗口对整个程序”的聚合视图，主要给坏窗口定位和窗口级归因使用。
3. run_features.parquet 保留“整个 run 一条记录”的聚合特征，主要给单程序优化打分使用。
4. 特征视图通过单独文件或视图配置维护，不要在训练代码里临时拼接。

### 4.1 为什么需要 run_features

`window_metrics.jsonl` 适合做窗口级分析，但不适合直接喂给“单程序优化程度打分”模型。原因是同一次 run 往往会包含：

1. 多个窗口。
2. 多个 PID 或 TID。
3. 局部噪声与短时峰值。

因此，单程序打分需要先把一次 run 聚合成一条稳定样本，也就是 `run_features`。

建议采用两步聚合：

1. 先将同一 `window_id` 下的多 PID/TID 记录聚合成 `run_windows`。
2. 再从 `run_windows` 聚合成一条 `run_features` 记录。

### 4.2 run_windows 聚合规则

`run_windows.parquet` 每个 `run_id + window_id` 只有一条记录。

聚合规则建议如下：

1. 计数字段用求和：如 `cycles`、`instructions`、`llc_load_misses`、`minor_faults`。
2. 时间字段保留窗口原始边界：`start_ns`、`end_ns`。
3. 活跃实体信息单独保留：`active_entities`、`active_pids`、`active_tids`。
4. 可选保留 Top-N 实体摘要，供后续热点解释使用。

推荐字段：

1. run_id
2. window_id
3. start_ns
4. end_ns
5. active_entities
6. cycles
7. instructions
8. llc_loads
9. llc_load_misses
10. llc_stores
11. llc_store_misses
12. dtlb_loads
13. dtlb_misses
14. itlb_load_misses
15. minor_faults
16. major_faults
17. samples
18. ipc
19. llc_load_miss_rate
20. llc_mpki
21. dtlb_miss_rate
22. dtlb_mpki
23. fault_per_ms

### 4.3 run_features 的设计目标

`run_features` 不是简单地把所有窗口求和，而是要同时服务两个目标：

1. 给单程序优化打分模型提供稳定输入。
2. 给单程序诊断报告提供可解释证据。

因此字段应分成五组，而不是只保留原始总量。

### 4.4 run_features 字段分组

#### A. 标识与实验上下文字段

这些字段不一定进入模型，但必须保留在表里，便于切分、回溯和与基线对齐。

1. run_id
2. program_id
3. program_family
4. layout_variant
5. layout_group
6. input_id
7. input_size
8. repeat_id
9. compiler_id
10. compiler_flags
11. machine_id
12. run_ts
13. baseline_layout_variant
14. window_sec
15. aggregation_scope

#### B. 时间真值与参考系字段

这些字段主要用于标签、验证和锚点对齐，默认不进入 non-time 模型输入。

1. wall_time_sec
2. cpu_time_sec
3. fixed_work_time_sec
4. throughput
5. baseline_time_sec
6. anchor_set_id
7. reference_score

其中：

1. `baseline_time_sec` 是同 family 基线版本的时间真值。
2. `anchor_set_id` 标记这次单程序推理所使用的锚点集合。
3. `reference_score` 是可选字段，表示真实相对基线分数。

#### C. 运行级总量字段

这组字段回答“整个 run 一共发生了多少”。

1. total_cycles
2. total_instructions
3. total_llc_loads
4. total_llc_load_misses
5. total_llc_stores
6. total_llc_store_misses
7. total_dtlb_loads
8. total_dtlb_misses
9. total_itlb_load_misses
10. total_minor_faults
11. total_major_faults
12. total_page_faults
13. total_samples
14. window_count
15. active_window_count
16. active_entity_count_mean

#### D. 运行级归一化主特征

这组字段是单程序打分模型最应该优先消费的核心输入。

1. ipc = total_instructions / total_cycles
2. cpi = total_cycles / total_instructions
3. llc_load_miss_rate = total_llc_load_misses / total_llc_loads
4. llc_store_miss_rate = total_llc_store_misses / total_llc_stores
5. llc_mpki = total_llc_load_misses / total_instructions × 1000
6. dtlb_miss_rate = total_dtlb_misses / (total_dtlb_loads + total_dtlb_stores)
7. dtlb_mpki = total_dtlb_misses / total_instructions × 1000
8. itlb_mpki = total_itlb_load_misses / total_instructions × 1000
9. page_faults_per_ms = total_page_faults / wall_time_ms
10. minor_fault_ratio = total_minor_faults / total_page_faults
11. samples_per_ms = total_samples / wall_time_ms

如果后续接入 `context_switches` 或 `cpu_migrations`，也应在这一组追加对应归一化率，而不是只加原始计数。

#### E. 时间窗分布特征

这组字段回答“问题是均匀存在，还是集中爆发”。它们对单程序瓶颈归因非常重要。

建议对下面几类窗口级量都提取统计特征：

1. cycles
2. instructions
3. llc_load_misses
4. dtlb_misses
5. itlb_load_misses
6. page_faults
7. ipc

每类量至少提取：

1. mean
2. std
3. max
4. p95
5. p50
6. peak_share = max / sum

例如可以落成：

1. win_llc_miss_mean
2. win_llc_miss_std
3. win_llc_miss_p95
4. win_llc_miss_peak_share
5. win_ipc_mean
6. win_ipc_min
7. win_fault_max

第一版不必对所有指标都做全量统计，但至少应覆盖 LLC miss、fault、cycles、ipc 这四组。

#### F. 诊断辅助字段

这组字段不一定直接进入模型，但应保留在 `run_features` 中，方便后续自动生成单程序诊断报告。

1. worst_window_id_by_cycles
2. worst_window_id_by_llc_miss
3. worst_window_id_by_faults
4. worst_window_id_by_ipc
5. hotspot_window_count
6. llc_hot_window_count
7. tlb_hot_window_count
8. fault_hot_window_count

这些字段是从 `run_windows` 直接派生的摘要索引，不是模型必须输入，但对解释层很有价值。

### 4.5 推荐的最小 run_features 视图

如果只做一版最小可用字段集合，建议模型先使用下面这组：

1. ipc
2. cpi
3. llc_load_miss_rate
4. llc_mpki
5. dtlb_miss_rate
6. dtlb_mpki
7. itlb_mpki
8. page_faults_per_ms
9. win_llc_miss_peak_share
10. win_fault_max
11. win_ipc_min
12. active_entity_count_mean

这组字段规模不大，但已经能同时覆盖整体效率、cache 压力、TLB 压力、fault 压力和时间局部异常。

### 4.6 文件建议

推荐直接固定三类处理后文件：

1. run_windows.parquet：窗口级聚合结果
2. run_features.parquet：完整运行级聚合特征
3. run_features_non_time.parquet：去除时间真值后的模型输入视图

## 5. L2 成对样本构造规则

### 5.1 配对原则

仅在以下维度完全相同时允许配对：

1. program_family 相同。
2. input_id 相同。
3. input_size 相同。
4. 编译条件相同。
5. 机器与采集环境相同。

### 5.2 标签定义

若版本 i 与 j 的时间分别为 T_i 和 T_j，则：

$$
Y_{i,j}^{ratio} = \frac{T_j}{T_i}
$$

1. 若 $Y_{i,j}^{ratio} > 1 + \epsilon$，则 i 更优。
2. 若 $Y_{i,j}^{ratio} < 1 - \epsilon$，则 j 更优。
3. 若 $|Y_{i,j}^{ratio} - 1| \le \epsilon$，则视为近似持平。

这里的 $\epsilon$ 用于吸收重复运行波动，第一版可以从 0.02 或 0.05 开始试。

### 5.3 成对文件建议

推荐文件：

1. pairs.parquet
2. pairs_time_only.parquet
3. pairs_non_time.parquet
4. pairs_full.parquet

每条 pair 应至少包含：

1. pair_id
2. program_family
3. version_i
4. version_j
5. features_i
6. features_j
7. label_direction
8. label_ratio

## 6. 训练/验证/测试切分

切分应优先按 program family 做 group split，而不是随机抽样。原因是如果同一家族版本同时进入训练和测试，模型可能学到程序身份而不是优化规律。

建议三种切分同时保留：

1. in-family split：检验同一程序家族内泛化。
2. cross-family split：检验跨程序家族泛化。
3. leave-one-layout-group-out：检验对未见布局模式的外推能力。

## 7. 第二阶段时序扩展

如果第一阶段成立，再增加窗口级时序数据层：

```text
data/
  processed/
    windows.parquet
    sequences/
      pair_000001.npz
```

时序字段建议包括：

1. window_id
2. window_duration_ms
3. window_cycles
4. window_instructions
5. window_llc_misses
6. window_dtlb_misses
7. window_faults

时序层的目标不是替代运行级数据，而是作为模型升级路径。
