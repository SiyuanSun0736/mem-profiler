# 指标正确性对照与引用说明

本文档把“指标正确性对照”收成一条固定流程，用来生成可直接写进论文或答辩材料的对照表。

如果你要整理的是“开销、稳定性、参数敏感性”的阈值和默认配置，而不是“eBPF 指标 vs 外部基准”的正确性对照，请优先看 [docs/design.md](./design.md) 第 5 到第 6 节，以及 `experiments/overhead`、`experiments/stability`、`experiments/sensitivity` 下的脚本。

当前统一入口是 [scripts/build_correctness_tables.py](../scripts/build_correctness_tables.py)。它会把 eBPF 采集结果与外部参考结果整理成 CSV 和 Markdown 两种产物。

如果你手头还没有 `perf stat`、`perf report`、`/proc/<pid>/stat` 和 `/proc/<pid>/maps` 这些原始参考文件，可以直接使用 [experiments/correctness/run_reference_capture.sh](../experiments/correctness/run_reference_capture.sh) 一次性收齐，再由 `build_correctness_tables.py` 自动生成对照表。

---

## 0. 数据口径说明

本页的“外部基准正确性对照”与 `train_set` 中的训练/评分结果不是同一层数据口径。若正文同时引用两类结果，建议先固定下面四点，避免把 `145 x 4` curated run list、过滤后的 `run_features` 子集、strict-time 真值子集和 O2/O3 难例边界混为一谈。

### 0.1 为什么 `145 x 4` curated 最后变成 `509 runs`

1. [scripts/freeze_curated_manifest.py](../scripts/freeze_curated_manifest.py) 会从每个 variant 的 raw manifest 中，为每个 program 只保留“最新且文件完整”的一条 run，并强制 `O0/O1/O2/O3` 的 program 集合完全一致。因此当前 curated manifests 的基线账本是严格的 `145 x 4 = 580` runs。
2. 训练链路实际消费的不是这 `580` 条原始 curated runs，而是经过语义过滤后的 `run_features` 子集。按照 [docs/new-repo-plan/current-data-quality-audit.md](./new-repo-plan/current-data-quality-audit.md) 当前快照，`580` 条里有 `71` 条被过滤，只剩 `509 runs`、`1494 pairs` 和 `374 anchors`。
3. 这 `71` 条被过滤 run 的主因是 `low_active_pid_count`；其中有 `57` 条同时出现 `nonpositive_cycles_per_iter`，说明虽然 run 目录存在，但不能稳定提供可用的运行级特征。过滤后各 variant 保留数分别是 `O0=128`、`O1=127`、`O2=128`、`O3=126`。

### 0.2 为什么最后只剩 `122` 个完整四变体程序

1. 过滤后的 `run_features` 只覆盖 `132` 个程序，其中只有 `122` 个程序仍然同时保留 `O0/O1/O2/O3` 四个 variant，另外 `10` 个程序缺至少一个 variant。
2. 这 `10` 个不完整程序包括 `BitBench_uudecode`、`BitBench_uuencode`、`Bullet`、`MiBench_security-sha`、`PAQ8p`、`Prolangs-C++_city`、`mafft`、`mediabench_gsm_toast`、`mediabench_mpeg2_mpeg2dec` 和 `tramp3d-v4`。缺失来源不是 curated 账本不齐，而是语义过滤后某些 variant 被剔除。
3. 等价地看，`145` 个 curated 程序中有 `13` 个程序在语义过滤后一个 variant 也没保住，另有 `10` 个程序只保住了部分 variant，所以正文凡是涉及“四变体可比”的统计，都应按 `122` 个完整程序口径报告，而不是按 `145` 个 curated 程序口径报告。

### 0.3 strict-time 口径为什么还会继续收缩

1. strict-time 不是对全部 `509 runs` 直接打真值，而是额外要求时间标签满足更严格的活跃窗口条件。当前审计脚本使用的关键阈值是 `active_window_ratio >= 0.10`，低于这个阈值的 run 会先被排除出 strict-time 输入。
2. 按 [docs/new-repo-plan/current-data-quality-audit.md](./new-repo-plan/current-data-quality-audit.md) 当前快照，`run_features` 里有 `109 runs` 进不了 strict-time 输入，主要原因都是 `low_active_window_ratio`；典型例子包括 `BitBench_uuencode (0.0181)`、`FreeBench_pcompress2 (0.0991)`、`MiBench_network-dijkstra (0.0385)` 和 `MiBench_security-sha (0.0117)`。
3. 即便通过 strict 输入检查，也还需要存在 strict `O0` baseline 才能形成成对时间标签。当前另有 `10` 条 run（分布在 `Bullet`、`PAQ8p`、`mafft`、`tramp3d-v4` 这 `4` 个程序上）属于“可做 strict 输入，但缺 strict `O0` baseline”的情况，因此不能直接纳入 strict-time 配对评估。
4. 这也是为什么你会同时看到两组 strict-time 数字：面向全量 `run_features` 的输入审计是“`109` 条输入过滤 + `10` 条缺 strict `O0` baseline”，而面向 `anchor` 真值子集的 `score_time_eval` 则是从 `374` 条 loose-valid 样本里再剔除 `80` 条 `low_active_window_ratio`，最终只剩 `294` 条 strict-valid 样本。两组数字对应的是不同统计层级，并不矛盾。

### 0.4 O2/O3 为什么是当前最难的边界

1. 当前最难的近邻变体边界不是 `O0-O3`，而是 `O2-O3`。在审计快照里，`O2-O3` 一共覆盖 `126` 个程序，`tie_rate=0.4524`，`median |log_ratio|=0.0587`，对应测试集 `acc_3cls=0.4500`、`aux_tie_recall=0.5455`，明显比 `O1-O2` 和 `O1-O3` 更难分。
2. [scripts/audit_train_set_quality.py](../scripts/audit_train_set_quality.py) 当前把 `O2-O3` 难例按 `|log_ratio|` 分成三档：`|log_ratio| <= 0.05` 记为 tie 阈值候选，`0.05 < |log_ratio| <= 0.25` 记为 near-tie 阈值候选，其余才进入“更像真实差异、需要查时序特征”的复核桶。
3. 在此基础上，若 `O2` 或 `O3` 任一侧的 `active_window_ratio < 0.10`，或该侧 run 缺失，则优先标记为 repeat-timing 候选。当前 `O2/O3` 全量程序中，`28` 个属于 repeat-timing 候选，`87` 个属于 tie/near-tie 阈值候选，剩余 `11` 个更适合优先检查时序特征或阶段性访存行为。
4. 因此正文如果讨论当前模型的主要误差来源，应该把重点放在“`O2/O3` 边界高度 near-tie，且一部分样本仍受时间真值质量限制”这一组合问题，而不是简单归因为模型容量不足。

---

## 1. 可生成的对照表

脚本当前支持四类表：

1. PMU 指标与 `perf stat` 对照。
2. 函数级热点与 `perf report` Top-K 对照。
3. fault 计数与 `/proc/<pid>/stat` 前后快照对照。
4. 微基准方向正确性对照。

输出目录下会生成：

1. `pmu_vs_perf_stat.csv/md`
2. `function_hotspot_vs_perf_report_summary.csv/md`
3. `function_hotspot_vs_perf_report_detail.csv/md`
4. `fault_vs_proc_stat.csv/md`
5. `microbench_direction_check.csv/md`
6. `correctness_tables.md`

其中 `correctness_tables.md` 会把所有已生成的表合并成一份总表，便于直接引用。

---

## 2. 输入材料怎么准备

### 2.0 一键收集入口

如果你希望一次性把 eBPF、`perf stat`、`perf report`、`/proc` 快照和最终表格都跑出来，推荐直接使用：

```bash
sudo bash experiments/correctness/run_reference_capture.sh \
  --metric llc_load_misses \
  --duration 30 \
  --output results/correctness_xsbench \
  -- bash -lc 'exec ./bin/xsbench -g 10000 -p 400000'
```

这个脚本会自动完成：

1. eBPF 采集一轮，并保存 `window_metrics.jsonl`、`events.jsonl`。
2. 保存 `/proc/<pid>/stat` 前后快照和 `/proc/<pid>/maps` 快照。
3. 重放 workload 跑一轮 `perf stat`。
4. 重放 workload 跑一轮 `perf record`，随后导出 `perf report --stdio`。
5. 调用 [analysis/attribution.py](../analysis/attribution.py) 和 [scripts/build_correctness_tables.py](../scripts/build_correctness_tables.py) 生成最终 CSV/Markdown 对照表。

默认输出目录结构如下：

1. `ebpf/`：eBPF 原始采集结果。
2. `reference/`：`perf stat`、`perf report`、`/proc` 快照和 `perf.data`。
3. `attribution/`：函数级热点归因结果。
4. `tables/`：最终可引用的对照表。
5. `logs/`：三轮 workload 和分析步骤的日志。

注意：

1. 如果 workload 通过 `bash -lc` 或其他 shell 包装启动，请尽量在命令里使用 `exec`，这样 eBPF 能拿到真实 workload PID，而不是外层 shell PID。
2. 如果 workload 会再 fork 大量子进程，PMU 总量对照会更接近 `perf`，但函数热点对照仍更适合选择单进程或 `exec` 后的主进程场景。

### 2.1 PMU 对 `perf stat`

需要两类输入：

1. 同一 workload 的 `window_metrics.jsonl`
2. 同一 workload 的 `perf stat` 输出文件

推荐用 `-x,` 生成稳定格式，便于脚本解析：

```bash
perf stat -x, \
  -e cycles,instructions,cache-misses,dTLB-load-misses,iTLB-load-misses,page-faults,minor-faults,major-faults \
  -- <your command> \
  2> perf_stat.csv
```

说明：

1. `cache-misses` 在当前对照表里会和 `llc_load_misses + llc_store_misses` 对齐，应解释为粗粒度 proxy 对照。
2. 如果机器不支持 `dTLB-load-misses` 或 `iTLB-load-misses`，对应行会自动缺省。
3. 如果 PMU 资源不足导致 multiplex，建议在表下注明硬件限制，并优先保证 `cycles`、`instructions`、`cache-misses` 三行完整。

### 2.2 函数热点对 `perf report`

需要两类输入：

1. `analysis/attribution.py` 生成的 `function_hotspot_<metric>.csv`
2. `perf report --stdio` 导出的文本文件

推荐流程：

```bash
# 先采集逐事件证据
sudo python3 src/loader.py --pid <PID> --emit-events --duration 30 --output data/run_ref/

# 再做函数级归因
python3 analysis/attribution.py \
  --data data/run_ref/ \
  --pid <PID> \
  --binary /path/to/binary \
  --metric llc_load_misses \
  --output results/run_ref/

# perf 侧生成热点参考
perf report --stdio --no-children --percent-limit 0 > perf_report.txt
```

说明：

1. 两边必须针对同一程序、同一输入、同一二进制版本。
2. 如果要做更硬的对照，建议 `perf record` 与 eBPF 采集都固定绑核和输入规模。
3. 若缺少调试符号，`source_file` 和 `source_line` 可能为空，但函数符号对照仍然有效。

### 2.3 fault 对 `/proc/<pid>/stat`

需要三类输入：

1. 采集窗口对应的 `window_metrics.jsonl`
2. 采集前的 `/proc/<pid>/stat` 快照
3. 采集后的 `/proc/<pid>/stat` 快照

推荐流程：

```bash
cat /proc/<PID>/stat > proc_stat_before.txt

sudo python3 src/loader.py \
  --pid <PID> \
  --duration 30 \
  --output data/run_fault_ref/

cat /proc/<PID>/stat > proc_stat_after.txt
```

说明：

1. 对照脚本会把 `minor_faults` 对齐到 `minflt` 增量，把 `major_faults` 对齐到 `majflt` 增量。
2. 快照必须尽量贴近采集起止点，否则误差会被非采集阶段的 fault 混入。

### 2.4 微基准方向正确性

如果你当前更关心“方向正确性”而不是绝对 fault 数量一致性，可以直接使用已有微基准脚本：

```bash
sudo bash experiments/micro_benchmark/run_micro_bench.sh
```

它默认会生成四组场景：

1. `baseline_sequential`
2. `high_llc_miss`
3. `high_dtlb_miss`
4. `high_page_fault`

当前对照脚本会自动检查：

1. `high_llc_miss` 的 `llc_load_misses` 是否高于基线
2. `high_dtlb_miss` 的 `dtlb_misses` 是否高于基线
3. `high_page_fault` 的 `minor_faults` 是否高于基线

---

## 3. 生成对照表

下面给出一条完整示例命令：

```bash
python3 scripts/build_correctness_tables.py \
  --pmu-window-metrics data/run_ref/window_metrics.jsonl \
  --perf-stat results/reference/perf_stat.csv \
  --function-hotspot results/run_ref/function_hotspot_llc_load_misses.csv \
  --perf-report results/reference/perf_report.txt \
  --fault-window-metrics data/run_fault_ref/window_metrics.jsonl \
  --proc-stat-before results/reference/proc_stat_before.txt \
  --proc-stat-after results/reference/proc_stat_after.txt \
  --microbench-root results/micro_bench_20260515_120000 \
  --output results/correctness_tables
```

如果某次只做其中一类对照，只传入那一类所需的完整输入对即可。

例如只做 PMU 对照：

```bash
python3 scripts/build_correctness_tables.py \
  --pmu-window-metrics data/run_ref/window_metrics.jsonl \
  --perf-stat results/reference/perf_stat.csv \
  --output results/correctness_tables_pmu
```

---

## 4. 论文里怎么引用

建议把几类表分开引用，而不是混成一张大表。

推荐的表题写法：

1. 表 X  eBPF PMU 指标与 `perf stat` 对照结果
2. 表 Y  eBPF 函数级热点与 `perf report` Top-K 对照结果
3. 表 Z  eBPF fault 计数与 `/proc/<pid>/stat` 增量对照结果
4. 表 W  微基准方向正确性验证结果

推荐的解释口径：

1. `cycles`、`instructions` 和 fault 增量更适合做数量一致性对照。
2. `cache-misses` 与 `llc_load_misses + llc_store_misses` 的关系应写成“近似 proxy 对照”，不要写成严格一一等价。
3. 函数热点更适合写 Top-1、Top-3、Top-5 重合度和关键函数是否一致，不宜只报一个总百分比。
4. 微基准表主要支持“方向正确”，不替代严格的绝对计数对照。

---

## 5. 完整案例链路（XSBench）

下面给出 3 个可直接放进论文正文或答辩材料的 XSBench case study。三者共享同一 workload 参数：`-s small -g 1250 -l 1000000`，但对应不同优化级别 O0 / O2 / O3。

说明：

1. 当前 curated 运行只保留了 PID 级 `window_metrics.jsonl`，因此这里的“PID/TID 归因”实际落在 PID 粒度，不能再下钻到线程级。
2. 函数级定位层来自同 workload 的 gprof 辅助 run，而不是原始 case_study 目录中缺失的 `events.jsonl`。对应文件分别是 [results/case_study/xsbench/O0/gprof_O0.txt](../results/case_study/xsbench/O0/gprof_O0.txt)、[results/case_study/xsbench/O2/gprof_O2.txt](../results/case_study/xsbench/O2/gprof_O2.txt)、[results/case_study/xsbench/O3/gprof_O3.txt](../results/case_study/xsbench/O3/gprof_O3.txt)。
3. 为了让 O2 / O3 这类时序更平滑的 run 仍能暴露出最强突发窗口，下面的 case tracing 继续使用 `zscore` 方法，但把 case-study 热点阈值收敛到 `1.4`，而不是默认的 `2.0`。

### 5.1 Case A: XSBench O0

对应材料位于 [results/case_study/xsbench/O0](../results/case_study/xsbench/O0)。

1. 程序级异常：O0 的总 `llc_load_misses` 为 `87,285,400`，明显高于 O2 和 O3；程序级 Top PID 中，[hotspot_llc_load_misses.csv](../results/case_study/xsbench/O0/hotspot_llc_load_misses.csv) 里的 `pid=15292` 单独就累计了 `12,058,484` 次 LLC load miss。
2. 窗口热点：在 [window_hotspots_llc_load_misses.csv](../results/case_study/xsbench/O0/window_hotspots_llc_load_misses.csv) 中，共有 `7` 个热点窗口；最强窗口是 `window 16`，窗口值为 `5,785,237`，z-score 为 `3.0249`，其次是 `window 17`，窗口值为 `5,059,767`。
3. PID 归因：[window_attribution_llc_load_misses.csv](../results/case_study/xsbench/O0/window_attribution_llc_load_misses.csv) 表明热点窗口高度集中在单个 worker PID 上。例如 `window 16` 中 `pid=15292` 占 `98.91%`，`window 17` 中同一 PID 占 `99.31%`；`window 9` 和 `window 10` 则主要落在 `pid=15285`。
4. 函数级定位：[gprof_O0.txt](../results/case_study/xsbench/O0/gprof_O0.txt) 的 flat profile 将热点收敛到 XS 查表路径本身：`calculate_micro_xs` 占 `53.76%` 自身时间，`binary_search` 占 `25.19%`，`grid_search` 占 `8.27%`，`calculate_macro_xs` 占 `3.76%`。这说明 O0 的 LLC miss 异常并不是随机噪声，而是稳定落在截面查找与网格搜索函数族上。

可直接写成的结论是：O0 的访存异常表现为“总 miss 量最高 + 多个连续突发窗口 + 单 PID 极强主导 + 函数热点集中在 XS lookup 内核”。

### 5.2 Case B: XSBench O2

对应材料位于 [results/case_study/xsbench/O2](../results/case_study/xsbench/O2)。

1. 程序级异常：O2 的总 `llc_load_misses` 为 `66,347,482`，比 O0 低约 `24.0%`，但程序级 Top PID 仍然维持在 `2.9M` 到 `3.4M` 区间；[hotspot_llc_load_misses.csv](../results/case_study/xsbench/O2/hotspot_llc_load_misses.csv) 中最热的 `pid=1313703` 累计了 `3,438,026` 次 miss。
2. 窗口热点：[window_hotspots_llc_load_misses.csv](../results/case_study/xsbench/O2/window_hotspots_llc_load_misses.csv) 给出了 `7` 个热点窗口，其中最强窗口是 `window 43`，窗口值为 `2,841,200`，z-score 为 `1.6311`；其次是 `window 36` 的 `2,837,651` 和 `window 31` 的 `2,820,421`。
3. PID 归因：[window_attribution_llc_load_misses.csv](../results/case_study/xsbench/O2/window_attribution_llc_load_misses.csv) 显示热点窗口几乎完全是一窗口一 PID：`window 2 -> pid=1313676`，`window 14 -> pid=1313683`，`window 19 -> pid=1313686`，`window 31 -> pid=1313696`，`window 36 -> pid=1313699`，`window 43 -> pid=1313703`，每个窗口的 Top PID 份额都是 `1.0`。
4. 函数级定位：[gprof_O2.txt](../results/case_study/xsbench/O2/gprof_O2.txt) 中，优化后的热点折叠到更粗的函数边界：`calculate_macro_xs` 占 `60.47%`，`binary_search` 占 `31.40%`，`set_grid_ptrs` 占 `4.65%`。这说明 O2 没有改变热点所属的算法家族，主要是把 O0 中更细碎的查表代价收缩到了宏观 lookup 包装层和搜索层。

可直接写成的结论是：O2 已显著压低总 miss 量，但热点仍然沿 worker PID 轮转出现，根因仍是 `calculate_macro_xs / binary_search` 这条查找路径。

### 5.3 Case C: XSBench O3

对应材料位于 [results/case_study/xsbench/O3](../results/case_study/xsbench/O3)。

1. 程序级异常：O3 的总 `llc_load_misses` 为 `66,716,322`，与 O2 接近，但程序级分布更均匀；[hotspot_llc_load_misses.csv](../results/case_study/xsbench/O3/hotspot_llc_load_misses.csv) 里最热的 `pid=1880590` 为 `3,522,794`，其后多个 PID 都稳定落在 `3.0M+`。
2. 窗口热点：[window_hotspots_llc_load_misses.csv](../results/case_study/xsbench/O3/window_hotspots_llc_load_misses.csv) 中共识别出 `6` 个热点窗口；最强窗口是 `window 4`，窗口值 `3,025,837`，z-score `1.8355`。其后 `window 48`、`window 34`、`window 25`、`window 11` 也都在 `2.74M` 到 `2.78M` 区间。
3. PID 归因：[window_attribution_llc_load_misses.csv](../results/case_study/xsbench/O3/window_attribution_llc_load_misses.csv) 显示 O3 的热点窗口呈现明显的“顺序交接”特征：`window 4 -> pid=1880590`，`window 11 -> pid=1880594`，`window 25 -> pid=1880603`，`window 34 -> pid=1880612`，`window 41 -> pid=1880616`，`window 48 -> pid=1880620`，每个热点窗口仍由单 PID `100%` 主导。
4. 函数级定位：[gprof_O3.txt](../results/case_study/xsbench/O3/gprof_O3.txt) 继续把热点锁定在同一条查表主链：`calculate_macro_xs` 占 `60.47%`，`binary_search` 占 `31.40%`，`hash` 与 `pick_mat` 各占 `2.33%`。与 O2 相比，O3 没有消除 lookup hotspot family，而是把热点窗口压缩成更规则、更可预测的 burst 形态。

可直接写成的结论是：O3 的热点已经从“高噪声、高波动”收敛成“更规则的单 PID burst”，但函数级根因仍然位于 XS 查表主链，没有发生算法性迁移。

### 5.4 这 3 个案例共同说明什么

1. 从程序级总量到窗口级峰值，再到 PID 归因，XSBench 的异常始终不是分散噪声，而是可以稳定收敛到少数 worker 的突发性 lookup 阶段。
2. 即使优化级别从 O0 提升到 O2 / O3，函数级定位仍然反复落在 `calculate_macro_xs`、`calculate_micro_xs`、`binary_search` 这组 XS lookup 核心函数上，说明归因链路具备跨编译配置的一致性。
3. 对当前仓库来说，这一组案例已经把“程序级异常 -> 窗口热点 -> PID 归因 -> 函数级定位”的证据链完整闭合；唯一缺口是线程级明细，因为原始 curated run 不是按 `--per-tid` 采集的。

---

## 6. 当前脚本的边界

1. `perf report` 的文本格式在不同系统上可能略有差异，建议优先保留 `--stdio` 文本原件，必要时按本仓库脚本的正则格式微调。
2. `/proc/<pid>/stat` 只能验证 `minor_faults` 和 `major_faults`，不能直接验证更细的 fault subtype。
3. 如果目标 workload 本身不稳定，应先做稳定性实验，再把正确性表作为正文主证据。
