# 实验状态盘点与数据口径核查（Status & Data Lineage）

> 记录当前项目中哪些实验与模型已基于最新采集器完成、哪些基于旧版采集器（需注意数据口径或重跑），以及哪些尚未运行或待开发。

---

## 一、 基于最新优化版采集器的可用结果（可直接用于论文）

最新采集器核心特性：`perf_event_open` 真计数（`auto`）+ `tracepoint:exceptions:page_fault_user` + `SIGSTOP`/`SIGCONT` 冷启动同步 + 内核态 Per-CPU 加权 IP 聚合 + `4000 Hz` 频率采样 + `/proc/kallsyms` & 用户态过滤符号化（已剥离 LBR）。

| 实验项目 | 结果目录 / 文件 | 核心可用数据 |
| :--- | :--- | :--- |
| **测量正确性对齐**（`bench_mem -r 160`） | `results/correctness_optimized_fixed/`<br>`results/correctness_optimized_agg/`<br>`results/correctness_optimized_hotspot/` | • `minor_faults` vs `/proc/<pid>/stat`：误差 **0.0042%**（比值 `1.0000`）<br>• `instructions` vs `perf stat`：误差 **2.00%**（比值 `0.9800`）<br>• `cycles` vs `perf stat`：误差 **5.79%**（比值 `0.9421`）<br>• 函数热点归因：`<unknown>` **0%**，Top-1 匹配 **100%**（`workload_random_llc` @ `bench_mem.c:25`） |
| **跨工具性能与存储横评**（`bench_mem -r 80`，5 轮均值） | `results/cross_tool_benchmark/cross_tool_overhead.md` | • Baseline：`4765.41 ± 76.16 ms` (`+0.00%`)<br>• `perf stat`：`4740.70 ± 79.69 ms` (`-0.52%`, `0.25 KB`)<br>• `perf record (-g)`：`4587.79 ± 102.56 ms` (`-3.73%`, `1740.70 KB`)<br>• `eBPF (WindowSnapshot)`：`4602.70 ± 106.41 ms` (`-3.41%`, `4.84 KB`)<br>• `eBPF (Event Attribution)`：`4814.64 ± 37.89 ms` (`+1.03%`, `38.57 KB`) |
| **微基准访存方向验证** | `results/micro_bench_20261122_224921/` | 顺序基线、随机 LLC Miss、跨页 dTLB Miss（激增 12.5 倍）、高密缺页 4 类模式趋势全部符合预期 |
| **汇总对比报告** | `docs/perf_comparison_and_validation_report.md` | 已更新并同步至 `mem-ebpf/实验评测-perf对比测试与正确性验证报告.md` |

---

## 二、 基于旧版采集器的历史数据与模型（需注意数据口径）

以下结果与模型训练于采集器重构之前（见 `train_set/run_feature_filter_summary.json`，原数据采集于 `data/llvm_test_suite/bcc/...`，当时使用旧版 `bcc` 采样模式与 `kprobe` 缺页探针）：

| 项目 | 位置 | 现状与口径影响评估 |
| :--- | :--- | :--- |
| **方法学开销历史实验** | `results/overhead_20261122_230543/` | 旧版 kprobe 时代测试，开销数据已被 `cross_tool_benchmark/` 替代 |
| **主线 Pairwise Transformer** | `train_set/model_transformer.pt`<br>`train_set/model_transformer_eval.json` | 90/19/20 程序划分（240 测试对）。模型结构与特征（`BASE_NON_TIME_COLS`）**未使用 LBR**，架构有效，但输入特征分布来自旧版 `bcc`+`kprobe` |
| **目标函数消融实验** | `train_set/objective_ablation.md` | `reg_ce`（$\lambda=0.05$）最优：Test $R^2=0.8031$，方向准确率 `90.69%`，辅助三分类 `84.17%` |
| **打分与校准基线** | `train_set/score_eval.json`<br>`train_set/pair_calibration_report.md`<br>`train_set/uncertainty_anchor_weighting_comparison.md` | 509 runs，`corr_score_log = 0.9072`；同样基于旧版数据集 |
| **扩展 Token 分支实验** | `train_set/category_token*`, `hotspot_token*`, `fixed_work_*` | 已跑出对比输出，基于旧版数据集 |

### 口径变化说明（为什么建议重采数据集并重训）
1. **LBR 无影响**：`scripts/feature_columns.py` 中的 `BASE_NON_TIME_COLS` 本身未包含任何 LBR 特征。
2. **缺页细分类变化**：切到 `tracepoint:exceptions:page_fault_user` 后，不再读取内核 `vma->vm_file`，因此 `file_fault_ratio` 会退化为 0（若需训练集中保持非零，可评估是否剔除或用 `--fault-backend kprobe`）。
3. **PMU 与冷启动精度提升**：新版 `perf_event_open` + `SIGSTOP`/`SIGCONT` 消除了冷启动盲区，`cycles`、`instructions`、`ipc` 及 `warmup_*`/`steady_*` 阶段特征信噪比显著高于旧版 `bcc` 采样。
4. **建议**：为确保毕业论文第五章全篇数据口径 100% 统一，建议在条件允许时用**最新版采集器**重跑 `experiments/llvm_test_suite/collect_dataset_all_variants.sh`，再重跑 `train_transformer.py` 与消融脚本。

---

## 三、 本机尚未运行或尚待开发的实验与模块

### 1. 有脚本但本机尚未运行的专项实验
- `experiments/sensitivity/run_sensitivity.sh`：采样频率、时间窗粒度参数敏感性扫描
- `experiments/stability/run_stability.sh`：多轮独立重复运行稳定性与变异系数测试
- `experiments/llvm_test_suite/bolt_optimize.sh`：LLVM BOLT 二进制代码重排对比案例（对应论文 5.4.2 节）

### 2. 代码与实验均待开发的模块（详见 `docs/todo_phase_and_xsbench.md`）
- **阶段自动切分算法**（对应论文 4.2 节）
- **XSBench 科学计算多变体案例**（对应论文 5.4.1 节）
