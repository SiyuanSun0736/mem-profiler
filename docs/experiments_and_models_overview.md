# 项目实验与模型总览（Experiments & Models Overview）

> 梳理 `mem-profiler` 当前已实现的实验体系、特征工程与评估模型，以及尚待补充的模块。

---

## 一、 实验体系（`experiments/` & `results/`）

### 1. 测量正确性验证（`experiments/correctness/` & `experiments/micro_benchmark/`）
- **权威工具对齐采集**：[`experiments/correctness/run_reference_capture.sh`](../experiments/correctness/run_reference_capture.sh) + [`scripts/build_correctness_tables.py`](../scripts/build_correctness_tables.py)
  - 同一负载分别由 eBPF、`perf stat`、`perf record + perf report` 与 `/proc/<pid>/stat` 采集并生成对照表。
  - 验证内容：硬件 PMU 计数对齐、`minor_faults`/`major_faults` 误差、Top-K 函数级热点归因及源码行定位准确率。
- **可控微基准测试**：[`experiments/micro_benchmark/bench_mem.c`](../experiments/micro_benchmark/bench_mem.c) & [`run_micro_bench.sh`](../experiments/micro_benchmark/run_micro_bench.sh)
  - 覆盖 4 类访存敏感型负载：顺序基线、随机 LLC Miss（`workload_random_llc`）、跨页 dTLB Miss（`workload_stride_tlb`）、高密度 `mmap` 次要缺页（`workload_page_fault`）及纯计算对照（`workload_cpu_compute`）。
  - 配套同步机制：[`launch_sync.py`](../experiments/micro_benchmark/launch_sync.py) 与 `libsync.so`（`SIGSTOP`/`SIGCONT` 零盲区冷启动同步）。

### 2. 观测开销与方法学横评（`experiments/overhead/`、`sensitivity/`、`stability/`）
- **跨工具性能与存储横评**：[`experiments/overhead/benchmark_cross_tool.py`](../experiments/overhead/benchmark_cross_tool.py)
  - 横向对比 Baseline（Native）、`perf stat`、`perf record (-g)`、`eBPF (WindowSnapshot)` 与 `eBPF (Event Attribution)` 的运行时延时开销（%）与落盘体积（KB）。
- **方法学专项评测**（配套 [`scripts/build_methodology_tables.py`](../scripts/build_methodology_tables.py)）：
  - [`experiments/overhead/run_overhead.sh`](../experiments/overhead/run_overhead.sh)：不同探针组合下的运行时开销测试；
  - [`experiments/sensitivity/run_sensitivity.sh`](../experiments/sensitivity/run_sensitivity.sh)：采样频率、时间窗跨度等参数敏感性测试；
  - [`experiments/stability/run_stability.sh`](../experiments/stability/run_stability.sh)：多轮独立重复采样的稳定性与变异系数测试。

### 3. 编译器多变体数据集采集与 BOLT 案例（`experiments/llvm_test_suite/`）
- **多优化等级批量采集**：[`collect_dataset_all_variants.sh`](../experiments/llvm_test_suite/collect_dataset_all_variants.sh)、[`collect_dataset_testbench.sh`](../experiments/llvm_test_suite/collect_dataset_testbench.sh)、[`collect_single_bcc_testbench.sh`](../experiments/llvm_test_suite/collect_single_bcc_testbench.sh)
  - 基于 LLVM Test Suite 构建 `O0` / `O1` / `O2` / `O3` 四个优化等级变体（[`cmake-caches/`](../experiments/llvm_test_suite/cmake-caches/)），批量采集微架构特征并通过 [`dedup_dataset_variant.py`](../experiments/llvm_test_suite/dedup_dataset_variant.py) 去重。
- **BOLT 代码重排优化对比**：[`bolt_optimize.sh`](../experiments/llvm_test_suite/bolt_optimize.sh) & [`extract_elf.sh`](../experiments/llvm_test_suite/extract_elf.sh)
  - 采集二进制代码布局重排前后的 iTLB / 缓存与执行周期变化。

### 4. 尚待补充的实验与算法
- 详见 [`docs/todo_phase_and_xsbench.md`](./todo_phase_and_xsbench.md)：
  1. 阶段自动切分算法（论文 4.2 节）
  2. XSBench 科学计算多变体案例（论文 5.4.1 节）

---

## 二、 分析脚本与评估模型（`analysis/`、`scripts/` & `train_set/`）

### 1. 时序分析与热点归因（`analysis/`）
- **窗口级异常检测与 Co-Spike 共振**：[`analysis/hotspot.py`](../analysis/hotspot.py) & [`analysis/report.py`](../analysis/report.py)
- **函数级 / 源码行归因**：[`analysis/attribution.py`](../analysis/attribution.py)、[`analysis/attribution_report.py`](../analysis/attribution_report.py)、[`analysis/symbolize.py`](../analysis/symbolize.py)（支持 `/proc/<pid>/maps`、`addr2line`、`/proc/kallsyms` 及用户态过滤）
- **数据集热点与指标相关性分析**：[`analysis/dataset_hotspot.py`](../analysis/dataset_hotspot.py)、[`analysis/dataset_hotspot_report.py`](../analysis/dataset_hotspot_report.py)、[`analysis/metric_relation_report.py`](../analysis/metric_relation_report.py)

### 2. 数据构建与质量审计（`scripts/` & `train_set/`）
- **特征与成对样本构建**：[`build_run_features.py`](../scripts/build_run_features.py)、[`build_pair_table.py`](../scripts/build_pair_table.py)、[`build_anchor_set.py`](../scripts/build_anchor_set.py)、[`feature_columns.py`](../scripts/feature_columns.py)
- **数据审计与清单冻结**：[`audit_train_set_quality.py`](../scripts/audit_train_set_quality.py)、[`freeze_curated_manifest.py`](../scripts/freeze_curated_manifest.py)、[`collect_repeat_timing.py`](../scripts/collect_repeat_timing.py)

### 3. 主线成对比较模型（Pairwise Transformer Encoder）
- **核心训练脚本**：[`scripts/train_transformer.py`](../scripts/train_transformer.py)（输出权重：[`train_set/model_transformer.pt`](../train_set/model_transformer.pt)）
  - **网络架构**：将变体特征对 `(x_i, x_j)` 作为两个 token 经共享线性投影输入 `TransformerEncoder`，拼接 `[out_i ; out_j ; out_i - out_j]` 后通过回归头预测 `log(total_cycles_j / total_cycles_i)`，辅以三分类头（`i_better` / `tie` / `j_better`）。
- **消融、加权与校准**：
  - 目标函数消融：[`run_transformer_objective_ablation.py`](../scripts/run_transformer_objective_ablation.py)、[`compare_selection_objectives.py`](../scripts/compare_selection_objectives.py)
  - 不确定性与锚点加权：[`compare_uncertainty_anchor_weighting.py`](../scripts/compare_uncertainty_anchor_weighting.py)
  - 成对校准与模型融合：[`fit_pair_calibration.py`](../scripts/fit_pair_calibration.py)、[`compare_pair_calibration_blends.py`](../scripts/compare_pair_calibration_blends.py)
  - 超参调优：[`tune_transformer.py`](../scripts/tune_transformer.py)

### 4. 规则/特征打分基线模型
- **程序性能打分与微调**：[`scripts/score_program.py`](../scripts/score_program.py)、[`tune_score_program_fine.py`](../scripts/tune_score_program_fine.py)
- **打分与运行时间一致性评估**：[`evaluate_score_vs_time.py`](../scripts/evaluate_score_vs_time.py)、[`build_time_score_table.py`](../scripts/build_time_score_table.py)

### 5. 扩展 Token 分支实验模型（`scripts/experimental/`）
- 在 [`scripts/experimental/`](../scripts/experimental/) 下探索了多种特征 Token 化变体：
  - `category_token` / `category_token_ablation`（按指标类别分组 token）
  - `hotspot_token`（引入函数级热点特征 token）
  - `fixed_work_token` / `fixed_work_token_pooled` / `fixed_work_hotspot_token`（固定工作量表征与池化分支）
  - 横向评测脚本：[`compare_token_branches.py`](../scripts/experimental/compare_token_branches.py)、[`compare_token_branches_all.py`](../scripts/experimental/compare_token_branches_all.py)、[`compare_token_branches_extended.py`](../scripts/experimental/compare_token_branches_extended.py)
