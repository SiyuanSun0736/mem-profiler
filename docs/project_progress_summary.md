# 毕设项目当前进度总览（2026-11-28）

> 对照毕业论文《基于 eBPF 的程序细粒度访存特征提取与优化评估方法》各章节要求，梳理当前代码实现、实验验证与模型评估的整体进度。

---

## 一、 整体完成度概览

| 模块 / 论文对应章节 | 状态 | 核心交付物与实测指标 |
| :--- | :---: | :--- |
| **1. eBPF 双平面采集引擎（第 3 章）** | **已完成（已优化）** | • `perf_event_open` 硬件真计数 + `tracepoint:exceptions:page_fault_user` 轻量缺页探测<br>• `SIGSTOP`/`SIGCONT` 冷启动零盲区同步（`libsync.so` + `--sync-target`）<br>• 内核态 Per-CPU 加权 IP 聚合（`ip_hotspot_map`）+ `4000 Hz` 自适应频率采样<br>• 剥离非通用 Intel LBR 依赖，全面适配 AMD Zen 4 / 通用 x86_64 平台 |
| **2. 多层次下钻归因分析（第 4 章）** | **已完成** | • 时间窗异常检测与 Co-Spike 共振（`analysis/hotspot.py`）<br>• **阶段自动切分算法（4.2 节）**：`analysis/phase_segmentation.py`（自动识别 `init` / `steady` / `teardown`）<br>• 符号化与源码行归因：`analysis/symbolize.py` + `analysis/attribution.py`（支持 `/proc/kallsyms` 与用户态过滤，`<unknown>` 降至 **0%**） |
| **3. 测量正确性与开销横评（第 5.2~5.3 节）** | **已完成（最新实测）** | • **正确性**：`minor_faults` vs `/proc` 误差 **0.0042%**；`instructions` vs `perf stat` 误差 **2.00%**；Top-1 热点匹配 **100%**<br>• **开销与存储**：`WindowSnapshot` 延时 **`-3.41%~+1.43%`**（`4.84 KB`）；`Event Attribution` 延时 **`+1.03%`**（`38.57 KB`，比 `perf record -g` 小 **45 倍**）<br>• 报告：`docs/perf_comparison_and_validation_report.md` |
| **4. 典型案例一：XSBench 科学计算（第 5.4.1 节）** | **已完成（最新实测）** | • `xsbench_O0` / `O2` / `O3`（5.6 GB 规模）全链路下钻完成<br>• 自动切分 `init` 与 `steady`（$R^2 \approx 0.60$），量化 `dTLB MPKI` 随优化升至 `21.22` 的“内存墙”转移，头号瓶颈 100% 收敛至 `grid_search` (`Simulation.c:387/391`)<br>• 报告：`docs/xsbench_case_study_report.md` |
| **5. 评估模型与消融实验（第 4.5 节 & 数据集）** | **已有历史版本（待按新口径重跑）** | • 主线 Pairwise Transformer（`model_transformer.pt`）、目标函数消融（`objective_ablation.md`）、单程序打分与校准均已跑通<br>• **注意**：`train_set/` 的 509 条样本采集于采集器重构之前（旧版 `bcc`+`kprobe`），建议重采重训以统一全篇口径 |
| **6. 其余专项实验与案例（第 5.3 / 5.4.2 节）** | **待在本机运行** | • 参数敏感性扫描：`experiments/sensitivity/run_sensitivity.sh`<br>• 重复运行稳定性：`experiments/stability/run_stability.sh`<br>• BOLT 代码重排案例（5.4.2 节）：`experiments/llvm_test_suite/bolt_optimize.sh` |

---

## 二、 接下来建议的推进顺序

1. **跑通方法学专项与 BOLT 案例（脚本已就绪，可直接运行）**：
   - 运行 `run_sensitivity.sh` 与 `run_stability.sh`，补全第 5.3 节的时间窗粒度 Pareto 权衡与重复运行变异系数表；
   - 运行 `bolt_optimize.sh` 案例，补全第 5.4.2 节（BOLT 代码重排前后 iTLB / 缓存阻力转移）。
2. **统一评估模型数据口径（可选但推荐）**：
   - 使用最新版采集器重跑一轮 `llvm_test_suite` 数据集采集与 `train_transformer.py`，使模型训练数据与系统评测数据完全对齐。
3. **论文正文撰写与图表汇总**：
   - 第 3、4、5 章的核心方法、算法与主要实验数据均已齐备，可直接结合 `docs/` 下的系列报告开展论文正文撰写。
