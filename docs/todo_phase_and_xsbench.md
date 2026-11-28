# 待补充任务清单：阶段自动切分算法 & XSBench 科学计算案例

> 记录当前毕设代码与实验尚缺的两项核心模块，便于后续直接推进实现与论文撰写。

---

## 1. 阶段自动切分算法（对应论文 4.2 节）

### 1.1 现状与缺口
- **现有能力**：`analysis/hotspot.py` 已实现基于时间窗序列的异常窗口检测（局部均值/方差/Z-Score 阈值）以及 CPU-访存联合共振（Co-Spike）检测。
- **缺失部分**：尚未实现**程序运行周期的阶段自动切分算法**，即从 `window_metrics.jsonl` 连续时序中自动区分：
  1. **初始化/预热期（Init / Warmup Phase）**：大量 `mmap`、`minor_faults`、指令/数据缓存冷加载与工作集建立阶段；
  2. **核心计算稳态期（Steady-state Compute Phase）**：`minor_faults` 回落、IPC 与访存失效率（LLC/dTLB MPKI）进入平稳或周期性波动的核心计算阶段；
  3. **收尾/释放期（Teardown Phase）**：`munmap` 集中释放或负载收尾阶段（可选）。

### 1.2 实现规划
- **核心文件**：`analysis/phase_segmentation.py`（或集成至 `analysis/hotspot.py` 与 `analysis/report.py`）。
- **算法思路**：
  - 输入 `window_metrics.jsonl` 时间窗序列，构造多维特征向量（如 `minor_faults/s`、`mmap_bytes/s`、`ipc`、`llc_miss_rate`、`dtlb_miss_rate`）；
  - 采用基于**滑窗变化率 / 变点检测（Change-Point Detection，如 CUSUM 或滑动双窗方差比）**结合**虚存事件特征规则（如 `minor_faults` 峰值回落且 `ipc` 方差收敛）**，自动定位阶段边界 `t_warmup_end`；
  - 输出每个阶段的起止时间窗、阶段类型标签（`init` / `steady` / `teardown`）及各阶段聚合统计摘要，供 `report.py` 与论文 4.2 节 / 5.3 节（时序切片分辨力验证）引用。

---

## 2. XSBench 科学计算案例（对应论文 5.4.1 节）

### 2.1 现状与缺口
- **现有能力**：`experiments/correctness/run_reference_capture.sh` 仅在帮助示例注释中提及 `xsbench`，仓库内尚未引入 XSBench 源码与编译脚本，也未开展多编译优化变体下的全链路归因实验。
- **缺失部分**：论文 5.4.1 节《案例一：科学计算应用在不同优化变体下的全链路下钻归因》所需的**真实访存密集型科学计算负载实验与归因报告**。

### 2.2 实现规划
- **基准准备**：在 `experiments/xsbench/`（或 `experiments/case_studies/xsbench/`）下引入 XSBench（OpenMP 蒙特卡洛中子输运宏截面查找基准），保留 `-g -fno-omit-frame-pointer` 调试符号，构建多组对照变体：
  - **编译优化变体**：如 `-O0` vs `-O2` / `-O3`；
  - **或访存模式变体**：如 XSBench 原生支持的 `event` vs `history` 模式，或不同线程数/网格规模（`-g` / `-p`）。
- **全链路下钻实验**：
  1. 使用阶段切分 + 异常窗口识别，自动区分 XSBench 网格构建初始化期（大量 `minor_faults`）与核截面随机查表计算期（高 LLC Miss / dTLB Miss）；
  2. 锁定核心计算窗口内的责任 Worker 线程（`per-tid` 贡献度排序 + Co-Spike）；
  3. 触发 `--emit-events` 与符号化归因，精确收敛至核截面二分查找与散列查表源码行（如 `Calculate_micro_xs` / `grid_search`）；
  4. 对比跨优化变体下宏观指标（IPC、LLC MPKI、dTLB MPKI）与瓶颈函数分布的收敛趋势，生成案例报告用于撰写论文 5.4.1 节。
