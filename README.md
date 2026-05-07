# ebpf-mem-profiler

> 面向细粒度进程访存指标提取与归因分析的 eBPF 实验平台  
> 论文：**基于 eBPF 的细粒度进程访存性能指标提取与分析方法研究**

---

## 定位与职责

本仓库是论文的**主线实验仓库**，承担以下核心任务：

| 能力 | 说明 |
|------|------|
| eBPF 数据面 | CO-RE 风格内核程序 + BCC Python 原型，追踪 LLC / TLB / page fault，并可选导出 LBR |
| 细粒度指标提取 | 按 PID 或 TID 和时间窗聚合，输出标准化 JSONL |
| 函数级热点归因 | 符号化采样地址 → 函数 / 文件 / 行号（P2 阶段） |
| 分析报告生成 | 时序图、热点条形图、指标相关性热力图（matplotlib PDF） |
| 测量方法学验证 | 采集开销 / 重复稳定性 / 参数敏感性 / 微基准校验实验 |
| 基线对接 | `export/to_baseline.py` 将窗口指标转换为 [ebpf-mem-analyzer](../ebpf-mem-analyzer) 可消费的 CSV |

与 `ebpf-mem-analyzer` 的关系：两者通过 `window_metrics.jsonl` 和 CSV 转换脚本弱连接，**不共享代码**，互不破坏各自的实验结论。

---

## 目录结构

```
ebpf-mem-profiler/
├── bpf/
│   ├── mem_events.bpf.c        # CO-RE eBPF 内核程序（libbpf / Makefile）
│   └── mem_events.h            # 内核 ↔ 用户态共享类型
├── src/
│   ├── bcc_prog.c              # BCC 兼容版 eBPF 程序（Python 原型加载）
│   ├── loader.py               # CLI 入口：采集 session 管理
│   ├── collector.py            # BCC 加载、map 读取、差分计算
│   ├── filter.py               # /proc 扫描：comm → PID 解析
│   └── exporter.py             # 写入 run_metadata.jsonl / window_metrics.jsonl
├── analysis/
│   ├── symbolize.py            # /proc/maps + addr2line 符号化
│   ├── hotspot.py              # 热点识别、时序 CSV（P1）
│   ├── attribution.py          # 函数级归因（P2，需 --emit-events）
│   └── report.py               # matplotlib 图表生成
├── export/
│   └── to_baseline.py          # 格式转换适配器
├── scripts/
│   ├── freeze_curated_manifest.py # 冻结 145x4 curated run list
│   ├── audit_train_set_quality.py # 生成当前数据质量审计与问题样本清单
│   ├── build_run_features.py   # 原始窗口 → 运行级特征
│   ├── build_pair_table.py     # 运行级特征 → pair 数据
│   └── build_anchor_set.py     # pair 结果 → 锚点集
├── experiments/
│   ├── llvm_test_suite/       # llvm-test-suite 提取与 PMU 采集脚本
│   ├── overhead/               # P3：采集开销测试
│   ├── stability/              # P3：重复运行稳定性测试
│   ├── sensitivity/            # P3：参数敏感性测试
│   └── micro_benchmark/        # P3：微基准校验
├── third_party/
│   └── llvm-test-suite/        # llvm-test-suite submodule
├── data/                       # 原始采集数据（gitignore）
├── results/                    # 分析结果与图表（gitignore）
├── docs/
│   └── design.md               # 系统设计文档
├── Makefile                    # 编译 CO-RE eBPF 程序
└── requirements.txt            # Python 依赖
```

---

## llvm-test-suite 数据口径

当前仓库里和 llvm-test-suite 相关的数据，需要明确区分 raw 采集层和训练快照层。

1. 最新 raw manifests 位于 `data/llvm_test_suite/manifest_bcc_O0~O3.jsonl`，curated manifests 位于 `data/llvm_test_suite/manifest_curated_O0~O3.jsonl`。
2. 当前 raw 与 curated 两层都已经收敛为严格的 `145 x 4`：四个 variant 各 145 条记录，`shared_program_count=145`。
3. 当前 `train_set` 不是“另一轮更早的冻结快照”，而是从这轮 curated manifests 继续经过下游过滤得到的训练子集：`580 curated runs -> 509 run_features -> 1494 pairs -> 374 anchors`。
4. 如果要从最新 raw data 重建运行级特征，应优先使用 curated manifests，而不是直接顺序扫描 raw manifests。
5. 当前完整的问题样本清单、缺失变体程序和 O2/O3 难例分流建议，统一由 `python scripts/audit_train_set_quality.py` 生成，产物为 `train_set/data_quality_audit.json` 与 [docs/new-repo-plan/current-data-quality-audit.md](docs/new-repo-plan/current-data-quality-audit.md)。

---

## 环境准备

### 系统依赖

```bash
# Ubuntu 22.04 / 24.04
sudo apt install \
    clang llvm libbpf-dev bpftool \
    linux-headers-$(uname -r) \
    python3-bcc \
    binutils   # addr2line / nm
```

> **内核版本要求**：Linux >= 5.8（BPF ring buffer）

### Python 依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### （可选）编译 CO-RE 生产版 eBPF 程序

```bash
make
# 产出：bpf/mem_events.bpf.o  bpf/mem_events.skel.h
```

### 采集指定进程的访存事件

```bash
# 需要 root 权限或 CAP_BPF + CAP_PERFMON
sudo python src/loader.py --pid <PID> --window 1.0 --output data/run_001/

# 按进程名（自动解析 PID）
sudo python src/loader.py --comm nginx --window 1.0 --output data/run_001/ --duration 60

# 按线程聚合并只观察指定 TID
sudo python src/loader.py --pid <PID> --per-tid --tid <TID> --output data/run_tid/

# 启用逐事件与 LBR 分支栈记录（P2 归因分析所需）
sudo python src/loader.py --pid <PID> --emit-events --lbr --output data/run_001/
```

### 热点分析（P1：PID 级）

```bash
python analysis/hotspot.py \
    --data   data/run_001/ \
    --output results/run_001/ \
    --metric llc_load_misses \
    --top 20
```

### 批量热点分析（多次 run 目录）

适用于 `data/llvm_test_suite/bcc/O3-g` 这类目录下包含多个 `aha_*` 运行子目录的数据集。

```bash
python analysis/dataset_hotspot.py \
    --data-root data/llvm_test_suite/bcc/O3-g \
    --output results/llvm_test_suite/aha_O3-g_hotspots \
    --metric llc_load_misses \
    --top 20
```

默认输出：

- `run_hotspot_summary.csv/jsonl`：每个 run 的热点窗口数量、最大热点分数、指标总量
- `dataset_hotspots_<metric>.csv/jsonl`：跨 run 热点窗口排行
- `dataset_attribution_<metric>.csv/jsonl`：每个热点窗口的 Top-N 归因实体
- `entity_hotspots_<metric>.csv/jsonl`：按 run 内 PID/TID 聚合的热点实体摘要

一次分析所有指标：

```bash
python analysis/dataset_hotspot.py \
    --data-root data/llvm_test_suite/bcc/O3-g \
    --output results/llvm_test_suite/aha_O3-g_hotspots \
    --all-metrics \
    --top 20
```

多指标模式还会额外输出：

- `metrics_overview.csv/jsonl`：所有指标的热点窗口数量、峰值热点分数与最强热点窗口位置
- `run_hotspot_summary_<metric>.csv/jsonl`：按指标拆分的 run 级摘要

绘制跨 run 热点图：

```bash
python analysis/dataset_hotspot_report.py \
    --results results/llvm_test_suite/aha_O3-g_hotspots \
    --output results/llvm_test_suite/aha_O3-g_hotspots/figures \
    --top 10
```

默认生成：

- `metrics_overview.pdf`：多指标热点总览
- `dataset_hotspots_<metric>.pdf`：跨 run 热点窗口条形图
- `entity_hotspots_<metric>.pdf`：热点归因实体条形图

### 一键生成归因报告

默认直接读取 `data/llvm_test_suite/bcc/O3-g`，一次跑完全部预定义指标，并输出到 `results/llvm_test_suite/aha_O3-g_attribution_report`：

```bash
python analysis/attribution_report.py
```

如果只想跑单个指标，或者改数据目录、输出目录：

```bash
python analysis/attribution_report.py \
    --data-root data/llvm_test_suite/bcc/O3-g \
    --output results/llvm_test_suite/custom_attribution_report \
    --metric dtlb_misses
```

默认产出：

- `dataset_attribution_<metric>.csv/jsonl`：每个指标对应的热点窗口归因实体明细
- `entity_hotspots_<metric>.csv/jsonl`：每个指标对应的 run 内热点实体摘要
- `run_hotspot_summary_<metric>.csv/jsonl`：多指标模式下每个指标的 run 级汇总
- `metrics_overview.csv/jsonl`：指标总览
- `attribution_report.md`：Markdown 归因摘要
- `figures/*.pdf`：热点窗口、归因实体及多指标总览图表

### 一键生成指标时序关系报告

默认直接读取 `data/llvm_test_suite/bcc/O3-g`，输出到 `results/llvm_test_suite/aha_O3-g_metric_relations`：

```bash
python analysis/metric_relation_report.py
```

也可以自定义数据目录、输出目录和滞后窗口范围：

```bash
python analysis/metric_relation_report.py \
    --data-root data/llvm_test_suite/bcc/O3-g \
    --output results/llvm_test_suite/custom_metric_relations \
    --max-lag 8
```

默认产出：

- `run_metric_relation_summary.csv/jsonl`：每个 run 的可用指标数、指标对数和最强指标对
- `dataset_metric_pairs.csv/jsonl`：跨 run 的指标对明细，包含 `pearson_r`、`peak_lag`、`co_spike_count`
- `metric_pair_overview.csv/jsonl`：按指标对汇总的均值相关性、主导滞后和联合热点统计
- `metric_relation_report.md`：Markdown 时序关系摘要
- `figures/*.pdf`：指标对强度总览和联合热点总览图表

### 函数级归因（P2，需先以 `--emit-events` 采集）

```bash
python analysis/attribution.py \
    --data    data/run_001/ \
    --pid     <PID> \
    --binary  /path/to/target_binary \
    --output  results/run_001/ \
    --metric  llc_load_misses
```

### 生成图表

```bash
python analysis/report.py \
    --results results/run_001/ \
    --output  results/run_001/figures/
```

### 与基线仓库对接

```bash
python export/to_baseline.py \
    --input  data/run_001/ \
    --output /path/to/ebpf-mem-analyzer/data/new_input/
```

### llvm-test-suite 数据集准备

```bash
git submodule update --init --recursive

# 在 third_party/llvm-test-suite 中完成构建后，提取 ELF 和 .test 运行规格
bash experiments/llvm_test_suite/extract_elf.sh -n

# 正式提取
bash experiments/llvm_test_suite/extract_elf.sh -b build-O1 -v O1

# 基于提取结果执行单个 VARIANT 的 BCC 采集（默认 VARIANT=O3）
sudo bash experiments/llvm_test_suite/collect_dataset_testbench.sh

# 一次顺序采集 data/llvm_test_suite 下全部 VARIANT（自动检测 O0/O1/O2/O3）
sudo bash experiments/llvm_test_suite/collect_dataset_all_variants.sh

# 将最新 raw manifests 冻结为可复现的 145x4 curated run list
python scripts/freeze_curated_manifest.py --data-root data/llvm_test_suite

# 基于 curated manifests 重建运行级特征
python scripts/build_run_features.py \
    --data-root data/llvm_test_suite \
    --manifest-prefix manifest_curated \
    --output train_set_curated
```

默认输出：
- `data/llvm_test_suite/bin/<VARIANT>`：提取的 ELF
- `data/llvm_test_suite/test/<VARIANT>`：对应 .test 与运行时文件
- `data/llvm_test_suite/bcc/<VARIANT>/<bench>_<timestamp>/`：BCC JSONL 采集结果
- `data/llvm_test_suite/manifest_bcc_<VARIANT>.jsonl`：批量采集清单
- `data/llvm_test_suite/manifest_curated_<VARIANT>.jsonl`：冻结后的 145x4 curated run list
- `data/llvm_test_suite/manifest_curated_summary.json`：curated run list 汇总信息
- `results/llvm_test_suite/log/`：批量脚本运行日志

说明：

1. `manifest_bcc_<VARIANT>.jsonl` 表示最新 raw 采集清单，不保证天然是干净的 `145 x 4` 训练输入。
2. `manifest_curated_<VARIANT>.jsonl` 才是“按最新完整 run 冻结”后的可复现 run list。
3. 如果后续要重建 `run_features`、`pairs` 或 `anchor_set`，建议从 curated manifests 开始。

---

## 方法学验证实验

```bash
# P3-1：采集开销测试
sudo bash experiments/overhead/run_overhead.sh

# P3-2：重复运行稳定性（指定目标进程）
sudo bash experiments/stability/run_stability.sh --pid <PID> --repeat 10

# P3-3：参数敏感性扫描
sudo bash experiments/sensitivity/run_sensitivity.sh --pid <PID>

# P3-4：微基准校验（验证方向正确性）
sudo bash experiments/micro_benchmark/run_micro_bench.sh
```

---

## 阶段进度

| 阶段 | 目标 | 状态 |
|------|------|------|
| **P0** | 边界定义 · 目录结构 · 最小输出链路 | ✅ 完成 |
| **P1** | 最小可用 eBPF 原型（稳定 attach · PID 过滤 · 时间窗落盘） | 🔲 待验证 |
| **P2** | 函数级热点归因 | 🔲 待开发 |
| **P3** | 测量方法学验证实验 | 🔲 待开发 |
| **P4** | 与基线仓库弱连接 · 补充对比实验 | 🔲 待开发 |

---

## 核心输出文件

当前主输出文件：
- `window_metrics.jsonl`：时间窗级聚合指标（主要数据文件）
- `run_metadata.jsonl`：采集 session 元信息
- `hotspot_summary.jsonl`：热点摘要（analysis/ 脚本产出）

---

## 已知限制

- **BCC 原型**（P1）：当前通过 raw perf attr 绑定更多 cache/TLB 事件，但不同 CPU 微架构对事件可用性支持不同；不可用事件会在启动时被自动跳过。
- **符号化**：无 DWARF 调试信息的二进制文件，addr2line 仅返回 `??:0`，建议以 `-g` 编译目标程序。
- **CO-RE 版本**：`bpf/mem_events.bpf.c` 需通过 `make` 编译生成 skeleton header 后才能在 libbpf 用户态程序中使用。

详见 [docs/design.md#已知限制](docs/design.md#已知限制p1-阶段)。
