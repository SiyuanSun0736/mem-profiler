# 设计文档

> 状态：草稿  
> 最后更新：2026-04

本文档描述 `ebpf-mem-profiler` 的系统设计思路，包括架构分层、各模块职责、与基线仓库的关系，以及各阶段任务边界。

---

## 整体定位

本仓库是论文"基于 eBPF 的细粒度进程访存性能指标提取与分析方法研究"的**主线实验仓库**。

| 仓库 | 职责 | 阶段 |
|------|------|------|
| `ebpf-mem-profiler`（本库） | eBPF 采集 · 细粒度指标提取 · 归因分析 · 方法学验证 | 题目主线 |
| `ebpf-mem-analyzer` | PMU/LBR 相对性能预测 · Siamese 模型 · 已有实验结论 | 基线子项目 |

两个仓库通过**稳定数据接口**（`window_metrics.jsonl`）连接，**不共享代码实现**。

---

## 架构分层

```
┌─────────────────────────────────────────────────┐
│                  用户接口层                       │
│  src/loader.py  （CLI 入口，信号处理，采集循环）  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                  采集控制层                       │
│  src/collector.py   （BCC 加载 · map 读取 · 差分）│
│  src/filter.py      （PID/comm 解析）             │
│  src/exporter.py    （JSONL 写入）                │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                  eBPF 数据面                      │
│  src/bcc_prog.c     （BCC 原型 eBPF 程序）        │
│  bpf/mem_events.bpf.c（CO-RE 生产版 eBPF 程序） │
│  bpf/mem_events.h   （共享类型定义）              │
└──────────────────────┬──────────────────────────┘
                       │  window_metrics.jsonl
┌──────────────────────▼──────────────────────────┐
│                  离线分析层                       │
│  analysis/hotspot.py    （热点识别 · 时序分析）   │
│  analysis/attribution.py（函数级归因，P2）        │
│  analysis/symbolize.py  （addr2line 封装）        │
│  analysis/report.py     （图表生成）              │
└──────────────────────┬──────────────────────────┘
                       │  baseline_input.csv
┌──────────────────────▼──────────────────────────┐
│                  导出 / 对接层                    │
│  export/to_baseline.py  （格式转换适配器）        │
│  export/schema/         （JSON Schema 定义）      │
└─────────────────────────────────────────────────┘
```

---

## 事件类型与采集机制

| 事件 | 内核机制 | eBPF 程序类型 | 说明 |
|------|---------|--------------|------|
| LLC load miss | `PERF_COUNT_HW_CACHE_MISSES` | `SEC("perf_event")` | 周期性采样，sample_rate 可调 |
| LLC store miss | 同上（部分 CPU 支持单独计数） | `SEC("perf_event")` | 硬件支持时有效 |
| dTLB load miss | `PERF_COUNT_HW_CACHE_MISSES`（fallback） | `SEC("perf_event")` | 理想情况需 RAW event |
| minor page fault | `kprobe/handle_mm_fault` | `SEC("kprobe/...")` | `FAULT_FLAG_MAJOR` 位区分 major/minor |
| major page fault | 同上 | 同上 | |

---

## 阶段任务边界

### P0（当前）：边界定义与仓库初始化

- [x] 确定目录结构和模块分工
- [x] 定义数据协议（docs/data_protocol.md）
- [x] 创建所有核心文件骨架

### P1：最小可用 eBPF 原型

目标：能稳定 attach、能按 PID 过滤、能按时间窗输出结果、能落盘为 JSONL。

- [ ] 在真实 Linux 环境运行 `src/loader.py`，验证 BCC 加载正常
- [ ] 验证 LLC miss kprobe 数据与 `perf stat` 量级一致（10% 误差内）
- [ ] 验证 page fault 计数与 `/proc/<pid>/stat` 字段一致
- [ ] 写入完整 `run_metadata.jsonl` + `window_metrics.jsonl`

### P2：函数级热点归因

目标：将采样 IP 地址映射到函数名，输出 `function_hotspot.jsonl`。

- [ ] 验证 `--emit-events` 模式下 `events.jsonl` 的地址格式合法
- [ ] 验证 `analysis/symbolize.py` 能正确处理 PIE（位置无关可执行文件）的地址偏移
- [ ] 对标 `perf report` 输出，验证 Top-5 热点函数与本工具一致
- [ ] 写入论文 case study 图表（`analysis/report.py`）

### P3：测量方法学验证

目标：量化采集开销、稳定性和参数敏感性。

- [ ] 完成 `experiments/overhead/run_overhead.sh`，报告 CPU overhead < X%
- [ ] 完成 `experiments/stability/run_stability.sh`，报告各指标 CV < 15%
- [ ] 完成 `experiments/sensitivity/run_sensitivity.sh`，确定 sample_rate 推荐值
- [ ] 完成 `experiments/micro_benchmark/run_micro_bench.sh`，验证方向正确性

### P4：与基线仓库弱连接

- [ ] 运行 `export/to_baseline.py`，生成 `baseline_input.csv`
- [ ] 验证基线仓库能读取新 CSV 并正常执行推理
- [ ] 撰写"新特征 vs 旧特征"对比实验报告

---

## 已知限制（P1 阶段）

1. **dTLB miss**：现有 fallback 实现（共享 CACHE_MISSES perf event）并非真正独立的 dTLB 计数器，后续可通过 `PERF_TYPE_RAW` 配置具体硬件 event code。

2. **LLC store miss**：Intel 架构下 BCC 的 `PerfHWConfig.CACHE_MISSES` 主要计数 LLC load miss，store miss 需要单独配置原始事件或使用 `perf_event_attr.config1`。

3. **符号化精度**：无 DWARF 调试信息时，`addr2line` 仅能返回 `??:0`，需要 `-g` 编译目标程序。

4. **内核版本**：ring buffer（`BPF_MAP_TYPE_RINGBUF`）需要 Linux >= 5.8；`PERCPU_HASH` 差分逻辑需要 >= 5.2。

---

## 参考资料

- [BPF CO-RE Reference Guide](https://nakryiko.com/posts/bpf-core-reference-guide/)
- [libbpf API 文档](https://libbpf.readthedocs.io/)
- [BCC Python Developer Tutorial](https://github.com/iovisor/bcc/blob/master/docs/tutorial_bcc_python_developer.md)
- [Linux Perf Events ABI](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- `perf list` — 查看当前内核/CPU 支持的硬件事件列表
