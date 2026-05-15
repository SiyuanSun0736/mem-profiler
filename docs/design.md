# 设计文档

> 状态：工作基线  
> 最后更新：2026-05

本文档不再按早期 P0/P1/P2 计划稿来描述仓库，而是直接按论文题目“基于 eBPF 的细粒度进程访存性能指标提取与分析方法研究”整理当前主线、现状证据、剩余工作和优化优先级。

---

## 1. 论文主线定位

当前仓库真正服务的主问题是：

1. 如何用 eBPF 在进程或线程粒度上持续采集访存相关运行时指标。
2. 如何把原始采样组织成时间窗级、实体级和函数级的细粒度分析对象。
3. 如何证明这套提取与分析方法在正确性、可解释性和工程可用性上是可信的。

按这个题目收口，仓库里的内容应分成两层：

| 层级 | 是否属于题目主线 | 当前对应内容 |
|------|------------------|-------------|
| eBPF 采集、窗口聚合、热点归因、函数级证据、方法学验证 | 是 | `src/`、`bpf/`、`analysis/`、`experiments/` |
| 基于运行级摘要做 pairwise 训练、锚点评分、单程序优化程度推断 | 属于扩展分析能力 | `scripts/train_transformer.py`、`scripts/score_program.py`、[docs/dual-tower-architecture.md](./dual-tower-architecture.md) |

这意味着文档和论文叙事都应把“细粒度指标提取与分析方法”放在前面，把双塔评分作为扩展应用或增强分析，而不是反过来让模型成为主角。

---

## 2. 主线闭环

从题目出发，当前仓库已经形成了比较完整的技术闭环：

```
eBPF / perf_event / kprobe
          ↓
按 PID/TID 聚合的窗口级观测
          ↓
window_metrics.jsonl / events.jsonl / run_metadata.jsonl
          ↓
热点检测 / 时序关系 / 函数级归因 / 报告生成
          ↓
方法学验证（开销、稳定性、敏感性、微基准）
          ↓
可选扩展：跨 run 建模与单程序评分
```

对应的主线文件如下：

| 环节 | 主要文件 | 当前作用 |
|------|---------|---------|
| 在线采集入口 | `src/loader.py` | 调度采集周期、参数和输出目录 |
| 采集与窗口聚合 | `src/collector.py` | 绑定 perf_event / kprobe / ring buffer，聚合为 `WindowSnapshot` |
| eBPF 数据面 | `src/bcc_prog.c`、`bpf/mem_events.bpf.c`、`bpf/mem_events.h` | 提供 PMU、fault、LBR、mm syscall 等观测入口 |
| 数据落盘 | `src/exporter.py` | 写出 `run_metadata.jsonl`、`window_metrics.jsonl`、`events.jsonl` |
| 单程序分析 | [docs/single-program-analysis-architecture.md](./single-program-analysis-architecture.md) | 对应题目中的“细粒度指标提取与分析”主链路 |
| 数据集级分析 | `analysis/dataset_hotspot.py`、`analysis/metric_relation_report.py`、`analysis/attribution_report.py` | 形成批量统计、关系分析和图表产物 |
| 扩展建模 | [docs/dual-tower-architecture.md](./dual-tower-architecture.md) | 基于运行级摘要的 pairwise 训练和单程序评分 |

---

## 3. 文档阅读顺序

当前最合理的阅读顺序不是先看双塔，而是按题目主线逐层看：

1. [README.md](../README.md)：看仓库定位、命令入口、数据口径和当前支持的实验面。
2. [docs/single-program-analysis-architecture.md](./single-program-analysis-architecture.md)：看“单程序采集到归因报告”的主链路。
3. 本文档：看论文口径下的现状、缺口与优先级。
4. [docs/dual-tower-architecture.md](./dual-tower-architecture.md)：只在需要写“扩展分析/评分能力”时再读。
5. [docs/new-repo-plan/current-data-quality-audit.md](./new-repo-plan/current-data-quality-audit.md)：看训练/评估口径、过滤链路和当前缺口。

说明：`docs/new-repo-plan/` 下部分文件仍保留较早期的快照假设，当前应优先以 [README.md](../README.md)、本文档、`train_set/` 产物和 `current-data-quality-audit.md` 为准。

---

## 4. 当前已经完成到什么程度

### 4.1 采集与指标层

目前已经不只是“最小原型”。从代码和文档口径看，仓库已具备以下采集面：

1. LLC、dTLB、iTLB、cycles、instructions 等 PMU 指标。
2. minor / major page fault，以及 anon/file/shared/private/write/instruction fault subtype。
3. `mmap / munmap / mprotect / brk` 等 mm syscall 计数与字节量。
4. 可选 LBR 分支栈采样和逐事件 `events.jsonl` 证据流。
5. 按 PID 或 TID 聚合的窗口级输出协议。

这部分已经基本满足“细粒度进程访存性能指标提取”的系统实现要求。

### 4.2 单程序与数据集分析层

当前离线分析层已经形成三级证据：

1. 程序级和时间窗级热点：`analysis/hotspot.py`。
2. 数据集级热点与指标关系：`analysis/dataset_hotspot.py`、`analysis/metric_relation_report.py`。
3. 函数级归因：`analysis/attribution.py`、`analysis/symbolize.py`、`analysis/attribution_report.py`。

这意味着题目中的“分析方法”部分已经有比较具体的落地，不再只是采完数据再人工看表。

### 4.3 数据与评估口径

按当前仓库产物，数据链路已经有较明确的规模：

1. raw manifests 与 curated manifests 已收敛到严格的 `145 x 4`。
2. 当前训练链路使用的是过滤后的子集：`580 curated runs -> 509 run_features -> 1494 pairs -> 374 anchors`。
3. 当前过滤后覆盖 132 个程序，其中完整四变体程序 122 个。

这些数字说明仓库已经不是“只有少量 case study 样例”，而是有一套可复现的数据整理链路。

### 4.4 扩展建模层

如果你打算把双塔作为扩展章节，当前也已经有可写的结果：

1. `train_set/model_transformer_eval.json` 显示 test `dir_acc=0.9020`、`acc_3cls=0.7958`。
2. `train_set/score_eval.json` 显示单程序评分 `corr_score_log=0.9072`、`mae_score_log=0.2723`。
3. `train_set/anchor_set.stats.json` 显示当前锚点数为 374，默认锚点变体为 `O0/O2/O3`。

但按论文题目，这部分应放在“基于提取指标的扩展应用”或“进一步分析尝试”，不宜替代题目主线。

---

## 5. 还需要做什么

下面这些工作里，前四项是论文主线必须补齐的，后两项属于提升论文完整度和说服力的增强项。

### 5.1 必须补齐

1. **补指标正确性验证**  
    需要把 eBPF 输出和外部基准真正对齐成论文可引用的结果，而不是只停留在脚本存在。至少应补三类对照：
    - PMU 指标对 `perf stat` 的量级一致性；
    - 函数级热点对 `perf report` 的热点一致性；
    - fault 计数对 `/proc` 或已知微基准行为的方向与数量一致性。

2. **把方法学验证收敛成固定阈值和默认配置**  
    这部分现在应统一落到 [docs/methodology-validation.md](./methodology-validation.md) 和 [scripts/build_methodology_tables.py](../scripts/build_methodology_tables.py)，而不是只保留“实验脚本存在”。正文建议直接固定下面四件事：
    - 开销目标阈值 `<= 5%`，可接受上界 `<= 10%`；
    - 指标稳定性统一用 `mean / std / CV` 描述，并以 `CV <= 10%` 作为稳定阈值；
    - 默认推荐配置写成 `window=1.0s`、`sample-rate=100`、`probe_all`；
    - 超过稳定性或敏感性预算的指标只做趋势判断，不做严格绝对量比较。

3. **补 2 到 3 个完整 case study**  
    题目里有“分析方法研究”，所以最好至少给出几组完整链路：
    - 程序级异常发现；
    - 时间窗级热点定位；
    - PID/TID 级归因；
    - 函数级证据回落到源码或二进制符号。

4. **把贡献边界重新写清楚**  
    论文和文档里要明确区分：
    - 主贡献：eBPF 细粒度访存指标提取、统一输出协议、多层归因分析和方法学验证；
    - 扩展贡献：基于这些指标做的 pairwise 建模和单程序评分。

### 5.2 建议增强

5. **补严格时间口径与覆盖缺口说明**  
    当前 `train_set` 里已有较清楚的数据质量审计，但论文中还应直接解释：
    - 为什么 145x4 curated 最后只剩 509 runs 可进入训练链路；
    - 为什么只剩 122 个完整四变体程序；
    - strict-time 过滤和缺失 O0 baseline 会如何影响结论边界。

6. **补难例分析而不是只报总指标**  
    当前 O2/O3 是最难区分的一组近邻变体，适合在论文里单列成“困难样本分析”，说明：
    - 近 tie 为什么难；
    - 仅靠运行时间和仅靠非时间特征各自的局限；
    - 细粒度时序或更稳定真值是否能改善判别。

---

## 6. 还需要优化什么

### 6.1 文档结构优化

1. 把本文档作为总入口，避免 README、design、dual-tower 三份文档互相抢主线。
2. 在 [docs/dual-tower-architecture.md](./dual-tower-architecture.md) 明确标注“扩展分析链路”，避免读者误以为模型是题目主体。
3. 对 `docs/new-repo-plan/` 下仍沿用旧快照口径的文件加显式说明，避免论文写作时误引旧数字。

### 6.2 实验表达优化

1. 把“脚本存在”升级为“图表和表格已固定输出”。
2. 为每个方法学实验固定统一摘要字段，例如 `mean`、`std`、`cv`、`overhead_pct`、`recommended_setting`；当前统一由 `scripts/build_methodology_tables.py` 生成。
3. 统一 case study 输出目录和图表命名，减少后期写论文时重新整理。

### 6.3 采集可信度优化

1. dTLB 与 LLC store miss 仍应优先评估 RAW event 支持，减少 proxy 含义不清的问题。
2. 如果后续要把 PMU 计量写得更硬，建议补 `time_enabled/time_running` 或 group leader 可见性。
3. 若要把 LBR 作为正式证据链的一部分，最好补一段说明硬件限制与继承模式限制。

### 6.4 题目贴合度优化

1. 章节标题里多用“提取协议、时间窗聚合、热点归因、函数证据、方法学验证”，少用“评分、模型、双塔”做一级标题。
2. 先证明“能可靠采、能稳定看、能解释到函数”，再展示“还能进一步做评分”。
3. 把双塔链路放在“应用示例”或“扩展研究”更稳，因为它依赖的前提正是前面的 eBPF 指标体系。

---

## 7. 建议的写作与答辩收口方式

如果你现在开始整理论文正文，最稳的结构是：

1. 先讲问题背景、现有工具不足和为什么要用 eBPF 做细粒度访存观测。
2. 再讲采集架构、指标体系、时间窗协议和多层归因链路。
3. 接着讲方法学验证，证明这套方法不是“能跑”，而是“可信”。
4. 然后给 case study，证明它确实能定位问题。
5. 最后再写扩展分析，例如双塔评分，把它作为“基于该指标体系的进一步应用”。

这个顺序和题目最一致，也最容易让评审接受：先证明方法成立，再展示方法能支持更复杂的分析任务。

---

## 8. 当前最值得优先完成的三件事

如果只按性价比排优先级，建议先做下面三项：

1. 固化一组“指标正确性 + 开销 + 稳定性”的主实验表格。
2. 选 2 到 3 个代表性程序，跑通“窗口热点 -> 实体归因 -> 函数归因”的完整 case study。
3. 统一文档口径，把双塔明确降级为扩展章节，避免论文主线跑偏。

---

## 9. 参考资料

- [BPF CO-RE Reference Guide](https://nakryiko.com/posts/bpf-core-reference-guide/)
- [libbpf API 文档](https://libbpf.readthedocs.io/)
- [BCC Python Developer Tutorial](https://github.com/iovisor/bcc/blob/master/docs/tutorial_bcc_python_developer.md)
- [Linux Perf Events ABI](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- `perf list`：查看当前内核和 CPU 支持的硬件事件列表
