# 方法学验证阈值、表格与默认配置

本文档把“开销、稳定性、参数敏感性”从“有脚本”收敛成固定的阈值、固定的表格产物和固定的推荐配置。

它和 [docs/correctness-validation.md](./correctness-validation.md) 的分工不同：

1. correctness-validation 负责回答“eBPF 指标对不对”。
2. methodology-validation 负责回答“这套方法能不能低开销、稳定地用，并且默认该怎么配”。

统一生成入口是 [scripts/build_methodology_tables.py](../scripts/build_methodology_tables.py)。

---

## 1. 固定验收阈值

建议正文和答辩统一使用下面这套口径：

| 维度 | 目标阈值 | 可接受阈值 | 超阈后的解释口径 |
| --- | --- | --- | --- |
| 开销 | 相对开销 `<= 5%` | `<= 10%` | 超过 `10%` 不作为默认常规配置 |
| 稳定性 | `CV <= 10%` | `CV <= 15%` | 超过 `15%` 只做趋势或热点排序解释 |
| 采样率敏感性 | 相对参考配置最大漂移 `<= 10%` | `<= 20%` | 超过 `20%` 不作为默认采样率 |
| 窗口长度敏感性 | 相对参考配置最大漂移 `<= 10%` | `<= 20%` | 超过 `20%` 不作为默认窗口长度 |
| 探针组合 | `llc + dtlb + fault` 全开 | 无 | 精简探针只作为定向诊断模式 |

这里的参考配置固定为：

1. `samplerate_100`
2. `window_1.0s`
3. `probe_all`

敏感性表里的漂移统一按“每秒速率”计算，而不是直接比总量，避免窗口长度变化引入虚假的规模偏差。

---

## 2. 推荐默认配置

### 2.1 常规采集默认配置

建议把下面这套写成正文默认设置：

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| `window` | `1.0s` | 在时间定位能力和跨 run 稳定性之间取平衡 |
| `sample-rate` | `100` | 作为默认参考点；若后续敏感性表显示更稀疏采样仍在 10% 漂移预算内，可再上调 |
| probe profile | `probe_all` | 保持 `llc + dtlb + itlb + fault + mm_syscalls` 覆盖 |
| `emit-events` | 关闭 | 不作为低开销默认配置的一部分 |
| `lbr` | 关闭 | 只在一轮性深度诊断中开启 |

推荐写法：常规采集使用 `window=1.0s`、`sample-rate=100`、`probe_all`，并保持 `emit-events` 与 `lbr` 关闭，以满足低开销和稳定性预算。

### 2.2 深度归因配置

如果目标是函数级定位，而不是长期低开销观测，则可在常规配置基础上：

1. 开启 `--emit-events`
2. 必要时再开启 `--lbr`

推荐写法：逐事件和 LBR 属于诊断模式，不计入低开销默认配置。

---

## 3. 固定表格产物

运行方法学实验后，结果目录里应固定出现以下表：

1. `overhead_summary.csv/md`
2. `stability_summary.csv/md`
3. `sensitivity_samplerate_summary.csv/md`
4. `sensitivity_window_summary.csv/md`
5. `sensitivity_probe_summary.csv/md`
6. `methodology_recommendations.csv/md`
7. `methodology_validation.md`

其中：

1. `overhead_summary.*` 给出 wall-time 与可选 perf stat 对照，并直接标记 `pass / caution / fail`。
2. `stability_summary.*` 给出各指标的 `mean_rate_per_sec`、`std_rate_per_sec` 和 `CV`。
3. `sensitivity_*_summary.*` 给出相对参考配置的 `median_drift_pct`、`max_drift_pct` 和默认配置推荐。
4. `methodology_recommendations.*` 固化论文正文可直接引用的阈值和推荐设置。
5. `methodology_validation.md` 把所有已生成表合并成一份总表，便于直接贴到论文材料里。

---

## 4. 怎么生成这些表

### 4.1 直接跑实验脚本

三类实验脚本现在都会在结束时自动调用汇总器：

```bash
# 开销
sudo bash experiments/overhead/run_overhead.sh

# 稳定性
sudo bash experiments/stability/run_stability.sh --pid <PID> --repeat 10

# 参数敏感性
sudo bash experiments/sensitivity/run_sensitivity.sh --pid <PID>
```

跑完后可直接查看对应结果目录下的 `*.md` 摘要表。

### 4.2 单独汇总已有结果目录

如果三类实验是分开跑的，也可以手工汇总：

```bash
python3 scripts/build_methodology_tables.py \
  --overhead-dir results/overhead_20260515_120000 \
  --stability-dir results/stability_20260515_121500 \
  --sensitivity-dir results/sensitivity_20260515_123000 \
  --output results/methodology_tables
```

如果某次只做其中一类实验，只传入对应目录即可。

---

## 5. 论文里怎么写

建议把三类结论分开表达，而不是混成一个笼统结论。

### 5.1 开销

推荐口径：默认配置下的采集开销以 `5%` 为目标预算，`10%` 为可接受上界；超过该上界时，不作为常规持续采集配置。

### 5.2 稳定性

推荐口径：仅对 `CV <= 10%` 的指标做绝对量比较；`10% < CV <= 15%` 的指标保留但需同时报告波动；`CV > 15%` 的指标仅做趋势判断或热点排序解释。

### 5.3 参数敏感性

推荐口径：默认采样率和窗口长度应满足相对参考配置 `10%` 以内的最大漂移预算；若某设置超过 `20%` 漂移，不应作为默认配置。

### 5.4 探针组合

推荐口径：`probe_all` 作为常规配置；`probe_llc_only` 或 `probe_fault_only` 仅作为定向诊断模式，不替代正文默认设置。

---

## 6. 与正确性验证的衔接

建议正文顺序如下：

1. 先用 [docs/correctness-validation.md](./correctness-validation.md) 证明“采到的指标方向和量级是对的”。
2. 再用本文档证明“默认配置开销可控、结果稳定、参数可解释”。
3. 最后再进入 case study、热点定位和函数级归因。

这样论文主线会更清楚：先证明方法可信，再证明方法好用。