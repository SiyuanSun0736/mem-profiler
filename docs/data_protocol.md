# 数据协议（Data Protocol）

> 版本：2.0  
> 文件格式：JSON Lines（每行一个 JSON 对象，UTF-8 编码，`.jsonl` 扩展名）

本文档定义 `ebpf-mem-profiler` 与下游消费方（包括 `ebpf-mem-analyzer` 基线仓库）之间的稳定数据接口。  
只要协议版本不变，下游无需感知上游采集实现的任何变化。

---

## 目录结构约定

每次采集运行产出一个独立子目录，命名建议为 `data/run_<timestamp>/`：

```
data/run_001/
├── run_metadata.jsonl       # 元信息（每次运行 1 条 start 记录 + 1 条 end 记录）
├── window_metrics.jsonl     # 时间窗级聚合指标（主要数据文件）
├── events.jsonl             # 逐事件记录（仅在 --emit-events 时生成）
└── [analysis results]       # analysis/ 脚本写入同目录或 results/ 子目录
```

---

## 1. `run_metadata.jsonl`

每次运行写入两条记录：**start record**（运行开始时）和 **end record**（运行结束时）。

### Start Record 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `"2.0"` | 协议版本，用于兼容性检查 |
| `run_id` | string (UUID v4) | 本次运行唯一标识，与 `window_metrics.jsonl` 关联 |
| `start_ts_iso` | string (ISO 8601) | 采集开始时间 |
| `end_ts_iso` | null | 运行中为 null，结束后由 end record 携带 |
| `target_pid` | integer | 目标 PID；0 = 采集所有进程 |
| `target_tid` | integer | 目标 TID；0 = 不按线程过滤 |
| `target_comm` | string | 目标进程名（通过 `--comm` 指定时填入） |
| `aggregation_scope` | string | `per_pid` 或 `per_tid` |
| `window_sec` | number | 时间窗大小（秒） |
| `sample_rate` | integer | 兼容旧 CLI 名称；实际语义是 perf sample_period（每 N 次事件触发一次） |
| `collection_backend` | string | 当前采集后端，例如 `bcc`、`perf_event_open`、`hybrid_perf_event_open_bcc`、`libbpf` |
| `enabled_probes` | object | `{ llc: bool, dtlb: bool, itlb: bool, fault: bool, lbr: bool }` |
| `observations` | array<object> | 本次运行启用的观测项定义；用于表达 PMU grouping、multiplex 处理方式以及未来的 LBR 接入点 |
| `host_info` | object | `{ hostname, kernel_version, cpu_model, num_cpus }` |

### End Record 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `"2.0"` | |
| `run_id` | string | 与 start record 相同 |
| `end_ts_iso` | string (ISO 8601) | 采集结束时间 |
| `_record_type` | `"run_end"` | 标识这是 end record |

### 示例

```jsonl
{"schema_version":"2.0","run_id":"a1b2c3d4-...","start_ts_iso":"2026-04-14T10:00:00+00:00","end_ts_iso":null,"target_pid":1234,"target_tid":0,"target_comm":"nginx","aggregation_scope":"per_tid","window_sec":1.0,"sample_rate":100,"collection_backend":"bcc","enabled_probes":{"llc":true,"dtlb":true,"itlb":true,"fault":true,"lbr":true},"observations":[{"observation_id":"pmu_llc_load_miss","kind":"pmu_sampling","backend":"bcc_perf_event_raw","metrics":["llc_load_misses","samples"],"scope":"per_tid","sample_period":100,"perf_event":{"type":"hw_cache","config":"ll.read.miss"},"group":{"id":"pmu_cache"},"multiplex":{"mode":"opaque_backend","scaling_fields":[]},"notes":"Accumulated via ctx->sample_period per handler invocation."}],"host_info":{"hostname":"dev01","kernel_version":"6.8.0","cpu_model":"Intel Core i7-12700","num_cpus":20}}
{"schema_version":"2.0","run_id":"a1b2c3d4-...","end_ts_iso":"2026-04-14T10:01:05+00:00","_record_type":"run_end"}
```

### `observations[]` 对象

`observations` 为可选扩展字段，建议在需要表达 PMU 分组、multiplex 缩放质量，或未来接入 LBR 时使用。

| 字段 | 类型 | 说明 |
|------|------|------|
| `observation_id` | string | 观测项唯一标识，例如 `pmu_llc_load_miss` |
| `kind` | string | 观测源类别，例如 `pmu_sampling`、`pmu_counting`、`trace_hook`、`lbr_sampling` |
| `backend` | string | 具体后端，例如 `bcc_perf_event`、`perf_event_open` |
| `metrics` | array<string> | 该观测项最终贡献到哪些输出字段 |
| `scope` | string | 采集粒度，例如 `per_pid`、`per_tid` |
| `sample_period` | integer | 采样周期；仅采样型 observation 需要 |
| `perf_event` | object | perf 事件描述，如 `{ type, config }` |
| `group` | object | PMU 分组信息，如 `{ id, role }`；LBR 建议独立成单独组 |
| `multiplex` | object | multiplex 策略，如 `{ mode, scaling_fields }` |
| `notes` | string | 兼容性或准确性备注 |

设计建议：

- 优先增加 `observation` 对象，而不是为 LBR 单独再开一套顶层 schema。LBR 应作为 `kind = "lbr_sampling"` 的 observation 接入。
- 只有语义上必须同步读出的 PMU 才放进同一 `group.id`；像当前 dTLB fallback 与 LLC 近似事件，不应强行声明为同组。
- 如果后端支持 `perf_event_open` 的 fd 读取，推荐使用 `PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING`，并把 `multiplex.mode` 设为 `time_enabled_running`。
- 如果后端像当前 BCC 原型一样不暴露 `time_enabled/time_running`，则显式标注 `multiplex.mode = "opaque_backend"`，不要伪装成“已正确缩放”。

---

## 2. `window_metrics.jsonl`

每个时间窗每个 PID 或 TID 写入一条记录。一次采集 N 个窗口、每窗口 M 个活跃实体，则共有 N×M 条记录。

**所有计数字段均为差分值**（本窗口内的增量，而非累积值）。

说明：

- `cycles`、`instructions`、LLC、dTLB、iTLB 等 PMU 字段的具体语义由 `observations[]` 中对应 observation 的 backend 决定。
- 当 PMU backend 为 `bcc_perf_event_raw` 时，字段为近似事件计数，通过每次 handler 触发时累加 `ctx->sample_period` 得到。
- 当 PMU backend 为 `perf_event_open` 时，字段来自用户态定期读取 perf fd 的累计计数，并按 `time_enabled/time_running` 做缩放。
- `samples` 仍表示 eBPF handler 的触发次数，也就是采样命中数，而不是放大后的近似事件总数。
- `minor_faults` / `major_faults` 来自 trace hook，表示实际 fault 次数，不经过 `sample_period` 放大。

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `"2.0"` | |
| `run_id` | string | 对应 `run_metadata.jsonl` 中的 `run_id` |
| `window_id` | integer ≥ 0 | 单调递增窗口序号，第一个窗口为 0 |
| `start_ns` | integer | 窗口起始时间戳（CLOCK_MONOTONIC，纳秒） |
| `end_ns` | integer | 窗口结束时间戳（CLOCK_MONOTONIC，纳秒） |
| `entity_scope` | string | `pid` 或 `tid` |
| `pid` | integer | 进程 ID |
| `tid` | integer | 线程 ID；仅 `entity_scope = "tid"` 时出现 |
| `comm` | string | 进程名 |
| `llc_loads` | integer ≥ 0 | 本窗口 LLC load access 计数；具体来源由 observation backend 决定 |
| `llc_load_misses` | integer ≥ 0 | 本窗口 LLC load miss 计数；具体来源由 observation backend 决定 |
| `llc_stores` | integer ≥ 0 | 本窗口 LLC store access 计数；具体来源由 observation backend 决定 |
| `llc_store_misses` | integer ≥ 0 | 本窗口 LLC store miss 计数；具体来源由 observation backend 决定 |
| `dtlb_loads` | integer ≥ 0 | 本窗口 dTLB load access 计数；具体来源由 observation backend 决定 |
| `dtlb_load_misses` | integer ≥ 0 | 本窗口 dTLB load miss 计数；具体来源由 observation backend 决定 |
| `dtlb_stores` | integer ≥ 0 | 本窗口 dTLB store access 计数；具体来源由 observation backend 决定 |
| `dtlb_store_misses` | integer ≥ 0 | 本窗口 dTLB store miss 计数；具体来源由 observation backend 决定 |
| `dtlb_misses` | integer ≥ 0 | 本窗口 dTLB load/store miss 总计数；具体来源由 observation backend 决定 |
| `itlb_load_misses` | integer ≥ 0 | 本窗口 iTLB load miss 计数；具体来源由 observation backend 决定 |
| `cycles` | integer ≥ 0 | 本窗口 CPU cycles 计数；具体来源由 observation backend 决定 |
| `instructions` | integer ≥ 0 | 本窗口 retired instructions 计数；具体来源由 observation backend 决定 |
| `minor_faults` | integer ≥ 0 | 本窗口 minor page fault 次数 |
| `major_faults` | integer ≥ 0 | 本窗口 major page fault 次数 |
| `lbr_samples` | integer ≥ 0 | 本窗口 LBR 分支栈样本数 |
| `lbr_entries` | integer ≥ 0 | 本窗口累计导出的 LBR 分支条目数 |
| `samples` | integer ≥ 0 | 本窗口 eBPF handler 触发总次数 |

### 示例

```jsonl
{"schema_version":"2.0","run_id":"a1b2c3d4-...","window_id":0,"start_ns":1000000000,"end_ns":1001000000000,"entity_scope":"tid","pid":1234,"tid":1239,"comm":"nginx","llc_loads":981200,"llc_load_misses":452100,"llc_stores":208800,"llc_store_misses":10200,"dtlb_loads":33000,"dtlb_load_misses":8900,"dtlb_stores":7400,"dtlb_store_misses":1100,"dtlb_misses":10000,"itlb_load_misses":700,"cycles":6103200,"instructions":5862600,"minor_faults":12,"major_faults":0,"lbr_samples":45,"lbr_entries":318,"samples":4723}
```

---

## 3. `events.jsonl`（可选）

仅在采集时指定 `--emit-events` 时生成，记录每个被采样事件的详细信息。  
适用于函数级归因（P2 阶段）。

说明：`events.jsonl` 仍是一条采样命中对应一条记录，不会按 `sample_period` 放大。

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `"2.0"` | |
| `run_id` | string | |
| `ts_ns` | integer | 事件时间戳 |
| `pid` | integer | |
| `tid` | integer | 线程 ID |
| `comm` | string | |
| `event_type` | integer | 1=llc_load_miss 2=llc_store_miss 3=dtlb_miss 4=minor_fault 5=major_fault 6=lbr_sample |
| `addr` | integer | 相关内存地址（page fault 时为出错地址，perf_event 时为 IP） |
| `ip` | integer | 采样时的指令指针 |
| `lbr` | array<object> | 可选，仅 LBR 事件出现；每项含 `{ from_ip, to_ip, flags }`，当前最多导出 8 条 |

---

## 4. `hotspot_summary.jsonl`（下游 / 分析产物）

由 `analysis/hotspot.py` 或 `analysis/attribution.py` 生成，描述热点实体。  
Schema 见 `export/schema/hotspot_summary.schema.json`。

---

## 与 ebpf-mem-analyzer 的对接方式

使用 `export/to_baseline.py` 将 `window_metrics.jsonl` 转换为基线仓库期望的 CSV 格式：

```bash
python export/to_baseline.py \
    --input  data/run_001/ \
    --output /path/to/ebpf-mem-analyzer/data/new_input/
```

列映射在 `export/to_baseline.py` 的 `COLUMN_MAP` 字典中集中维护，下游接口变更时只需更新该字典。

---

## 版本兼容性

- 消费方应检查 `schema_version` 字段，遇到未知版本时应拒绝处理而非静默解析。
- 新增可选字段不递增版本号；删除或重命名字段、修改字段语义时递增主版本号。
