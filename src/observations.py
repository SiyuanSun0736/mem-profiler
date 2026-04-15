"""
observations.py — 采集观测项描述

为 run_metadata.jsonl 提供稳定的 observation 元数据层，描述：
  • 一个指标来自哪类观测源（PMU / kprobe / 未来的 LBR）
  • 是否参与 PMU grouping
  • multiplex 如何处理或当前后端是否无法暴露缩放质量

当前 BCC 原型阶段仍直接输出 window_metrics.jsonl 的聚合结果；
observation 只承担“采集设计说明”的职责，便于后续迁移到
perf_event_open / libbpf 时保持数据协议稳定。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationSpec:
    observation_id: str
    kind: str
    backend: str
    metrics: list[str]
    scope: str = "per_pid"
    sample_period: int | None = None
    perf_type: str | None = None
    perf_config: str | None = None
    group_id: str | None = None
    group_role: str | None = None
    multiplex_mode: str = "not_applicable"
    scaling_fields: list[str] | None = None
    notes: str | None = None

    def to_metadata(self) -> dict:
        data = {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "backend": self.backend,
            "metrics": self.metrics,
            "scope": self.scope,
            "multiplex": {
                "mode": self.multiplex_mode,
                "scaling_fields": self.scaling_fields or [],
            },
        }
        if self.sample_period is not None:
            data["sample_period"] = self.sample_period
        if self.perf_type is not None or self.perf_config is not None:
            data["perf_event"] = {}
            if self.perf_type is not None:
                data["perf_event"]["type"] = self.perf_type
            if self.perf_config is not None:
                data["perf_event"]["config"] = self.perf_config
        if self.group_id is not None:
            data["group"] = {"id": self.group_id}
            if self.group_role is not None:
                data["group"]["role"] = self.group_role
        if self.notes is not None:
            data["notes"] = self.notes
        return data


def build_default_observations(
    sample_rate: int,
    enable_llc: bool,
    enable_dtlb: bool,
    enable_itlb: bool,
    enable_fault: bool,
    enable_lbr: bool,
    scope: str,
    llc_store_via_generic: bool = False,
) -> list[dict]:
    observations: list[ObservationSpec] = []

    if enable_llc:
        observations.extend([
            ObservationSpec(
                observation_id="pmu_llc_load",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["llc_loads", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="ll.read.access",
                group_id="pmu_cache",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_llc_load_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["llc_load_misses", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="ll.read.miss",
                group_id="pmu_cache",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_llc_store",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["llc_stores", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hardware" if llc_store_via_generic else "hw_cache",
                perf_config="cache_references" if llc_store_via_generic else "ll.write.access",
                group_id="pmu_cache",
                multiplex_mode="opaque_backend",
                notes=(
                    "Hardware does not support LLC write sampling (e.g. Intel Skylake). "
                    "Proxy: PERF_COUNT_HW_CACHE_REFERENCES (all LLC accesses). "
                    "Field llc_stores approximates total LLC reference samples."
                    if llc_store_via_generic else None
                ),
            ),
            ObservationSpec(
                observation_id="pmu_llc_store_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["llc_store_misses", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hardware" if llc_store_via_generic else "hw_cache",
                perf_config="cache_misses" if llc_store_via_generic else "ll.write.miss",
                group_id="pmu_cache",
                multiplex_mode="opaque_backend",
                notes=(
                    "Hardware does not support LLC write miss sampling (e.g. Intel Skylake). "
                    "Proxy: PERF_COUNT_HW_CACHE_MISSES (all LLC misses). "
                    "Field llc_store_misses approximates total LLC miss samples."
                    if llc_store_via_generic else (
                        "BCC raw perf_event attach does not expose time_enabled/time_running, "
                        "so per-window multiplex scaling quality is not exported."
                    )
                ),
            ),
        ])

    if enable_dtlb:
        observations.extend([
            ObservationSpec(
                observation_id="pmu_dtlb_load",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["dtlb_loads", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="dtlb.read.access",
                group_id="pmu_dtlb",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_dtlb_load_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["dtlb_load_misses", "dtlb_misses", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="dtlb.read.miss",
                group_id="pmu_dtlb",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_dtlb_store",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["dtlb_stores", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="dtlb.write.access",
                group_id="pmu_dtlb",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_dtlb_store_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["dtlb_store_misses", "dtlb_misses", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="dtlb.write.miss",
                group_id="pmu_dtlb",
                multiplex_mode="opaque_backend",
            ),
        ])

    if enable_itlb:
        observations.extend([
            ObservationSpec(
                observation_id="pmu_itlb_load",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["itlb_loads", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="itlb.read.access",
                group_id="pmu_itlb",
                multiplex_mode="opaque_backend",
            ),
            ObservationSpec(
                observation_id="pmu_itlb_load_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event_raw",
                metrics=["itlb_load_misses", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hw_cache",
                perf_config="itlb.read.miss",
                group_id="pmu_itlb",
                multiplex_mode="opaque_backend",
            ),
        ])

    if enable_fault:
        observations.append(
            ObservationSpec(
                observation_id="trace_page_fault",
                kind="trace_hook",
                backend="bcc_kprobe",
                metrics=["minor_faults", "major_faults"],
                scope=scope,
                multiplex_mode="not_applicable",
                notes="Kernel trace hook; no PMU multiplex or group leader semantics.",
            )
        )

    if enable_lbr:
        observations.append(
            ObservationSpec(
                observation_id="pmu_lbr_sample",
                kind="lbr_sampling",
                backend="bcc_perf_event_raw",
                metrics=["lbr_samples", "lbr_entries", "samples"],
                scope=scope,
                sample_period=sample_rate,
                perf_type="hardware",
                perf_config="branch_instructions+branch_stack",
                group_id="pmu_lbr",
                group_role="leader",
                multiplex_mode="opaque_backend",
                notes="LBR events are exported through events.jsonl with up to 8 branch records per sample.",
            )
        )

    return [item.to_metadata() for item in observations]