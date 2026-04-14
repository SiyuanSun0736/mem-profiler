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
    enable_fault: bool,
) -> list[dict]:
    observations: list[ObservationSpec] = []

    if enable_llc:
        observations.append(
            ObservationSpec(
                observation_id="pmu_llc_load_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event",
                metrics=["llc_load_misses", "samples"],
                sample_period=sample_rate,
                perf_type="hardware",
                perf_config="cache_misses",
                group_id="pmu_cache",
                group_role="leader",
                multiplex_mode="opaque_backend",
                notes=(
                    "BCC attach_perf_event does not expose group leader reads or "
                    "time_enabled/time_running, so multiplex scaling quality is not "
                    "exported per window."
                ),
            )
        )

    if enable_dtlb:
        observations.append(
            ObservationSpec(
                observation_id="pmu_dtlb_miss",
                kind="pmu_sampling",
                backend="bcc_perf_event",
                metrics=["dtlb_misses"],
                sample_period=sample_rate * 10,
                perf_type="hardware",
                perf_config="cache_misses_fallback",
                group_id="pmu_tlb",
                group_role="leader",
                multiplex_mode="opaque_backend",
                notes=(
                    "Current prototype uses CACHE_MISSES fallback rather than a raw "
                    "dTLB event; keep it in a separate group from LLC observations to "
                    "avoid implying lock-step PMU semantics."
                ),
            )
        )

    if enable_fault:
        observations.append(
            ObservationSpec(
                observation_id="trace_page_fault",
                kind="trace_hook",
                backend="bcc_kprobe",
                metrics=["minor_faults", "major_faults"],
                multiplex_mode="not_applicable",
                notes="Kernel trace hook; no PMU multiplex or group leader semantics.",
            )
        )

    return [item.to_metadata() for item in observations]