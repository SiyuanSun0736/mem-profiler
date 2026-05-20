#!/usr/bin/env python3

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable


RuleFn = Callable[[str], bool]


def _prefix_rule(*prefixes: str) -> RuleFn:
    return lambda col: any(col.startswith(prefix) for prefix in prefixes)


def _exact_or_prefix_rule(exact: set[str], prefixes: tuple[str, ...] = ()) -> RuleFn:
    return lambda col: col in exact or any(col.startswith(prefix) for prefix in prefixes)


CATEGORY_SPECS: tuple[tuple[str, str, RuleFn], ...] = (
    (
        "core",
        "Execution efficiency and IPC distribution features.",
        _exact_or_prefix_rule({"ipc", "cpi", "samples_per_ms"}, ("win_ipc_",)),
    ),
    (
        "cache",
        "LLC pressure, cache miss rates, and LLC phase features.",
        _exact_or_prefix_rule(
            {
                "llc_load_miss_rate",
                "llc_store_miss_rate",
                "llc_mpki",
                "llc_store_mpki",
                "warmup_llc_mpki",
                "steady_llc_mpki",
                "phase_llc_ratio",
            },
            ("win_llc_",),
        ),
    ),
    (
        "tlb",
        "dTLB/iTLB pressure and window-level TLB dynamics.",
        _exact_or_prefix_rule(
            {"dtlb_miss_rate", "dtlb_mpki", "itlb_mpki"},
            ("win_dtlb_", "win_itlb_"),
        ),
    ),
    (
        "fault",
        "Page fault rates, fault subtype ratios, and fault phase features.",
        _exact_or_prefix_rule(
            {
                "fault_per_ki",
                "fault_per_ms",
                "anon_fault_ratio",
                "file_fault_ratio",
                "write_fault_ratio",
                "instruction_fault_ratio",
                "phase_fault_ratio",
            },
            ("win_fault_",),
        ),
    ),
    (
        "mm",
        "MM syscall density and allocation throughput features.",
        _prefix_rule("mmap_", "munmap_", "brk_", "mm_syscall_"),
    ),
    (
        "phase",
        "Warmup versus steady-state IPC phase features.",
        _exact_or_prefix_rule({"warmup_ipc", "steady_ipc", "phase_ipc_ratio"}),
    ),
)


def build_category_feature_map(non_time_cols: list[str]) -> OrderedDict[str, list[str]]:
    feature_order = list(non_time_cols)
    assigned: set[str] = set()
    feature_map: OrderedDict[str, list[str]] = OrderedDict()

    for name, _, rule in CATEGORY_SPECS:
        cols = [col for col in feature_order if rule(col)]
        if not cols:
            raise ValueError(f"category '{name}' matched no features")
        overlap = sorted(set(cols).intersection(assigned))
        if overlap:
            raise ValueError(f"category '{name}' overlaps on: {', '.join(overlap)}")
        feature_map[name] = cols
        assigned.update(cols)

    unassigned = [col for col in feature_order if col not in assigned]
    if unassigned:
        raise ValueError("unassigned features: " + ", ".join(unassigned))

    return feature_map


def build_category_index_map(non_time_cols: list[str]) -> OrderedDict[str, list[int]]:
    feature_map = build_category_feature_map(non_time_cols)
    index_map: OrderedDict[str, list[int]] = OrderedDict()
    for name, cols in feature_map.items():
        index_map[name] = [non_time_cols.index(col) for col in cols]
    return index_map


def schema_metadata(
    non_time_cols: list[str],
    include_summary_token: bool = True,
) -> dict[str, object]:
    feature_map = build_category_feature_map(non_time_cols)
    descriptions = {name: desc for name, desc, _ in CATEGORY_SPECS}

    tokens: list[dict[str, object]] = []
    if include_summary_token:
        tokens.append(
            {
                "token": "summary",
                "description": "Global run-level summary projection over all non-time features.",
                "n_features": len(non_time_cols),
                "features": list(non_time_cols),
            }
        )

    for name, cols in feature_map.items():
        tokens.append(
            {
                "token": name,
                "description": descriptions[name],
                "n_features": len(cols),
                "features": cols,
            }
        )

    return {
        "include_summary_token": include_summary_token,
        "n_total_features": len(non_time_cols),
        "n_category_tokens": len(feature_map),
        "n_tokens_per_program": len(tokens),
        "category_order": list(feature_map.keys()),
        "token_order": [token["token"] for token in tokens],
        "tokens": tokens,
    }