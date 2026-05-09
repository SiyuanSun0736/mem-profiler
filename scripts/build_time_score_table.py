#!/usr/bin/env python3
"""
build_time_score_table.py — 构建带 repeat timing 优先逻辑的时间评分基准表
==========================================================================

原理
----
默认先尝试读取每个 output_dir 下的 repeat_timing.json。若 baseline 与当前 variant
都存在有效 repeat timing，则优先使用其中的 median_wall_time_sec 构造 score_time。

若缺少 repeat timing，则回退到现有 60 s BCC 采集窗口中的 proxy：

  time_per_iter(k)  =  wall_time_sec / active_pid_count
                     ≈  单次迭代平均挂钟时间（秒）

其中 active_pid_count 与 cycles_per_iter 的分母相同，保证两者量纲一致。

时间评分（参考 O0 基准）：
  score_time(k) = log( time_per_iter_O0 / time_per_iter_k )
                > 0 → 比 O0 快（优化有效）
                ≈ 0 → 与 O0 相当
                < 0 → 比 O0 慢（退化）

当程序缺少 O0 或 active_pid_count ≤ 0 时该程序从表中排除。

输出
----
    train_set/time_scores.parquet   — 含 proxy / repeat / preferred 三套时间列的评分表

用法
----
  python scripts/build_time_score_table.py
  python scripts/build_time_score_table.py --input train_set/run_features.parquet
  python scripts/build_time_score_table.py --baseline O0 --output train_set/time_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASELINE_VARIANT = "O0"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "llvm_test_suite"
DEFAULT_MIN_ACTIVE_PIDS = 5
DEFAULT_MIN_ACTIVE_WINDOW_RATIO = 0.10
DEFAULT_REPEAT_TIMING_NAME = "repeat_timing.json"
DEFAULT_REPEAT_SIDECAR_ROOT = DEFAULT_DATA_ROOT / "repeat_timing_sidecars"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=str(REPO_ROOT / "train_set" / "run_features.parquet"),
        help="run_features.parquet 路径",
    )
    p.add_argument(
        "--baseline",
        default=BASELINE_VARIANT,
        help="用作基线的变体名称（默认 O0）",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "train_set" / "time_scores.parquet"),
        help="输出 parquet 路径",
    )
    p.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "train_set" / "time_score_filter_summary.json"),
        help="输出严格时间真值过滤摘要 JSON 路径",
    )
    p.add_argument(
        "--min-active-pids",
        type=int,
        default=DEFAULT_MIN_ACTIVE_PIDS,
        help="构造严格时间真值时要求的最小 active_pid_count（默认 5）",
    )
    p.add_argument(
        "--min-active-window-ratio",
        type=float,
        default=DEFAULT_MIN_ACTIVE_WINDOW_RATIO,
        help="构造严格时间真值时要求的最小 active_window_count / window_count（默认 0.10）",
    )
    p.add_argument(
        "--repeat-timing-name",
        default=DEFAULT_REPEAT_TIMING_NAME,
        help="若 output_dir 下存在该 sidecar，则优先使用 repeat timing 中位数 wall time",
    )
    p.add_argument(
        "--repeat-sidecar-root",
        default=str(DEFAULT_REPEAT_SIDECAR_ROOT),
        help="collect_repeat_timing.py 在 output_dir 不可写时会将 sidecar 镜像写到该根目录，本脚本也会在此处查找",
    )
    return p.parse_args()


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _score_time_from_pair(t_base: float, t_variant: float) -> float:
    if not (math.isfinite(t_variant) and t_variant > 0 and math.isfinite(t_base) and t_base > 0):
        return float("nan")
    return math.log(t_base / t_variant)


def _strict_invalid_reasons(
    row: pd.Series,
    min_active_pids: int,
    min_active_window_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if int(row.get("active_pid_count", 0) or 0) < min_active_pids:
        reasons.append("low_active_pid_count")
    if float(row.get("active_window_ratio", 0.0) or 0.0) < min_active_window_ratio:
        reasons.append("low_active_window_ratio")
    return reasons


def _empty_repeat_timing_info() -> dict[str, object]:
    return {
        "repeat_timing_path": "",
        "repeat_timing_available": False,
        "repeat_timing_valid": False,
        "repeat_success_count": 0,
        "repeat_count": 0,
        "repeat_median_wall_time_sec": float("nan"),
        "repeat_mad_wall_time_sec": float("nan"),
        "time_per_iter_repeat": float("nan"),
    }


def _relative_to_repo(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _remap_output_dir_to_repo(path: pathlib.Path) -> pathlib.Path | None:
    anchor = ("data", "llvm_test_suite")
    parts = path.parts
    for index in range(len(parts) - len(anchor) + 1):
        if parts[index:index + len(anchor)] == anchor:
            return (DEFAULT_DATA_ROOT / pathlib.Path(*parts[index + len(anchor):])).resolve()
    return None


def _sidecar_subpath(output_dir: pathlib.Path) -> pathlib.Path:
    resolved_output_dir = output_dir.resolve()

    try:
        return resolved_output_dir.relative_to(DEFAULT_DATA_ROOT.resolve())
    except ValueError:
        pass

    try:
        return resolved_output_dir.relative_to(REPO_ROOT)
    except ValueError:
        pass

    remapped = _remap_output_dir_to_repo(resolved_output_dir)
    if remapped is not None:
        try:
            return remapped.relative_to(DEFAULT_DATA_ROOT.resolve())
        except ValueError:
            pass

    anchor = ("data", "llvm_test_suite")
    parts = resolved_output_dir.parts
    for index in range(len(parts) - len(anchor) + 1):
        if parts[index:index + len(anchor)] == anchor:
            return pathlib.Path(*parts[index + len(anchor):])

    sanitized_parts = [part for part in parts if part and part != resolved_output_dir.anchor]
    return pathlib.Path("external", *sanitized_parts)


def _candidate_repeat_paths(
    output_dir: object,
    repeat_timing_name: str,
    repeat_sidecar_root: pathlib.Path,
) -> list[pathlib.Path]:
    if output_dir is None:
        return []

    raw_output_dir = str(output_dir).strip()
    if not raw_output_dir:
        return []

    path = pathlib.Path(raw_output_dir)
    candidate_dirs: list[pathlib.Path] = []
    if path.is_absolute():
        candidate_dirs.append(path)
    else:
        candidate_dirs.append((REPO_ROOT / path).resolve())

    remapped = _remap_output_dir_to_repo(path)
    if remapped is not None:
        candidate_dirs.append(remapped)

    candidates: list[pathlib.Path] = []
    seen: set[str] = set()
    for candidate_dir in candidate_dirs:
        direct_path = candidate_dir / repeat_timing_name
        direct_key = str(direct_path)
        if direct_key not in seen:
            candidates.append(direct_path)
            seen.add(direct_key)

        mirror_path = repeat_sidecar_root / _sidecar_subpath(candidate_dir) / repeat_timing_name
        mirror_key = str(mirror_path)
        if mirror_key not in seen:
            candidates.append(mirror_path)
            seen.add(mirror_key)
    return candidates


def _load_repeat_timing_info(
    output_dir: object,
    repeat_timing_name: str,
    repeat_sidecar_root: pathlib.Path,
) -> dict[str, object]:
    info = _empty_repeat_timing_info()
    repeat_path = next(
        (candidate for candidate in _candidate_repeat_paths(output_dir, repeat_timing_name, repeat_sidecar_root) if candidate.exists()),
        None,
    )
    if repeat_path is None:
        return info

    info["repeat_timing_available"] = True
    info["repeat_timing_path"] = _relative_to_repo(repeat_path)

    try:
        payload = json.loads(repeat_path.read_text(encoding="utf-8"))
    except Exception:
        return info

    success_count = int(payload.get("success_count", 0) or 0)
    repeat_count = int(payload.get("repeat_count", 0) or 0)
    median_wall_time_sec = _safe_float(payload.get("median_wall_time_sec"))
    mad_wall_time_sec = _safe_float(payload.get("mad_wall_time_sec"))
    valid = bool(payload.get("valid", False))
    if not valid and repeat_count > 0 and success_count >= repeat_count:
        valid = True

    info.update({
        "repeat_timing_valid": bool(valid and math.isfinite(median_wall_time_sec) and median_wall_time_sec > 0.0),
        "repeat_success_count": success_count,
        "repeat_count": repeat_count,
        "repeat_median_wall_time_sec": median_wall_time_sec,
        "repeat_mad_wall_time_sec": mad_wall_time_sec,
        "time_per_iter_repeat": median_wall_time_sec if valid and math.isfinite(median_wall_time_sec) and median_wall_time_sec > 0.0 else float("nan"),
    })
    return info


def build_time_scores(
    rf: pd.DataFrame,
    baseline: str,
    min_active_pids: int,
    min_active_window_ratio: float,
    repeat_timing_name: str,
    repeat_sidecar_root: pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"program", "variant", "wall_time_sec", "active_pid_count"}
    missing = required - set(rf.columns)
    if missing:
        raise ValueError(f"run_features 缺少必要列: {missing}")

    rf = rf.copy()
    ratio_cols_present = {"window_count", "active_window_count"}.issubset(rf.columns)
    if ratio_cols_present:
        rf["active_window_ratio"] = rf.apply(
            lambda r: _safe_div(r["active_window_count"], r["window_count"]),
            axis=1,
        ).fillna(0.0)
    else:
        # 兼容旧版 run_features：若缺少窗口统计，则退化为不按该项过滤。
        rf["active_window_ratio"] = 1.0

    rf["time_per_iter"] = rf.apply(
        lambda r: _safe_div(r["wall_time_sec"], r["active_pid_count"]),
        axis=1,
    )
    rf["time_per_iter_proxy"] = rf["time_per_iter"]

    if "output_dir" in rf.columns:
        output_dir_values = rf["output_dir"].fillna("").astype(str)
        repeat_infos = {
            output_dir: _load_repeat_timing_info(output_dir, repeat_timing_name, repeat_sidecar_root)
            for output_dir in output_dir_values.unique()
        }
        repeat_df = pd.DataFrame(
            [repeat_infos[output_dir] for output_dir in output_dir_values],
            index=rf.index,
        )
    else:
        repeat_df = pd.DataFrame(
            [_empty_repeat_timing_info() for _ in range(len(rf))],
            index=rf.index,
        )
    rf = pd.concat([rf, repeat_df], axis=1)

    loose_base_rows = rf[rf["variant"] == baseline][["program", "time_per_iter_proxy"]].copy()
    loose_base_rows = loose_base_rows.dropna(subset=["time_per_iter_proxy"])
    loose_base_rows = loose_base_rows.rename(columns={"time_per_iter": "time_per_iter_base_loose"})
    loose_base_rows = loose_base_rows.rename(columns={"time_per_iter_proxy": "time_per_iter_base_loose"})
    loose_base_map = loose_base_rows.groupby("program")["time_per_iter_base_loose"].mean()
    rf["has_loose_baseline"] = rf["program"].isin(set(loose_base_map.index))
    rf["time_per_iter_base_loose"] = rf["program"].map(loose_base_map)
    rf["score_time_loose"] = rf.apply(
        lambda row: _score_time_from_pair(row["time_per_iter_base_loose"], row["time_per_iter_proxy"]),
        axis=1,
    )

    rf["time_score_invalid_reasons"] = rf.apply(
        lambda row: _strict_invalid_reasons(
            row,
            min_active_pids=min_active_pids,
            min_active_window_ratio=min_active_window_ratio,
        ),
        axis=1,
    )
    rf["time_score_input_ok"] = rf["time_score_invalid_reasons"].apply(lambda reasons: len(reasons) == 0)

    strict_base_rows = rf[
        (rf["variant"] == baseline) & rf["time_score_input_ok"]
    ][["program", "time_per_iter_proxy"]].copy()
    strict_base_rows = strict_base_rows.dropna(subset=["time_per_iter_proxy"])
    strict_base_rows = strict_base_rows.rename(columns={"time_per_iter_proxy": "time_per_iter_base"})
    strict_base_map = strict_base_rows.groupby("program")["time_per_iter_base"].mean()

    rf["has_strict_baseline"] = rf["program"].isin(set(strict_base_map.index))
    rf["time_per_iter_base"] = rf["program"].map(strict_base_map)
    rf["time_per_iter_base_proxy"] = rf["time_per_iter_base"]
    rf["score_time_proxy"] = rf.apply(
        lambda row: _score_time_from_pair(row["time_per_iter_base"], row["time_per_iter"])
        if row["time_score_input_ok"] and row["has_strict_baseline"]
        else float("nan"),
        axis=1,
    )

    repeat_base_rows = rf[
        (rf["variant"] == baseline) & rf["repeat_timing_valid"]
    ][["program", "time_per_iter_repeat"]].copy()
    repeat_base_rows = repeat_base_rows.dropna(subset=["time_per_iter_repeat"])
    repeat_base_rows = repeat_base_rows.rename(columns={"time_per_iter_repeat": "time_per_iter_base_repeat"})
    repeat_base_map = repeat_base_rows.groupby("program")["time_per_iter_base_repeat"].mean()

    rf["has_repeat_baseline"] = rf["program"].isin(set(repeat_base_map.index))
    rf["time_per_iter_base_repeat"] = rf["program"].map(repeat_base_map)
    rf["score_time_repeat"] = rf.apply(
        lambda row: _score_time_from_pair(row["time_per_iter_base_repeat"], row["time_per_iter_repeat"])
        if row["repeat_timing_valid"] and row["has_repeat_baseline"]
        else float("nan"),
        axis=1,
    )

    rf["time_per_iter_preferred"] = rf["time_per_iter_proxy"]
    rf["time_per_iter_base_preferred"] = rf["time_per_iter_base_proxy"]
    rf["score_time"] = rf["score_time_proxy"]
    rf["score_time_source"] = ""
    rf.loc[rf["score_time_proxy"].notna(), "score_time_source"] = "proxy_strict"
    repeat_mask = rf["score_time_repeat"].notna()
    rf.loc[repeat_mask, "time_per_iter_preferred"] = rf.loc[repeat_mask, "time_per_iter_repeat"]
    rf.loc[repeat_mask, "time_per_iter_base_preferred"] = rf.loc[repeat_mask, "time_per_iter_base_repeat"]
    rf.loc[repeat_mask, "score_time"] = rf.loc[repeat_mask, "score_time_repeat"]
    rf.loc[repeat_mask, "score_time_source"] = "repeat_timing"
    rf["time_score_strict_ok"] = rf["score_time"].notna()
    rf["time_score_invalid_reasons"] = rf["time_score_invalid_reasons"].apply(
        lambda reasons: "|".join(reasons)
    )

    reason_counts = {
        "low_active_pid_count": int(
            (rf["active_pid_count"].fillna(0).astype(int) < min_active_pids).sum()
        ),
        "low_active_window_ratio": int((rf["active_window_ratio"] < min_active_window_ratio).sum()),
        "missing_strict_baseline": int(
            (rf["time_score_input_ok"] & ~rf["has_strict_baseline"]).sum()
        ),
    }

    summary: dict[str, object] = {
        "baseline": baseline,
        "repeat_timing_name": repeat_timing_name,
        "repeat_sidecar_root": _relative_to_repo(repeat_sidecar_root),
        "min_active_pids": int(min_active_pids),
        "min_active_window_ratio": float(min_active_window_ratio),
        "active_window_ratio_available": bool(ratio_cols_present),
        "n_seen": int(len(rf)),
        "n_programs_seen": int(rf["program"].nunique()),
        "n_input_ok": int(rf["time_score_input_ok"].sum()),
        "n_input_filtered": int((~rf["time_score_input_ok"]).sum()),
        "n_programs_with_loose_baseline": int(len(loose_base_map)),
        "n_programs_with_strict_baseline": int(len(strict_base_map)),
        "n_programs_with_repeat_baseline": int(len(repeat_base_map)),
        "n_valid_loose": int(rf["score_time_loose"].notna().sum()),
        "n_valid_strict_proxy": int(rf["score_time_proxy"].notna().sum()),
        "n_valid_repeat": int(rf["score_time_repeat"].notna().sum()),
        "n_valid_strict": int(rf["score_time"].notna().sum()),
        "reasons": reason_counts,
        "repeat_timing": {
            "n_available_rows": int(rf["repeat_timing_available"].sum()),
            "n_valid_rows": int(rf["repeat_timing_valid"].sum()),
            "n_preferred_repeat": int((rf["score_time_source"] == "repeat_timing").sum()),
            "n_preferred_proxy": int((rf["score_time_source"] == "proxy_strict").sum()),
            "n_rescued_by_repeat_timing": int(
                (rf["score_time_repeat"].notna() & rf["score_time_proxy"].isna()).sum()
            ),
        },
        "by_variant": {},
    }

    example_cols = [
        "program",
        "variant",
        "active_pid_count",
        "active_window_ratio",
        "wall_time_sec",
        "output_dir",
        "time_score_invalid_reasons",
    ]
    available_example_cols = [c for c in example_cols if c in rf.columns]
    filtered_examples = rf[~rf["time_score_input_ok"]][available_example_cols].head(10).copy()
    if "active_window_ratio" in filtered_examples.columns:
        filtered_examples["active_window_ratio"] = filtered_examples["active_window_ratio"].round(6)
    if "wall_time_sec" in filtered_examples.columns:
        filtered_examples["wall_time_sec"] = filtered_examples["wall_time_sec"].round(6)
    summary["examples"] = filtered_examples.to_dict(orient="records")

    for variant, sub in rf.groupby("variant", sort=True):
        summary["by_variant"][str(variant)] = {
            "seen": int(len(sub)),
            "input_ok": int(sub["time_score_input_ok"].sum()),
            "input_filtered": int((~sub["time_score_input_ok"]).sum()),
            "valid_loose": int(sub["score_time_loose"].notna().sum()),
            "valid_strict_proxy": int(sub["score_time_proxy"].notna().sum()),
            "valid_repeat": int(sub["score_time_repeat"].notna().sum()),
            "valid_strict": int(sub["score_time"].notna().sum()),
            "preferred_repeat": int((sub["score_time_source"] == "repeat_timing").sum()),
            "preferred_proxy": int((sub["score_time_source"] == "proxy_strict").sum()),
        }

    keep_cols = [
        "program",
        "variant",
        "output_dir",
        "wall_time_sec",
        "active_pid_count",
        "active_window_ratio",
        "time_per_iter",
        "time_per_iter_proxy",
        "time_per_iter_repeat",
        "time_per_iter_preferred",
        "time_per_iter_base_loose",
        "score_time_loose",
        "time_score_input_ok",
        "has_loose_baseline",
        "has_strict_baseline",
        "has_repeat_baseline",
        "time_score_invalid_reasons",
        "time_per_iter_base",
        "time_per_iter_base_proxy",
        "time_per_iter_base_repeat",
        "time_per_iter_base_preferred",
        "repeat_timing_path",
        "repeat_timing_available",
        "repeat_timing_valid",
        "repeat_success_count",
        "repeat_count",
        "repeat_median_wall_time_sec",
        "repeat_mad_wall_time_sec",
        "score_time_proxy",
        "score_time_repeat",
        "score_time_source",
        "score_time",
        "time_score_strict_ok",
    ]
    out = rf[[c for c in keep_cols if c in rf.columns]].copy()
    out = out.sort_values(["program", "variant"]).reset_index(drop=True)
    return out, summary


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    summary_path = pathlib.Path(args.summary_json)

    if not input_path.exists():
        print(f"[error] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] 读取 {input_path.name} ...", flush=True)
    rf = pd.read_parquet(input_path)
    print(f"       {len(rf)} 行，{rf['program'].nunique()} 个程序，"
          f"{rf['variant'].nunique()} 个变体", flush=True)

    out, summary = build_time_scores(
        rf,
        baseline=args.baseline,
        min_active_pids=args.min_active_pids,
        min_active_window_ratio=args.min_active_window_ratio,
        repeat_timing_name=args.repeat_timing_name,
        repeat_sidecar_root=pathlib.Path(args.repeat_sidecar_root).resolve(),
    )

    n_valid_loose = int(out["score_time_loose"].notna().sum())
    n_valid = int(out["score_time"].notna().sum())
    n_valid_proxy = int(out["score_time_proxy"].notna().sum()) if "score_time_proxy" in out.columns else n_valid
    n_preferred_repeat = int((out["score_time_source"] == "repeat_timing").sum()) if "score_time_source" in out.columns else 0
    n_total = len(out)
    n_prog  = out["program"].nunique()
    print(
        f"[info] 时间评分: loose={n_valid_loose}/{n_total}，strict={n_valid}/{n_total} 行有效"
        f"（{n_prog} 个程序）",
        flush=True,
    )
    print(
        f"       strict 门槛: active_pid_count >= {args.min_active_pids}, "
        f"active_window_ratio >= {args.min_active_window_ratio:.3f}",
        flush=True,
    )
    print(
        f"       strict 来源: repeat={n_preferred_repeat}，proxy={n_valid - n_preferred_repeat}"
        f"（proxy-only strict={n_valid_proxy}，repeat rescue={summary['repeat_timing']['n_rescued_by_repeat_timing']}）",
        flush=True,
    )

    if n_valid > 0:
        valid = out.dropna(subset=["score_time"])
        print(f"       score_time 均值={valid['score_time'].mean():.4f}"
              f"  std={valid['score_time'].std():.4f}"
              f"  [min={valid['score_time'].min():.3f}, max={valid['score_time'].max():.3f}]",
              flush=True)

    print(
        f"[info] 严格过滤摘要: 输入过滤 {summary['n_input_filtered']} 行，"
        f"缺失严格基线 {summary['reasons']['missing_strict_baseline']} 行，"
        f"有效 repeat timing {summary['repeat_timing']['n_valid_rows']} 行",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[done] 已写入 {output_path}", flush=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] 已写入 {summary_path}", flush=True)


if __name__ == "__main__":
    main()
