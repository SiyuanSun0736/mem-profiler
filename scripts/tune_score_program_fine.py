#!/usr/bin/env python3
"""
tune_score_program_fine.py — score_program 评分层精调（variant 分开调参）
=========================================================================

只做评分层微调，不重训模型。
脚本会复用现有 model / anchor_set / run_features_zscore / time_scores，
先缓存 query-anchor 的原始 pair 预测，再对一组细粒度参数做网格搜索，
并按 query variant 分开选择最优组合。

默认目标
--------
1. 以 strict 时间口径的 Pearson 相关性 `time_corr` 为主排序。
2. 以 `time_spearman`、`score_corr`、`time_mae` 作为 tie-break。
3. 每个 variant 单独选最优参数，不做全局共用。

输出
----
  train_set/score_tune_fine_variant_trials.csv
  train_set/score_tune_fine_variant_best.json

用法
----
  python scripts/tune_score_program_fine.py --device cpu

  python scripts/tune_score_program_fine.py \
      --variants O2,O3 \
      --tie-margin-weight-alphas 0.25,0.30,0.35,0.40
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from score_program import (  # noqa: E402
    CLASS_LABELS,
    DEFAULT_MIN_ANCHOR_QUALITY,
    DEFAULT_OUTLIER_MAD_SCALE,
    DEFAULT_OUTLIER_MIN_DELTA,
    DEFAULT_TIE_GATE_THRESHOLD,
    DEFAULT_TIE_MARGIN_WEIGHT_ALPHA,
    DEFAULT_TIE_SHRINK_POWER,
    NON_TIME_COLS,
    _filter_anchor_estimates,
    _pair_vote_confidence,
    _to_tensor,
    _variant_distance_weight,
    load_model,
    select_device,
)


FINE_GRID: dict[str, list[float]] = {
    "tie_gate_threshold": [0.50, 0.55, 0.60],
    "tie_shrink_power": [0.75, 1.00, 1.25],
    "tie_margin_weight_alpha": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "min_anchor_quality": [DEFAULT_MIN_ANCHOR_QUALITY],
    "anchor_outlier_mad_scale": [DEFAULT_OUTLIER_MAD_SCALE],
    "anchor_outlier_min_delta": [DEFAULT_OUTLIER_MIN_DELTA],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 variant 分开精调 score_program 评分参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="train_set/model_transformer.pt")
    parser.add_argument("--anchor-set", default="train_set/anchor_set.parquet")
    parser.add_argument("--zscore", default="train_set/run_features_zscore.parquet")
    parser.add_argument("--time-scores", default="train_set/time_scores.parquet")
    parser.add_argument("--output-prefix", default="train_set/score_tune_fine_variant")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--variants",
        default="",
        help="要调参的 query variant，逗号分隔；为空时自动使用 zscore 中全部 variant",
    )
    parser.add_argument(
        "--tie-gate-thresholds",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["tie_gate_threshold"]),
        help="细粒度 tie gate 阈值列表，逗号分隔",
    )
    parser.add_argument(
        "--tie-shrink-powers",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["tie_shrink_power"]),
        help="细粒度 tie shrink power 列表，逗号分隔",
    )
    parser.add_argument(
        "--tie-margin-weight-alphas",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["tie_margin_weight_alpha"]),
        help="非 tie pair 的 class confidence / direction margin 混合系数列表，逗号分隔",
    )
    parser.add_argument(
        "--min-anchor-quality-values",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["min_anchor_quality"]),
        help="最小锚点质量门槛列表，逗号分隔",
    )
    parser.add_argument(
        "--anchor-outlier-mad-scales",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["anchor_outlier_mad_scale"]),
        help="锚点离群 MAD 倍率列表，逗号分隔",
    )
    parser.add_argument(
        "--anchor-outlier-min-deltas",
        default=",".join(f"{x:.2f}" for x in FINE_GRID["anchor_outlier_min_delta"]),
        help="锚点离群最小 delta 列表，逗号分隔",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每个 variant 在 JSON 中保留前 K 个候选",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        default=0,
        help="只跑前 N 个组合，用于快速验证；0 表示跑完整网格",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str, cast: type[float] | type[str]) -> list[Any]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if cast is str:
        return items
    values = [cast(item) for item in items]
    deduped: list[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    r, _ = stats.pearsonr(a, b)
    return float(r)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    r, _ = stats.spearmanr(a, b)
    return float(r)


def _safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def _metric_desc(value: float) -> float:
    return -1e18 if np.isnan(value) else float(value)


def _metric_asc(value: float) -> float:
    return 1e18 if np.isnan(value) else float(value)


def _decode_pair_from_probs(
    raw_log_ratio: float,
    class_probs: list[float],
    tie_gate_threshold: float,
    tie_shrink_power: float,
) -> dict[str, float | str | list[float]]:
    cls_idx = int(np.argmax(class_probs))
    decoded_class = CLASS_LABELS[cls_idx]
    tie_prob = float(class_probs[1])
    cls_conf = float(class_probs[cls_idx])
    gated_scale = max(0.0, 1.0 - tie_prob) ** tie_shrink_power
    gated_log_ratio = float(raw_log_ratio) * gated_scale

    if decoded_class == "tie" and tie_prob >= tie_gate_threshold:
        gated_log_ratio = 0.0
    elif decoded_class == "i_better":
        gated_log_ratio = abs(gated_log_ratio)
    elif decoded_class == "j_better":
        gated_log_ratio = -abs(gated_log_ratio)

    return {
        "decoded_class": decoded_class,
        "tie_prob": tie_prob,
        "class_confidence": cls_conf,
        "gated_scale": gated_scale,
        "gated_log_ratio": gated_log_ratio,
        "class_probs": [float(x) for x in class_probs],
    }


@torch.no_grad()
def build_pair_cache(
    df_queries: pd.DataFrame,
    df_anchors: pd.DataFrame,
    model: Any,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_lookup = {
        (str(row["program"]), str(row["variant"])): float(row["score_gt"])
        for _, row in df_anchors.iterrows()
    }
    anchor_map: dict[str, list[dict[str, Any]]] = {}
    for _, row in df_anchors.iterrows():
        anchor_map.setdefault(str(row["program"]), []).append(row.to_dict())

    query_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for query_id, (_, qrow) in enumerate(df_queries.iterrows()):
        program = str(qrow["program"])
        query_variant = str(qrow["variant"])
        query_rows.append({
            "query_id": query_id,
            "program": program,
            "variant": query_variant,
            "score_gt": gt_lookup.get((program, query_variant), float("nan")),
        })

        xi = _to_tensor({c: float(qrow.get(c, 0.0)) for c in NON_TIME_COLS}, device)
        anchors = [
            anchor for anchor in anchor_map.get(program, [])
            if str(anchor.get("variant", "")) != query_variant
        ]
        if not anchors:
            anchors = anchor_map.get(program, [])

        for anc in anchors:
            xj = _to_tensor({c: float(anc.get(c, 0.0)) for c in NON_TIME_COLS}, device)
            pred_log_ratio, cls_logits = model.forward_with_aux(xi, xj)
            probs = torch.softmax(cls_logits, dim=-1).detach().cpu().numpy().reshape(-1)
            pair_rows.append({
                "query_id": query_id,
                "program": program,
                "query_variant": query_variant,
                "anchor_variant": str(anc.get("variant", "")),
                "anchor_score_gt": float(anc.get("score_gt", 0.0)),
                "anchor_quality": float(anc.get("anchor_quality", 1.0) or 0.0),
                "distance_weight": float(_variant_distance_weight(query_variant, str(anc.get("variant", "")))),
                "raw_log_ratio": float(pred_log_ratio.cpu().item()),
                "prob_i": float(probs[0]),
                "prob_tie": float(probs[1]),
                "prob_j": float(probs[2]),
            })

    return pd.DataFrame(pair_rows), pd.DataFrame(query_rows)


def score_queries_for_params(
    pair_cache: pd.DataFrame,
    query_meta: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    meta_by_query = query_meta.set_index("query_id")

    for query_id, group in pair_cache.groupby("query_id", sort=False):
        anchor_estimates: list[dict[str, Any]] = []
        for row in group.itertuples(index=False):
            decoded = _decode_pair_from_probs(
                raw_log_ratio=float(row.raw_log_ratio),
                class_probs=[float(row.prob_i), float(row.prob_tie), float(row.prob_j)],
                tie_gate_threshold=float(params["tie_gate_threshold"]),
                tie_shrink_power=float(params["tie_shrink_power"]),
            )
            effective_confidence, _, _ = _pair_vote_confidence(
                list(decoded["class_probs"]),
                str(decoded["decoded_class"]),
                tie_margin_weight_alpha=float(params["tie_margin_weight_alpha"]),
            )

            base_weight = (
                float(row.anchor_quality)
                * float(row.distance_weight)
                * float(effective_confidence)
            )
            if float(row.anchor_quality) < float(params["min_anchor_quality"]):
                base_weight = 0.0

            score_estimate_raw = float(row.anchor_score_gt) + float(decoded["gated_log_ratio"])
            anchor_estimates.append({
                "weight": round(base_weight, 6),
                "score_estimate_raw": round(score_estimate_raw, 6),
                "used": base_weight > 0.0,
            })

        anchor_estimates = _filter_anchor_estimates(
            anchor_estimates,
            mad_scale=float(params["anchor_outlier_mad_scale"]),
            min_delta=float(params["anchor_outlier_min_delta"]),
        )
        kept = [item for item in anchor_estimates if item.get("used")]
        if not kept:
            kept = anchor_estimates

        weights = np.array([float(item["weight"]) for item in kept], dtype=np.float64)
        estimates = np.array([float(item["score_estimate_raw"]) for item in kept], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
            final_score = float(estimates.mean())
        else:
            final_score = float(np.average(estimates, weights=weights))

        meta = meta_by_query.loc[query_id]
        records.append({
            "query_id": int(query_id),
            "program": str(meta["program"]),
            "variant": str(meta["variant"]),
            "score_log": round(final_score, 6),
            "score_gt": float(meta["score_gt"]),
            "n_anchors_total": int(len(anchor_estimates)),
            "n_anchors_used": int(len(kept)),
        })

    return pd.DataFrame(records)


def evaluate_variant(
    scored: pd.DataFrame,
    df_time: pd.DataFrame,
    variant: str,
) -> dict[str, Any]:
    subset = scored[scored["variant"] == variant].copy()
    merged = subset.merge(
        df_time[[c for c in ["program", "variant", "score_time", "score_time_loose"] if c in df_time.columns]],
        on=["program", "variant"],
        how="left",
    )

    time_valid = merged.dropna(subset=["score_log", "score_time"]).copy()
    score_valid = merged.dropna(subset=["score_log", "score_gt"]).copy()

    time_pred = time_valid["score_log"].to_numpy(dtype=float)
    time_ref = time_valid["score_time"].to_numpy(dtype=float)
    score_pred = score_valid["score_log"].to_numpy(dtype=float)
    score_ref = score_valid["score_gt"].to_numpy(dtype=float)

    return {
        "variant": variant,
        "n_runs": int(len(subset)),
        "n_time_valid": int(len(time_valid)),
        "n_score_valid": int(len(score_valid)),
        "time_corr": round(_safe_pearson(time_pred, time_ref), 6),
        "time_spearman": round(_safe_spearman(time_pred, time_ref), 6),
        "time_mae": round(_safe_mae(time_pred, time_ref), 6),
        "score_corr": round(_safe_pearson(score_pred, score_ref), 6),
        "score_mae": round(_safe_mae(score_pred, score_ref), 6),
    }


def evaluate_overall(scored: pd.DataFrame, df_time: pd.DataFrame) -> dict[str, Any]:
    merged = scored.merge(
        df_time[[c for c in ["program", "variant", "score_time", "score_time_loose"] if c in df_time.columns]],
        on=["program", "variant"],
        how="left",
    )
    time_valid = merged.dropna(subset=["score_log", "score_time"]).copy()
    score_valid = merged.dropna(subset=["score_log", "score_gt"]).copy()

    time_pred = time_valid["score_log"].to_numpy(dtype=float)
    time_ref = time_valid["score_time"].to_numpy(dtype=float)
    score_pred = score_valid["score_log"].to_numpy(dtype=float)
    score_ref = score_valid["score_gt"].to_numpy(dtype=float)

    return {
        "variant": "ALL",
        "n_runs": int(len(scored)),
        "n_time_valid": int(len(time_valid)),
        "n_score_valid": int(len(score_valid)),
        "time_corr": round(_safe_pearson(time_pred, time_ref), 6),
        "time_spearman": round(_safe_spearman(time_pred, time_ref), 6),
        "time_mae": round(_safe_mae(time_pred, time_ref), 6),
        "score_corr": round(_safe_pearson(score_pred, score_ref), 6),
        "score_mae": round(_safe_mae(score_pred, score_ref), 6),
    }


def trial_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _metric_desc(float(row.get("time_corr", float("nan")))),
        _metric_desc(float(row.get("time_spearman", float("nan")))),
        _metric_desc(float(row.get("score_corr", float("nan")))),
        -_metric_asc(float(row.get("time_mae", float("nan")))),
        -_metric_asc(float(row.get("score_mae", float("nan")))),
    )


def main() -> None:
    args = parse_args()

    model_path = (REPO_ROOT / args.model).resolve()
    anchor_path = (REPO_ROOT / args.anchor_set).resolve()
    zscore_path = (REPO_ROOT / args.zscore).resolve()
    time_scores_path = (REPO_ROOT / args.time_scores).resolve()
    output_prefix = (REPO_ROOT / args.output_prefix).resolve()
    trials_path = output_prefix.with_name(output_prefix.name + "_trials.csv")
    best_path = output_prefix.with_name(output_prefix.name + "_best.json")

    for path in [model_path, anchor_path, zscore_path, time_scores_path]:
        if not path.exists():
            sys.exit(f"[error] 输入文件不存在: {path}")

    device = select_device(args.device)
    model = load_model(model_path, device)

    df_anchors = pd.read_parquet(anchor_path)
    df_queries = pd.read_parquet(zscore_path)
    df_time = pd.read_parquet(time_scores_path)

    variants = _parse_csv_list(args.variants, str) if args.variants else sorted(df_queries["variant"].dropna().unique().tolist())
    df_queries = df_queries[df_queries["variant"].isin(variants)].copy()
    if df_queries.empty:
        sys.exit("[error] 指定 variants 后没有可调参的 query")

    grid = {
        "tie_gate_threshold": _parse_csv_list(args.tie_gate_thresholds, float),
        "tie_shrink_power": _parse_csv_list(args.tie_shrink_powers, float),
        "tie_margin_weight_alpha": _parse_csv_list(args.tie_margin_weight_alphas, float),
        "min_anchor_quality": _parse_csv_list(args.min_anchor_quality_values, float),
        "anchor_outlier_mad_scale": _parse_csv_list(args.anchor_outlier_mad_scales, float),
        "anchor_outlier_min_delta": _parse_csv_list(args.anchor_outlier_min_deltas, float),
    }

    combos = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*grid.values())
    ]
    if args.limit_combos > 0:
        combos = combos[: args.limit_combos]
    if not combos:
        sys.exit("[error] 参数组合为空")

    print(f"[info] tuning variants: {', '.join(variants)}")
    print(f"[info] fine-grid combos: {len(combos)}")

    pair_cache, query_meta = build_pair_cache(df_queries, df_anchors, model, device)
    print(f"[info] cached query-anchor pairs: {len(pair_cache)}")

    all_trials: list[dict[str, Any]] = []
    for combo_idx, params in enumerate(combos, start=1):
        scored = score_queries_for_params(pair_cache, query_meta, params)
        overall = evaluate_overall(scored, df_time)
        overall.update(params)
        overall["combo_index"] = combo_idx
        all_trials.append(overall)

        for variant in variants:
            metrics = evaluate_variant(scored, df_time, variant)
            metrics.update(params)
            metrics["combo_index"] = combo_idx
            all_trials.append(metrics)

        print(
            "[trial] "
            f"{combo_idx:03d}/{len(combos)} "
            f"gate={params['tie_gate_threshold']:.2f} "
            f"shrink={params['tie_shrink_power']:.2f} "
            f"alpha={params['tie_margin_weight_alpha']:.2f} "
            f"overall_time_corr={overall['time_corr']:.4f}"
        )

    trials_df = pd.DataFrame(all_trials)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_path, index=False)

    summary: dict[str, Any] = {
        "variants": variants,
        "n_combos": len(combos),
        "grid": grid,
        "best_by_variant": {},
    }
    for variant in ["ALL", *variants]:
        subset = trials_df[trials_df["variant"] == variant].copy()
        rows = subset.to_dict(orient="records")
        rows.sort(key=trial_sort_key, reverse=True)
        summary["best_by_variant"][variant] = {
            "best": rows[0] if rows else None,
            "top_trials": rows[: max(args.top_k, 1)],
        }

    best_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"[ok] trials: {trials_path}")
    print(f"[ok] best:   {best_path}")
    for variant in variants:
        best = summary["best_by_variant"][variant]["best"]
        if not best:
            print(f"[warn] {variant}: no valid trial")
            continue
        print(
            f"[best] {variant}: "
            f"gate={best['tie_gate_threshold']:.2f} "
            f"shrink={best['tie_shrink_power']:.2f} "
            f"alpha={best['tie_margin_weight_alpha']:.2f} "
            f"time_corr={best['time_corr']:.4f} "
            f"score_corr={best['score_corr']:.4f}"
        )


if __name__ == "__main__":
    main()