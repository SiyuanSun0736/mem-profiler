#!/usr/bin/env python3
"""
Compare the hotspot-window-token experiment against the baseline PairTransformer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
EXPERIMENT_ROOT = SCRIPTS_ROOT / "experimental" / "hotspot_token"
for candidate in (SCRIPTS_ROOT, EXPERIMENT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from feature_columns import NON_TIME_COLS
from train_transformer import (
    NEAR_TIE_THRESHOLD,
    PairTransformer,
    TIE_THRESHOLD,
    compute_aux_metrics,
    compute_metrics,
    predict_with_aux_np,
    select_device,
    split_by_program,
)

from hotspot_token_schema import apply_hotspot_token_scaler, build_hotspot_token_map
from train_hotspot_token_transformer import (
    HotspotWindowTokenPairTransformer,
    _collect_run_keys,
    _load_pairs_table,
    _load_run_features,
    predict_with_aux_np_hotspot,
)


VARIANT_ORDER = {"O0": 0, "O1": 1, "O2": 2, "O3": 3}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canonical_pair(variant_i: str, variant_j: str) -> str:
    ordered = sorted((variant_i, variant_j), key=lambda name: VARIANT_ORDER[name])
    return f"{ordered[0]}-{ordered[1]}"


def _series_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    return float(value)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _delta_metrics(base: dict[str, Any], exp: dict[str, Any]) -> dict[str, float | None]:
    keys = ("mae", "rmse", "r2", "dir_acc", "acc_3cls", "aux_acc_3cls", "aux_tie_recall")
    out: dict[str, float | None] = {}
    for key in keys:
        base_value = _series_float(base.get(key))
        exp_value = _series_float(exp.get(key))
        out[key] = round(exp_value - base_value, 4) if base_value is not None and exp_value is not None else None
    return out


def _slice_summary(df: pd.DataFrame, y_pred: np.ndarray, cls_logits: np.ndarray) -> dict[str, Any]:
    y_true = df["log_ratio"].values.astype(np.float32)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(compute_aux_metrics(y_true, cls_logits))
    metrics["label_counts"] = {
        key: int(value)
        for key, value in df["label_class"].value_counts().sort_index().to_dict().items()
    }
    return metrics


def _mask_tie(df: pd.DataFrame) -> pd.Series:
    return df["log_ratio"].abs() <= TIE_THRESHOLD


def _mask_near_tie(df: pd.DataFrame, near_tie_threshold: float) -> pd.Series:
    abs_lr = df["log_ratio"].abs()
    return (abs_lr > TIE_THRESHOLD) & (abs_lr <= near_tie_threshold)


def _mask_o2_o3(df: pd.DataFrame) -> pd.Series:
    canonical = df.apply(lambda row: _canonical_pair(row["variant_i"], row["variant_j"]), axis=1)
    return canonical == "O2-O3"


def _report_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _load_baseline_model(path: pathlib.Path, device: torch.device) -> PairTransformer:
    checkpoint = torch.load(path, map_location=device)
    hparams = checkpoint.get("hparams", {})
    model = PairTransformer(
        feat_dim=int(hparams.get("feat_dim", len(NON_TIME_COLS))),
        d_model=int(hparams.get("d_model", 64)),
        nhead=int(hparams.get("nhead", 4)),
        num_layers=int(hparams.get("num_layers", hparams.get("nlayers", 3))),
        dim_feedforward=int(hparams.get("dim_feedforward", hparams.get("ffn_dim", 256))),
        dropout=float(hparams.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _load_hotspot_model(path: pathlib.Path, device: torch.device) -> tuple[HotspotWindowTokenPairTransformer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    hparams = checkpoint.get("hparams", {})
    model = HotspotWindowTokenPairTransformer(
        summary_dim=int(hparams.get("summary_dim", len(NON_TIME_COLS))),
        window_feat_dim=int(hparams.get("window_feat_dim", 10)),
        top_k=int(hparams.get("top_k", 6)),
        d_model=int(hparams.get("d_model", 64)),
        nhead=int(hparams.get("nhead", 4)),
        num_layers=int(hparams.get("num_layers", hparams.get("nlayers", 3))),
        dim_feedforward=int(hparams.get("dim_feedforward", hparams.get("ffn_dim", 256))),
        dropout=float(hparams.get("dropout", 0.1)),
        head_hidden=int(hparams.get("head_hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("hotspot_scaler", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline PairTransformer against hotspot-window-token experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--run-features", default="train_set/run_features.parquet")
    parser.add_argument("--baseline-model", default="train_set/model_transformer.pt")
    parser.add_argument("--baseline-eval", default="train_set/model_transformer_eval.json")
    parser.add_argument(
        "--experiment-model",
        default="train_set/hotspot_token_transformer/model_hotspot_token_transformer.pt",
    )
    parser.add_argument(
        "--experiment-eval",
        default="train_set/hotspot_token_transformer/model_hotspot_token_eval.json",
    )
    parser.add_argument("--output-dir", default="train_set/hotspot_token_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-tie-threshold", type=float, default=NEAR_TIE_THRESHOLD)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pairs_path = (REPO_ROOT / args.pairs).resolve()
    run_features_path = (REPO_ROOT / args.run_features).resolve()
    baseline_model_path = (REPO_ROOT / args.baseline_model).resolve()
    baseline_eval_path = (REPO_ROOT / args.baseline_eval).resolve()
    experiment_model_path = (REPO_ROOT / args.experiment_model).resolve()
    experiment_eval_path = (REPO_ROOT / args.experiment_eval).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (pairs_path, run_features_path, baseline_model_path, baseline_eval_path, experiment_model_path, experiment_eval_path):
        if not path.exists():
            sys.exit(f"[error] missing required file: {path}")

    device = select_device(args.device)
    df_pairs = _load_pairs_table(pairs_path)
    _, _, df_test = split_by_program(df_pairs, seed=args.seed)
    df_run = _load_run_features(run_features_path)
    test_keys = _collect_run_keys(df_test)
    run_key_df = df_run[
        [(str(program), str(variant)) in test_keys for program, variant in zip(df_run["program"], df_run["variant"])]
    ].copy()

    baseline_eval = _load_json(baseline_eval_path)
    experiment_eval = _load_json(experiment_eval_path)
    baseline_model = _load_baseline_model(baseline_model_path, device)
    experiment_model, scaler = _load_hotspot_model(experiment_model_path, device)

    raw_token_map, _ = build_hotspot_token_map(run_key_df, top_k=int(experiment_eval.get("token_schema", {}).get("top_k", 6)))
    token_map = apply_hotspot_token_scaler(raw_token_map, scaler)

    baseline_pred, baseline_logits = predict_with_aux_np(baseline_model, df_test, device)
    experiment_pred, experiment_logits = predict_with_aux_np_hotspot(experiment_model, df_test, token_map, device)

    slices = {
        "overall_test": pd.Series(True, index=df_test.index),
        "tie": _mask_tie(df_test),
        "near_tie": _mask_near_tie(df_test, near_tie_threshold=args.near_tie_threshold),
        "O2-O3": _mask_o2_o3(df_test),
    }

    slice_results: dict[str, Any] = {}
    for slice_name, mask in slices.items():
        subset = df_test[mask]
        if subset.empty:
            slice_results[slice_name] = {"n": 0, "baseline": None, "experiment": None, "delta": None}
            continue
        base_metrics = _slice_summary(subset, baseline_pred[mask.to_numpy()], baseline_logits[mask.to_numpy()])
        exp_metrics = _slice_summary(subset, experiment_pred[mask.to_numpy()], experiment_logits[mask.to_numpy()])
        slice_results[slice_name] = {
            "n": int(len(subset)),
            "baseline": base_metrics,
            "experiment": exp_metrics,
            "delta": _delta_metrics(base_metrics, exp_metrics),
        }

    overall = slice_results["overall_test"]
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": args.seed,
        "pairs_path": str(pairs_path.relative_to(REPO_ROOT)),
        "run_features_path": str(run_features_path.relative_to(REPO_ROOT)),
        "baseline": {
            "model_path": str(baseline_model_path.relative_to(REPO_ROOT)),
            "eval_path": str(baseline_eval_path.relative_to(REPO_ROOT)),
            "model_name": baseline_eval.get("model"),
            "n_params": baseline_eval.get("n_params"),
        },
        "experiment": {
            "model_path": str(experiment_model_path.relative_to(REPO_ROOT)),
            "eval_path": str(experiment_eval_path.relative_to(REPO_ROOT)),
            "model_name": experiment_eval.get("model"),
            "n_params": experiment_eval.get("n_params"),
            "config": experiment_eval.get("config"),
        },
        "thresholds": {
            "tie_threshold": TIE_THRESHOLD,
            "near_tie_threshold": args.near_tie_threshold,
        },
        "split": {
            "test_programs": int(df_test["program"].nunique()),
            "test_pairs": int(len(df_test)),
        },
        "overall_test": overall,
        "focused_slices": {key: value for key, value in slice_results.items() if key != "overall_test"},
    }

    json_path = out_dir / "hotspot_vs_baseline_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    overall_rows = []
    for metric in ("mae", "rmse", "r2", "dir_acc", "acc_3cls", "aux_acc_3cls", "aux_tie_recall"):
        overall_rows.append([
            metric,
            _fmt(_series_float(overall["baseline"].get(metric))),
            _fmt(_series_float(overall["experiment"].get(metric))),
            _fmt(_series_float(overall["delta"].get(metric))),
        ])

    slice_rows = []
    detail_rows = []
    for slice_name in ("tie", "near_tie", "O2-O3"):
        record = slice_results[slice_name]
        if not record["baseline"]:
            slice_rows.append([slice_name, str(record["n"]), "-", "-", "-", "-", "-"])
            continue
        slice_rows.append([
            slice_name,
            str(record["n"]),
            _fmt(_series_float(record["baseline"].get("dir_acc"))),
            _fmt(_series_float(record["experiment"].get("dir_acc"))),
            _fmt(_series_float(record["delta"].get("dir_acc"))),
            _fmt(_series_float(record["baseline"].get("aux_tie_recall"))),
            _fmt(_series_float(record["experiment"].get("aux_tie_recall"))),
        ])
        detail_rows.append([
            slice_name,
            _fmt(_series_float(record["baseline"].get("acc_3cls"))),
            _fmt(_series_float(record["experiment"].get("acc_3cls"))),
            _fmt(_series_float(record["delta"].get("acc_3cls"))),
            _fmt(_series_float(record["baseline"].get("aux_acc_3cls"))),
            _fmt(_series_float(record["experiment"].get("aux_acc_3cls"))),
            _fmt(_series_float(record["delta"].get("aux_acc_3cls"))),
        ])

    lines = [
        "# Hotspot Token vs Baseline Comparison",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Setup",
        "",
        f"- pairs: {summary['pairs_path']}",
        f"- run_features: {summary['run_features_path']}",
        f"- seed: {summary['seed']}",
        f"- test programs: {summary['split']['test_programs']}",
        f"- test pairs: {summary['split']['test_pairs']}",
        f"- baseline model: {summary['baseline']['model_name']} ({summary['baseline']['model_path']})",
        f"- experiment model: {summary['experiment']['model_name']} ({summary['experiment']['model_path']})",
        "",
        "## Overall Test",
        "",
        *_report_table(["Metric", "Baseline", "HotspotToken", "Delta(exp-base)"], overall_rows),
        "",
        "## Focused Slices",
        "",
        *_report_table(["Slice", "n", "dir_acc(base)", "dir_acc(exp)", "delta", "tie_rec(base)", "tie_rec(exp)"], slice_rows),
        "",
        *_report_table(["Slice", "acc_3cls(base)", "acc_3cls(exp)", "delta", "aux_3cls(base)", "aux_3cls(exp)", "delta"], detail_rows),
        "",
        "## Notes",
        "",
        f"- tie uses |log_ratio| <= {TIE_THRESHOLD}",
        f"- near_tie uses {TIE_THRESHOLD} < |log_ratio| <= {args.near_tie_threshold}",
        "- O2-O3 is computed as an unordered variant pair, so both directions are included.",
    ]

    markdown_path = out_dir / "hotspot_vs_baseline_comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n")

    print(f"[ok] comparison json: {json_path}")
    print(f"[ok] comparison markdown: {markdown_path}")


if __name__ == "__main__":
    main()