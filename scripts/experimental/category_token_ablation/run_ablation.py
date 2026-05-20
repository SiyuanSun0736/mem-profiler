#!/usr/bin/env python3
"""
Category-token ablation workflow.

Goal: determine whether the previous degradation comes mainly from semantic grouping
choices or from adding extra tokens at all.

This workflow is isolated from the existing baseline/category/hotspot experiments.
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
CATEGORY_ROOT = SCRIPTS_ROOT / "experimental" / "category_token"
for candidate in (SCRIPTS_ROOT, CATEGORY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from feature_columns import DROPPED_INPUT_FEATURES, NON_TIME_COLS
from train_transformer import (
    DEFAULT_AUX_CLASS_LAMBDA,
    DEFAULT_CLASS_BALANCE_POWER,
    DEFAULT_NEAR_TIE_REG_WEIGHT,
    DEFAULT_TIE_REG_WEIGHT,
    LOG_RATIO_CLIP,
    NEAR_TIE_THRESHOLD,
    PairTransformer,
    TIE_THRESHOLD,
    class_weight_dict,
    compute_aux_metrics,
    compute_metrics,
    magnitude_bin_counts,
    predict_with_aux_np,
    select_device,
    split_by_program,
    train,
)

from train_category_token_transformer import CategoryTokenPairTransformer
from ablation_schemas import AblationVariant, build_ablation_variants, build_schema_metadata


EXPERIMENT_CONFIGS: dict[str, dict[str, float | int]] = {
    "category_token_ablation_base": {
        "d_model": 64,
        "nhead": 4,
        "nlayers": 3,
        "ffn_dim": 256,
        "dropout": 0.10,
        "head_hidden": 64,
        "lr": 1.5e-4,
        "wd": 1e-4,
        "epochs": 300,
        "batch": 64,
        "patience": 55,
        "clip": LOG_RATIO_CLIP,
        "huber_delta": 0.5,
        "direction_lambda": 0.0,
        "aux_class_lambda": DEFAULT_AUX_CLASS_LAMBDA,
        "near_tie_threshold": NEAR_TIE_THRESHOLD,
        "tie_reg_weight": DEFAULT_TIE_REG_WEIGHT,
        "near_tie_reg_weight": DEFAULT_NEAR_TIE_REG_WEIGHT,
        "class_balance_power": DEFAULT_CLASS_BALANCE_POWER,
        "noise_std": 0.008,
    }
}


def _load_pairs_table(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] missing pairs table: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    sys.exit(f"[error] unsupported pairs table format: {path.suffix}")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _mask_tie(df: pd.DataFrame) -> pd.Series:
    return df["log_ratio"].abs() <= TIE_THRESHOLD


def _mask_near_tie(df: pd.DataFrame, near_tie_threshold: float) -> pd.Series:
    abs_lr = df["log_ratio"].abs()
    return (abs_lr > TIE_THRESHOLD) & (abs_lr <= near_tie_threshold)


def _mask_o2_o3(df: pd.DataFrame) -> pd.Series:
    return (
        ((df["variant_i"] == "O2") & (df["variant_j"] == "O3"))
        | ((df["variant_i"] == "O3") & (df["variant_j"] == "O2"))
    )


def _slice_summary(df: pd.DataFrame, y_pred: np.ndarray, cls_logits: np.ndarray) -> dict[str, Any]:
    y_true = df["log_ratio"].values.astype(np.float32)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(compute_aux_metrics(y_true, cls_logits))
    return metrics


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


def _load_category_model(path: pathlib.Path, device: torch.device) -> CategoryTokenPairTransformer:
    checkpoint = torch.load(path, map_location=device)
    category_feature_map = checkpoint.get("category_feature_map") or {}
    category_index_map = {name: [NON_TIME_COLS.index(col) for col in cols] for name, cols in category_feature_map.items()}
    hparams = checkpoint.get("hparams", {})
    model = CategoryTokenPairTransformer(
        feat_dim=int(hparams.get("feat_dim", len(NON_TIME_COLS))),
        category_index_map=category_index_map,
        d_model=int(hparams.get("d_model", 64)),
        nhead=int(hparams.get("nhead", 4)),
        num_layers=int(hparams.get("num_layers", hparams.get("nlayers", 3))),
        dim_feedforward=int(hparams.get("dim_feedforward", hparams.get("ffn_dim", 256))),
        dropout=float(hparams.get("dropout", 0.1)),
        head_hidden=int(hparams.get("head_hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _build_variant_eval(
    variant: AblationVariant,
    model: CategoryTokenPairTransformer,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    device: torch.device,
    out_dir: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    results: dict[str, dict[str, Any]] = {}
    predictions_test: np.ndarray | None = None
    logits_test: np.ndarray | None = None
    per_pair: dict[str, dict[str, Any]] = {}
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        pred, cls_logits = predict_with_aux_np(model, part, device)
        y_true = part["log_ratio"].values.astype(np.float32)
        metrics = compute_metrics(y_true, pred)
        metrics.update(compute_aux_metrics(y_true, cls_logits))
        results[name] = metrics
        if name == "test":
            predictions_test = pred
            logits_test = cls_logits

    for variant_i in ("O0", "O1", "O2"):
        for variant_j in ("O1", "O2", "O3"):
            if variant_i >= variant_j:
                continue
            mask = (df_test["variant_i"] == variant_i) & (df_test["variant_j"] == variant_j)
            if mask.sum() == 0:
                continue
            subset = df_test[mask]
            pred, cls_logits = predict_with_aux_np(model, subset, device)
            y_true = subset["log_ratio"].values.astype(np.float32)
            metrics = compute_metrics(y_true, pred)
            metrics.update(compute_aux_metrics(y_true, cls_logits))
            per_pair[f"{variant_i}-{variant_j}"] = metrics

    schema = build_schema_metadata(NON_TIME_COLS, variant.category_feature_map or {})
    eval_result = {
        "model": "CategoryTokenPairTransformer",
        "variant": variant.name,
        "seed": args.seed,
        "pairs_path": str((REPO_ROOT / args.pairs).relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "description": variant.description,
        "architecture": (
            f"[summary+{len(variant.category_feature_map or {})} ablation tokens]x2->"
            f"TransformerEncoder({args.nlayers}L,{args.nhead}H,ffn={args.ffn_dim})->"
            "[summary_i;summary_j;summary_i-summary_j]->reg_head+cls_head"
        ),
        "token_schema": schema,
        "n_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "device": str(device),
        "training_objective": {
            "aux_class_lambda": args.aux_class_lambda,
            "class_balance_power": args.class_balance_power,
            "class_weights": class_weight_dict(
                df_train["label_int"].values.astype(np.int64),
                power=args.class_balance_power,
            ),
        },
        "tie_strategy": {
            "tie_threshold": TIE_THRESHOLD,
            "near_tie_threshold": args.near_tie_threshold,
            "tie_reg_weight": args.tie_reg_weight,
            "near_tie_reg_weight": args.near_tie_reg_weight,
        },
        "splits": {
            "train_programs": int(df_train["program"].nunique()),
            "val_programs": int(df_val["program"].nunique()),
            "test_programs": int(df_test["program"].nunique()),
            "train_pairs": len(df_train),
            "val_pairs": len(df_val),
            "test_pairs": len(df_test),
        },
        "split_magnitude_bins": {
            name: magnitude_bin_counts(
                part["log_ratio"].values.astype(np.float32),
                near_tie_threshold=args.near_tie_threshold,
            )
            for name, part in (("train", df_train), ("val", df_val), ("test", df_test))
        },
        "results": results,
        "per_pair": per_pair,
    }
    assert predictions_test is not None and logits_test is not None
    return eval_result, predictions_test, logits_test


def _train_variant(
    variant: AblationVariant,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    device: torch.device,
    out_root: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    variant_out = out_root / variant.name
    variant_out.mkdir(parents=True, exist_ok=True)
    category_feature_map = variant.category_feature_map or {}
    category_index_map = {
        name: [NON_TIME_COLS.index(col) for col in cols]
        for name, cols in category_feature_map.items()
    }
    model = CategoryTokenPairTransformer(
        feat_dim=len(NON_TIME_COLS),
        category_index_map=category_index_map,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.nlayers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
    ).to(device)

    history = train(
        model=model,
        device=device,
        df_train=df_train,
        df_val=df_val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        patience=args.patience,
        huber_delta=args.huber_delta,
        noise_std=args.noise_std,
        direction_lambda=args.direction_lambda,
        aux_class_lambda=args.aux_class_lambda,
        near_tie_threshold=args.near_tie_threshold,
        tie_reg_weight=args.tie_reg_weight,
        near_tie_reg_weight=args.near_tie_reg_weight,
        class_balance_power=args.class_balance_power,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "hparams": {
                "feat_dim": len(NON_TIME_COLS),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "nlayers": args.nlayers,
                "num_layers": args.nlayers,
                "ffn_dim": args.ffn_dim,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
                "head_hidden": args.head_hidden,
            },
            "non_time_cols": NON_TIME_COLS,
            "category_feature_map": category_feature_map,
        },
        variant_out / "model_category_token_ablation.pt",
    )

    eval_result, predictions_test, logits_test = _build_variant_eval(
        variant=variant,
        model=model,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        device=device,
        out_dir=variant_out,
        args=args,
    )
    eval_result["history"] = {key: [round(value, 6) for value in values] for key, values in history.items()}
    eval_path = variant_out / "model_category_token_ablation_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))
    return eval_result, predictions_test, logits_test


def _load_reference_variant(
    variant: AblationVariant,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model_path = (REPO_ROOT / str(variant.reference_model_path)).resolve()
    eval_path = (REPO_ROOT / str(variant.reference_eval_path)).resolve()
    if not model_path.exists() or not eval_path.exists():
        sys.exit(f"[error] missing reference files for variant {variant.name}")
    model = _load_category_model(model_path, device)
    eval_result, predictions_test, logits_test = _build_variant_eval(
        variant=variant,
        model=model,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        device=device,
        out_dir=eval_path.parent,
        args=argparse.Namespace(
            seed=42,
            pairs="train_set/pairs.parquet",
            nlayers=3,
            nhead=4,
            ffn_dim=256,
            aux_class_lambda=DEFAULT_AUX_CLASS_LAMBDA,
            class_balance_power=DEFAULT_CLASS_BALANCE_POWER,
            near_tie_threshold=NEAR_TIE_THRESHOLD,
            tie_reg_weight=DEFAULT_TIE_REG_WEIGHT,
            near_tie_reg_weight=DEFAULT_NEAR_TIE_REG_WEIGHT,
        ),
    )
    loaded_eval = _load_json(eval_path)
    eval_result["reference_eval_results"] = loaded_eval.get("results")
    return eval_result, predictions_test, logits_test


def _diagnose(summary_rows: list[dict[str, Any]]) -> str:
    summary_only = next((row for row in summary_rows if row["name"] == "summary_only"), None)
    coarse_2way = next((row for row in summary_rows if row["name"] == "coarse_2way"), None)
    no_mm_phase_4way = next((row for row in summary_rows if row["name"] == "no_mm_phase_4way"), None)
    semantic_full = next((row for row in summary_rows if row["name"] == "semantic_full_reference"), None)

    if summary_only is None or coarse_2way is None or no_mm_phase_4way is None or semantic_full is None:
        return "insufficient_ablation_results"

    score_summary = (_series_float(summary_only["test"].get("dir_acc")) or -999, _series_float(summary_only["test"].get("acc_3cls")) or -999)
    score_coarse = (_series_float(coarse_2way["test"].get("dir_acc")) or -999, _series_float(coarse_2way["test"].get("acc_3cls")) or -999)
    score_4way = (_series_float(no_mm_phase_4way["test"].get("dir_acc")) or -999, _series_float(no_mm_phase_4way["test"].get("acc_3cls")) or -999)
    score_full = (_series_float(semantic_full["test"].get("dir_acc")) or -999, _series_float(semantic_full["test"].get("acc_3cls")) or -999)

    if score_summary >= score_coarse and score_summary >= score_4way and score_summary >= score_full:
        return "leans_multi_token_cost"
    if score_4way > score_full or score_coarse > score_full:
        return "grouping_matters_but_multi_token_still_costly"
    return "inconclusive_but_extra_tokens_not_helping"


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    preset = EXPERIMENT_CONFIGS.get(pre_args.config, {}) if pre_args.config else {}
    if pre_args.config and pre_args.config not in EXPERIMENT_CONFIGS:
        sys.exit("[error] unknown preset '" + str(pre_args.config) + "', available: " + ", ".join(EXPERIMENT_CONFIGS))

    parser = argparse.ArgumentParser(
        description="Run category-token ablations to separate grouping effects from multi-token effects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--baseline-model", default="train_set/model_transformer.pt")
    parser.add_argument("--baseline-eval", default="train_set/model_transformer_eval.json")
    parser.add_argument("--output", default="train_set/category_token_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256, dest="ffn_dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden", type=int, default=64, dest="head_hidden")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--huber-delta", type=float, default=1.0, dest="huber_delta")
    parser.add_argument("--direction-lambda", type=float, default=0.0, dest="direction_lambda")
    parser.add_argument("--aux-class-lambda", type=float, default=DEFAULT_AUX_CLASS_LAMBDA, dest="aux_class_lambda")
    parser.add_argument("--near-tie-threshold", type=float, default=NEAR_TIE_THRESHOLD, dest="near_tie_threshold")
    parser.add_argument("--tie-reg-weight", type=float, default=DEFAULT_TIE_REG_WEIGHT, dest="tie_reg_weight")
    parser.add_argument("--near-tie-reg-weight", type=float, default=DEFAULT_NEAR_TIE_REG_WEIGHT, dest="near_tie_reg_weight")
    parser.add_argument("--class-balance-power", type=float, default=DEFAULT_CLASS_BALANCE_POWER, dest="class_balance_power")
    parser.add_argument("--noise-std", type=float, default=0.0, dest="noise_std")
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip", type=float, default=LOG_RATIO_CLIP)

    if preset:
        parser.set_defaults(**preset)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs_path = (REPO_ROOT / args.pairs).resolve()
    baseline_model_path = (REPO_ROOT / args.baseline_model).resolve()
    baseline_eval_path = (REPO_ROOT / args.baseline_eval).resolve()
    out_dir = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    df = _load_pairs_table(pairs_path)
    if DROPPED_INPUT_FEATURES:
        print(f"[info] dropped dead input features: {', '.join(DROPPED_INPUT_FEATURES)}")

    xi_cols = [f"xi_{col}" for col in NON_TIME_COLS]
    xj_cols = [f"xj_{col}" for col in NON_TIME_COLS]
    missing = [col for col in xi_cols + xj_cols if col not in df.columns]
    if missing:
        sys.exit(f"[error] missing pair feature columns: {missing[:5]}...")

    sep = "=" * 72
    print(sep)
    print("Step 0: Split train / val / test by program")
    print(sep)
    df_train, df_val, df_test = split_by_program(df, seed=args.seed)
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        dist = "  ".join(
            f"{label}={part['label_class'].value_counts().get(label, 0)}"
            for label in ("i_better", "tie", "j_better")
        )
        mag_bins = magnitude_bin_counts(part["log_ratio"].values.astype(np.float32), near_tie_threshold=args.near_tie_threshold)
        mag_dist = "  ".join(f"{key}={mag_bins.get(key, 0)}" for key in mag_bins)
        print(
            f"  {name:5s}: {part['program'].nunique():3d} programs  {len(part):5d} pairs  labels: {dist}  |  |log_ratio| bins: {mag_dist}"
        )

    baseline_model = _load_baseline_model(baseline_model_path, device)
    baseline_eval = _load_json(baseline_eval_path)
    baseline_pred, baseline_logits = predict_with_aux_np(baseline_model, df_test, device)
    baseline_slices = {}
    for slice_name, mask in {
        "tie": _mask_tie(df_test),
        "near_tie": _mask_near_tie(df_test, args.near_tie_threshold),
        "O2-O3": _mask_o2_o3(df_test),
    }.items():
        subset = df_test[mask]
        baseline_slices[slice_name] = _slice_summary(subset, baseline_pred[mask.to_numpy()], baseline_logits[mask.to_numpy()])

    variants = build_ablation_variants(NON_TIME_COLS)
    print("\n" + sep)
    print("Step 1: Run ablation variants")
    print(sep)
    summary_rows: list[dict[str, Any]] = []
    per_variant_details: dict[str, Any] = {}

    for variant in variants:
        print(f"\n[variant] {variant.name}: {variant.description}")
        if variant.train_variant:
            eval_result, pred_test, logits_test = _train_variant(
                variant=variant,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                device=device,
                out_root=out_dir,
                args=args,
            )
        else:
            eval_result, pred_test, logits_test = _load_reference_variant(
                variant=variant,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                device=device,
            )

        slices = {}
        for slice_name, mask in {
            "tie": _mask_tie(df_test),
            "near_tie": _mask_near_tie(df_test, args.near_tie_threshold),
            "O2-O3": _mask_o2_o3(df_test),
        }.items():
            subset = df_test[mask]
            slices[slice_name] = _slice_summary(subset, pred_test[mask.to_numpy()], logits_test[mask.to_numpy()])

        row = {
            "name": variant.name,
            "description": variant.description,
            "test": eval_result["results"]["test"],
            "delta_vs_baseline": _delta_metrics(baseline_eval["results"]["test"], eval_result["results"]["test"]),
            "slices": {
                name: {
                    "metrics": metrics,
                    "delta_vs_baseline": _delta_metrics(baseline_slices[name], metrics),
                }
                for name, metrics in slices.items()
            },
        }
        summary_rows.append(row)
        per_variant_details[variant.name] = row

    diagnosis = _diagnose(summary_rows)
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": args.seed,
        "pairs_path": str(pairs_path.relative_to(REPO_ROOT)),
        "baseline": {
            "model_path": str(baseline_model_path.relative_to(REPO_ROOT)),
            "eval_path": str(baseline_eval_path.relative_to(REPO_ROOT)),
            "results": baseline_eval.get("results", {}),
            "focused_slices": baseline_slices,
        },
        "variants": summary_rows,
        "diagnosis": diagnosis,
    }
    summary_json = out_dir / "category_token_ablation_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))

    overall_rows = []
    for row in summary_rows:
        test = row["test"]
        delta = row["delta_vs_baseline"]
        overall_rows.append([
            row["name"],
            _fmt(_series_float(test.get("mae"))),
            _fmt(_series_float(delta.get("mae"))),
            _fmt(_series_float(test.get("dir_acc"))),
            _fmt(_series_float(delta.get("dir_acc"))),
            _fmt(_series_float(test.get("acc_3cls"))),
            _fmt(_series_float(delta.get("acc_3cls"))),
        ])

    slice_rows = []
    for row in summary_rows:
        slices = row["slices"]
        slice_rows.append([
            row["name"],
            _fmt(_series_float(slices["near_tie"]["metrics"].get("dir_acc"))),
            _fmt(_series_float(slices["near_tie"]["delta_vs_baseline"].get("dir_acc"))),
            _fmt(_series_float(slices["O2-O3"]["metrics"].get("dir_acc"))),
            _fmt(_series_float(slices["O2-O3"]["delta_vs_baseline"].get("dir_acc"))),
            _fmt(_series_float(slices["tie"]["metrics"].get("aux_tie_recall"))),
            _fmt(_series_float(slices["tie"]["delta_vs_baseline"].get("aux_tie_recall"))),
        ])

    diagnosis_text = {
        "leans_multi_token_cost": "Summary-only is the strongest ablation, so the current evidence leans toward extra-token cost rather than semantic grouping alone.",
        "grouping_matters_but_multi_token_still_costly": "Reduced semantic groupings recover part of the loss relative to full 6-way, so grouping matters, but extra tokens still appear costly overall.",
        "inconclusive_but_extra_tokens_not_helping": "No grouping recovered a clear gain over baseline; extra semantic tokens are still not helping under the current setup.",
    }.get(diagnosis, diagnosis)

    lines = [
        "# Category Token Ablation Summary",
        "",
        f"Generated: {summary_payload['generated_at']}",
        "",
        "## Setup",
        "",
        f"- pairs: {summary_payload['pairs_path']}",
        f"- seed: {summary_payload['seed']}",
        f"- baseline model: {summary_payload['baseline']['model_path']}",
        "- objective: separate token grouping effects from extra-token effects",
        "",
        "## Overall Test",
        "",
        *_report_table(["Variant", "MAE", "Delta", "dir_acc", "Delta", "acc_3cls", "Delta"], overall_rows),
        "",
        "## Focused Slices",
        "",
        *_report_table(["Variant", "near_dir", "Delta", "O2-O3 dir", "Delta", "tie_rec", "Delta"], slice_rows),
        "",
        "## Diagnosis",
        "",
        f"- {diagnosis_text}",
        "",
        "## Variants",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['name']}: {row['description']}")

    summary_md = out_dir / "category_token_ablation_summary.md"
    summary_md.write_text("\n".join(lines) + "\n")

    print(f"\n[ok] summary json: {summary_json}")
    print(f"[ok] summary markdown: {summary_md}")


if __name__ == "__main__":
    main()