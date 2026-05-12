#!/usr/bin/env python3
"""
fit_pair_calibration.py — PairTransformer 输出的 per-pair 线性校准
==================================================================

目标
----
不重训 backbone，只在模型原始 pairwise log-ratio 输出后拟合轻量校准：

    y' = a_pair * y_hat + b_pair

默认使用 slope-only 模式，也就是 `b_pair = 0`，避免把 O0 基线整体推离
0 点。需要探索截距时可传 `--mode affine`。

校准按 unordered variant pair 拟合，例如 O0-O2 与 O2-O0 共享同一组参数，
但反向样本会先翻转符号再进入拟合，推理时再翻回原方向。这样可以保持
pairwise 方向的基本反对称结构。

输出
----
  train_set/pair_calibration.json
  train_set/pair_calibration_report.md

用法
----
  python scripts/fit_pair_calibration.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from score_program import load_model  # noqa: E402
from train_transformer import compute_metrics, predict_with_aux_np, select_device, split_by_program  # noqa: E402


VARIANT_RANK = {"O0": 0, "O1": 1, "O2": 2, "O3": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 variant pair 拟合 PairTransformer raw log-ratio 线性校准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="train_set/model_transformer.pt")
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--output", default="train_set/pair_calibration.json")
    parser.add_argument("--report", default="train_set/pair_calibration_report.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fit-split",
        choices=["train", "val", "train-val", "all"],
        default="val",
        help="用于拟合校准器的数据拆分",
    )
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["slope", "affine"],
        default="slope",
        help="校准函数形式；slope 保持 0 点不漂移，affine 允许截距",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=8.0,
        help="向 identity 校准 a=1,b=0 收缩的岭先验强度",
    )
    return parser.parse_args()


def _canonical_pair(variant_i: str, variant_j: str) -> tuple[str, int]:
    rank_i = VARIANT_RANK.get(str(variant_i), 999)
    rank_j = VARIANT_RANK.get(str(variant_j), 999)
    if (rank_i, str(variant_i)) <= (rank_j, str(variant_j)):
        return f"{variant_i}-{variant_j}", 1
    return f"{variant_j}-{variant_i}", -1


def _with_canonical_columns(df: pd.DataFrame, pred_col: str = "pred_raw") -> pd.DataFrame:
    out = df.copy()
    keys: list[str] = []
    signs: list[int] = []
    for row in out.itertuples(index=False):
        key, sign = _canonical_pair(str(row.variant_i), str(row.variant_j))
        keys.append(key)
        signs.append(sign)
    out["pair_key"] = keys
    out["pair_sign"] = signs
    out["y_canon"] = out["pair_sign"].astype(float) * out["log_ratio"].astype(float)
    out["pred_canon"] = out["pair_sign"].astype(float) * out[pred_col].astype(float)
    return out


def _fit_affine_identity_prior(
    x: np.ndarray,
    y: np.ndarray,
    prior_strength: float,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.allclose(x, x[0]):
        return 1.0, 0.0

    design = np.column_stack([x, np.ones_like(x)])
    target = y
    if prior_strength > 0:
        root = math.sqrt(float(prior_strength))
        design = np.vstack([
            design,
            np.array([[root, 0.0], [0.0, root]], dtype=np.float64),
        ])
        target = np.concatenate([target, np.array([root, 0.0], dtype=np.float64)])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b = float(params[0]), float(params[1])
    if not math.isfinite(a) or not math.isfinite(b):
        return 1.0, 0.0
    return a, b


def _fit_slope_identity_prior(
    x: np.ndarray,
    y: np.ndarray,
    prior_strength: float,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return 1.0, 0.0
    prior = max(float(prior_strength), 0.0)
    denom = float(np.dot(x, x) + prior)
    if denom <= 1e-12:
        return 1.0, 0.0
    # Ridge-like shrinkage toward identity: minimize SSE + prior * (a - 1)^2.
    a = float((np.dot(x, y) + prior) / denom)
    if not math.isfinite(a):
        return 1.0, 0.0
    return a, 0.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _apply_calibrators(df: pd.DataFrame, calibrators: dict[str, dict[str, Any]]) -> np.ndarray:
    out: list[float] = []
    for row in df.itertuples(index=False):
        payload = calibrators.get(str(row.pair_key), {})
        a = float(payload.get("a", 1.0))
        b = float(payload.get("b", 0.0))
        pred_canon = float(row.pred_canon)
        calibrated = a * pred_canon + b
        out.append(float(row.pair_sign) * calibrated)
    return np.asarray(out, dtype=np.float64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = compute_metrics(np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32))
    metrics["pearson"] = round(_safe_corr(np.asarray(y_true), np.asarray(y_pred)), 4)
    return metrics


def _evaluate_split(df: pd.DataFrame, calibrators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    y_true = df["log_ratio"].to_numpy(dtype=np.float64)
    raw = df["pred_raw"].to_numpy(dtype=np.float64)
    calibrated = _apply_calibrators(df, calibrators)
    return {
        "n": int(len(df)),
        "raw": _metrics(y_true, raw),
        "calibrated": _metrics(y_true, calibrated),
        "mae_delta": round(_mae(y_true, calibrated) - _mae(y_true, raw), 6),
        "pearson_delta": round(_safe_corr(y_true, calibrated) - _safe_corr(y_true, raw), 6),
    }


def _write_report(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# per-pair calibration report")
    lines.append("")
    lines.append(f"> generated_at: {payload['generated_at']}")
    lines.append(
        f"> fit_split: `{payload['fit_split']}`, mode: `{payload['mode']}`, "
        f"prior_strength: `{payload['prior_strength']}`"
    )
    lines.append("")
    lines.append("## split metrics")
    lines.append("")
    lines.append("| split | n | raw MAE | cal MAE | delta MAE | raw r | cal r | delta r | raw dir | cal dir |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in payload["split_metrics"].items():
        raw = metrics["raw"]
        cal = metrics["calibrated"]
        lines.append(
            f"| {name} | {metrics['n']} | {raw['mae']:.4f} | {cal['mae']:.4f} | "
            f"{metrics['mae_delta']:+.4f} | {raw['pearson']:.4f} | {cal['pearson']:.4f} | "
            f"{metrics['pearson_delta']:+.4f} | {raw['dir_acc']:.4f} | {cal['dir_acc']:.4f} |"
        )
    lines.append("")
    lines.append("## calibrators")
    lines.append("")
    lines.append("| pair | enabled | n | a | b | fit raw MAE | fit cal MAE | fit raw r | fit cal r |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, cal in sorted(payload["calibrators"].items()):
        lines.append(
            f"| {key} | {'yes' if cal['enabled'] else 'no'} | {cal['n']} | "
            f"{cal['a']:.4f} | {cal['b']:.4f} | "
            f"{cal['fit_raw_mae']:.4f} | {cal['fit_calibrated_mae']:.4f} | "
            f"{cal['fit_raw_pearson']:.4f} | {cal['fit_calibrated_pearson']:.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = (REPO_ROOT / args.model).resolve()
    pairs_path = (REPO_ROOT / args.pairs).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    report_path = (REPO_ROOT / args.report).resolve()

    for path in (model_path, pairs_path):
        if not path.exists():
            sys.exit(f"[error] input not found: {path}")

    device = select_device(args.device)
    model = load_model(model_path, device)
    df = pd.read_parquet(pairs_path)
    df_train, df_val, df_test = split_by_program(df, seed=args.seed)

    split_frames = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }
    scored_frames: dict[str, pd.DataFrame] = {}
    for name, part in split_frames.items():
        pred, _ = predict_with_aux_np(model, part, device)
        scored = part.copy()
        scored["pred_raw"] = pred.astype(np.float64)
        scored_frames[name] = _with_canonical_columns(scored)

    if args.fit_split == "train":
        fit_df = scored_frames["train"]
    elif args.fit_split == "val":
        fit_df = scored_frames["val"]
    elif args.fit_split == "train-val":
        fit_df = pd.concat([scored_frames["train"], scored_frames["val"]], ignore_index=True)
    else:
        fit_df = pd.concat(list(scored_frames.values()), ignore_index=True)

    calibrators: dict[str, dict[str, Any]] = {}
    for key, sub in fit_df.groupby("pair_key", sort=True):
        n = int(len(sub))
        enabled = n >= args.min_samples
        if enabled:
            fit_fn = _fit_slope_identity_prior if args.mode == "slope" else _fit_affine_identity_prior
            a, b = fit_fn(
                sub["pred_canon"].to_numpy(dtype=np.float64),
                sub["y_canon"].to_numpy(dtype=np.float64),
                prior_strength=args.prior_strength,
            )
        else:
            a, b = 1.0, 0.0
        y_fit = sub["log_ratio"].to_numpy(dtype=np.float64)
        raw_fit = sub["pred_raw"].to_numpy(dtype=np.float64)
        cal_fit = _apply_calibrators(_with_canonical_columns(sub, pred_col="pred_raw"), {key: {"a": a, "b": b}})
        calibrators[str(key)] = {
            "enabled": bool(enabled),
            "n": n,
            "a": round(float(a), 8),
            "b": round(float(b), 8),
            "fit_raw_mae": round(_mae(y_fit, raw_fit), 6),
            "fit_calibrated_mae": round(_mae(y_fit, cal_fit), 6),
            "fit_raw_pearson": round(_safe_corr(y_fit, raw_fit), 6),
            "fit_calibrated_pearson": round(_safe_corr(y_fit, cal_fit), 6),
        }

    split_metrics = {
        name: _evaluate_split(frame, calibrators)
        for name, frame in scored_frames.items()
    }
    split_metrics["fit"] = _evaluate_split(fit_df, calibrators)

    payload: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model": str(model_path.relative_to(REPO_ROOT)),
        "pairs": str(pairs_path.relative_to(REPO_ROOT)),
        "seed": int(args.seed),
        "fit_split": args.fit_split,
        "mode": args.mode,
        "min_samples": int(args.min_samples),
        "prior_strength": float(args.prior_strength),
        "orientation": "unordered_pair_with_sign_flip",
        "calibrators": calibrators,
        "split_metrics": split_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(report_path, payload)

    print(f"[ok] calibration JSON: {output_path}")
    print(f"[ok] report:           {report_path}")
    for name in ("fit", "test"):
        metrics = split_metrics[name]
        print(
            f"[{name}] raw_mae={metrics['raw']['mae']:.4f} "
            f"cal_mae={metrics['calibrated']['mae']:.4f} "
            f"raw_r={metrics['raw']['pearson']:.4f} "
            f"cal_r={metrics['calibrated']['pearson']:.4f}"
        )


if __name__ == "__main__":
    main()
