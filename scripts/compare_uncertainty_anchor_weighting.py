#!/usr/bin/env python3
"""
compare_uncertainty_anchor_weighting.py — uncertainty-aware anchor weighting 扫描
================================================================================

回放多个 `score_program.py --uncertainty-weight-lambda X`，再用
`evaluate_score_vs_time.py` 做 strict time 外部验证，生成 JSON/Markdown 对比。
默认 lambda=0 作为 baseline；lambda>0 时使用 MC dropout 方差压低高不确定锚点权重。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 uncertainty-aware anchor weighting 并生成评分/时间对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambdas", default="0,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-calibration-blend", type=float, default=0.0)
    parser.add_argument("--output-dir", default="train_set/uncertainty_anchor_weighting")
    parser.add_argument("--summary-json", default="train_set/uncertainty_anchor_weighting_comparison.json")
    parser.add_argument("--markdown", default="train_set/uncertainty_anchor_weighting_comparison.md")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="若本轮 scores/eval/time_eval 已存在，则跳过回放，仅重建 summary")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: float) -> str:
    text = f"{value:.6g}".replace("-", "m").replace(".", "p")
    return text


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _fmt_lambda(value: float) -> str:
    return f"{float(value):.6g}"


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# uncertainty-aware anchor weighting comparison")
    lines.append("")
    lines.append(f"> generated_at: {payload['generated_at']}")
    lines.append(
        f"> samples={payload['samples']}, dropout={payload['dropout']}, "
        f"eps={payload['eps']}, seed={payload['seed']}, "
        f"pair_calibration_blend={payload['pair_calibration_blend']}"
    )
    lines.append("")
    lines.append("## summary")
    lines.append("")
    lines.append(
        "| lambda | proxy r | proxy MAE | proxy dir | proxy band | "
        "time r | time MAE | time dir | time band | repeat r |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["rows"]:
        lines.append(
            f"| {_fmt_lambda(row['lambda'])} | {_fmt(row['proxy_corr'])} | {_fmt(row['proxy_mae'])} | "
            f"{_fmt(row['proxy_dir_acc'])} | {_fmt(row['proxy_band_acc'])} | "
            f"{_fmt(row['time_corr'])} | {_fmt(row['time_mae'])} | {_fmt(row['time_dir_acc'])} | "
            f"{_fmt(row['time_band_acc'])} | {_fmt(row['repeat_corr'])} |"
        )
    lines.append("")
    best = payload.get("recommended") or {}
    if best:
        lines.append(
            f"recommended lambda: `{_fmt_lambda(best['lambda'])}` "
            f"(proxy r={_fmt(best['proxy_corr'])}, proxy dir={_fmt(best['proxy_dir_acc'])}, "
            f"time r={_fmt(best['time_corr'])}, "
            f"time MAE={_fmt(best['time_mae'])}, time band={_fmt(best['time_band_acc'])}, "
            f"repeat r={_fmt(best['repeat_corr'])})."
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((r for r in rows if abs(float(r["lambda"])) < 1e-12), rows[0])
    min_proxy = float(baseline["proxy_corr"]) - 0.005
    min_proxy_dir = float(baseline["proxy_dir_acc"]) - 0.001
    min_proxy_band = float(baseline["proxy_band_acc"]) - 0.01
    min_time_band = float(baseline["time_band_acc"]) - 0.005
    max_time_mae = float(baseline["time_mae"]) + 0.003
    candidates = [
        r for r in rows
        if (
            float(r["proxy_corr"]) >= min_proxy
            and float(r["proxy_dir_acc"]) >= min_proxy_dir
            and float(r["proxy_band_acc"]) >= min_proxy_band
            and float(r["time_band_acc"]) >= min_time_band
            and float(r["time_mae"]) <= max_time_mae
        )
    ]
    if not candidates:
        candidates = rows
    return max(
        candidates,
        key=lambda r: (
            -float(r["proxy_mae"]),
            float(r["time_corr"]),
            float(r.get("repeat_corr") or 0.0),
            float(r["proxy_band_acc"]),
        ),
    )


def main() -> None:
    args = parse_args()
    lambdas = [float(item.strip()) for item in args.lambdas.split(",") if item.strip()]
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for weight_lambda in lambdas:
        slug = _slug(weight_lambda)
        scores_path = output_dir / f"scores_lambda_{slug}.parquet"
        score_eval_path = output_dir / f"score_eval_lambda_{slug}.json"
        time_eval_path = output_dir / f"score_time_eval_lambda_{slug}.json"

        score_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "score_program.py"),
            "--pair-calibration-blend",
            f"{float(args.pair_calibration_blend):.6g}",
            "--uncertainty-samples",
            str(int(args.samples)),
            "--uncertainty-dropout",
            f"{float(args.dropout):.6g}",
            "--uncertainty-weight-lambda",
            f"{weight_lambda:.6g}",
            "--uncertainty-eps",
            f"{float(args.eps):.6g}",
            "--uncertainty-seed",
            str(int(args.seed)),
            "--output",
            str(scores_path.relative_to(REPO_ROOT)),
            "--eval-output",
            str(score_eval_path.relative_to(REPO_ROOT)),
        ]
        if args.device:
            score_cmd.extend(["--device", args.device])
        if args.reuse_existing and scores_path.exists() and score_eval_path.exists():
            print(f"[reuse] score outputs exist for lambda={weight_lambda:.6g}", flush=True)
        else:
            _run(score_cmd)

        time_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_score_vs_time.py"),
            "--scores",
            str(scores_path.relative_to(REPO_ROOT)),
            "--output",
            str(time_eval_path.relative_to(REPO_ROOT)),
        ]
        if args.reuse_existing and time_eval_path.exists():
            print(f"[reuse] time eval exists for lambda={weight_lambda:.6g}", flush=True)
        else:
            _run(time_cmd)

        score_eval = _load(score_eval_path)
        time_eval = _load(time_eval_path)
        repeat = time_eval.get("repeat_backed_only") or {}
        rows.append({
            "lambda": weight_lambda,
            "samples": int(args.samples) if weight_lambda > 0 else 0,
            "dropout": float(args.dropout) if weight_lambda > 0 else 0.0,
            "eps": float(args.eps) if weight_lambda > 0 else 0.0,
            "scores": str(scores_path.relative_to(REPO_ROOT)),
            "score_eval": str(score_eval_path.relative_to(REPO_ROOT)),
            "time_eval": str(time_eval_path.relative_to(REPO_ROOT)),
            "proxy_corr": score_eval.get("corr_score_log"),
            "proxy_mae": score_eval.get("mae_score_log"),
            "proxy_dir_acc": score_eval.get("dir_accuracy"),
            "proxy_band_acc": score_eval.get("band_accuracy"),
            "time_corr": time_eval.get("corr_model_time"),
            "time_mae": time_eval.get("mae_model_time"),
            "time_dir_acc": time_eval.get("dir_acc_model"),
            "time_band_acc": time_eval.get("band_acc_model"),
            "repeat_corr": repeat.get("corr_model_time"),
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "samples": int(args.samples),
        "dropout": float(args.dropout),
        "eps": float(args.eps),
        "seed": int(args.seed),
        "pair_calibration_blend": float(args.pair_calibration_blend),
        "recommendation_policy": (
            "keep proxy_corr within baseline-0.005, proxy_dir within baseline-0.001, "
            "proxy_band within baseline-0.01, "
            "time_band within baseline-0.005, time_mae within baseline+0.003; "
            "then prefer lower proxy_mae, higher time_corr, higher repeat_corr"
        ),
        "rows": rows,
        "recommended": _recommend(rows),
    }
    summary_path = (REPO_ROOT / args.summary_json).resolve()
    markdown_path = (REPO_ROOT / args.markdown).resolve()
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(markdown_path, payload)
    print(f"[done] summary: {summary_path}")
    print(f"[done] report:  {markdown_path}")


if __name__ == "__main__":
    main()
