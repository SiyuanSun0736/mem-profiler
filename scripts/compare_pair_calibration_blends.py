#!/usr/bin/env python3
"""
compare_pair_calibration_blends.py — per-pair calibration blend 扫描
===================================================================

回放多个 `score_program.py --pair-calibration-blend X`，再用
`evaluate_score_vs_time.py` 评估 strict time，生成 JSON/Markdown 对比。
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
        description="扫描 per-pair calibration blend 并生成评分/时间对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--blends", default="0,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--output-dir", default="train_set/pair_calibration_blends")
    parser.add_argument("--summary-json", default="train_set/pair_calibration_blend_comparison.json")
    parser.add_argument("--markdown", default="train_set/pair_calibration_blend_comparison.md")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# per-pair calibration blend comparison")
    lines.append("")
    lines.append(f"> generated_at: {payload['generated_at']}")
    lines.append("")
    lines.append("## summary")
    lines.append("")
    lines.append("| blend | proxy r | proxy MAE | proxy band | time r | time MAE | time band | repeat r |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["rows"]:
        lines.append(
            f"| {row['blend']:.2f} | {_fmt(row['proxy_corr'])} | {_fmt(row['proxy_mae'])} | "
            f"{_fmt(row['proxy_band_acc'])} | {_fmt(row['time_corr'])} | {_fmt(row['time_mae'])} | "
            f"{_fmt(row['time_band_acc'])} | {_fmt(row['repeat_corr'])} |"
        )
    lines.append("")
    best = payload.get("recommended") or {}
    if best:
        lines.append(
            f"recommended blend: `{best['blend']:.2f}` "
            f"(proxy r={_fmt(best['proxy_corr'])}, time r={_fmt(best['time_corr'])}, "
            f"repeat r={_fmt(best['repeat_corr'])})."
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((r for r in rows if abs(float(r["blend"])) < 1e-12), rows[0])
    min_proxy = float(baseline["proxy_corr"]) - 0.005
    min_band = float(baseline["proxy_band_acc"]) - 0.01
    candidates = [
        r for r in rows
        if float(r["proxy_corr"]) >= min_proxy and float(r["proxy_band_acc"]) >= min_band
    ]
    if not candidates:
        candidates = rows
    return max(
        candidates,
        key=lambda r: (
            float(r["time_corr"]),
            float(r.get("repeat_corr") or 0.0),
            -float(r["proxy_mae"]),
        ),
    )


def main() -> None:
    args = parse_args()
    blends = [float(item.strip()) for item in args.blends.split(",") if item.strip()]
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for blend in blends:
        slug = _slug(blend)
        scores_path = output_dir / f"scores_blend_{slug}.parquet"
        score_eval_path = output_dir / f"score_eval_blend_{slug}.json"
        time_eval_path = output_dir / f"score_time_eval_blend_{slug}.json"

        score_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "score_program.py"),
            "--pair-calibration-blend",
            f"{blend:.6g}",
            "--output",
            str(scores_path.relative_to(REPO_ROOT)),
            "--eval-output",
            str(score_eval_path.relative_to(REPO_ROOT)),
        ]
        if args.device:
            score_cmd.extend(["--device", args.device])
        _run(score_cmd)

        time_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_score_vs_time.py"),
            "--scores",
            str(scores_path.relative_to(REPO_ROOT)),
            "--output",
            str(time_eval_path.relative_to(REPO_ROOT)),
        ]
        _run(time_cmd)

        score_eval = _load(score_eval_path)
        time_eval = _load(time_eval_path)
        repeat = time_eval.get("repeat_backed_only") or {}
        rows.append({
            "blend": blend,
            "scores": str(scores_path.relative_to(REPO_ROOT)),
            "score_eval": str(score_eval_path.relative_to(REPO_ROOT)),
            "time_eval": str(time_eval_path.relative_to(REPO_ROOT)),
            "proxy_corr": score_eval.get("corr_score_log"),
            "proxy_mae": score_eval.get("mae_score_log"),
            "proxy_band_acc": score_eval.get("band_accuracy"),
            "time_corr": time_eval.get("corr_model_time"),
            "time_mae": time_eval.get("mae_model_time"),
            "time_band_acc": time_eval.get("band_acc_model"),
            "repeat_corr": repeat.get("corr_model_time"),
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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
