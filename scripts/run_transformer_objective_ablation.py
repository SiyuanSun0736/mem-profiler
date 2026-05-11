#!/usr/bin/env python3
"""
run_transformer_objective_ablation.py — Compare PairTransformer objectives.

Runs a small grid over the auxiliary three-class cross entropy weight and
summarizes whether CE improves direction/tie behavior without replacing the
continuous log-ratio regression head.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PairTransformer objective ablation for auxiliary CE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", help="Device passed to train_transformer.py")
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--output-dir", default="train_set/objective_ablation")
    parser.add_argument("--summary-json", default="train_set/transformer_objective_ablation.json")
    parser.add_argument("--markdown", default="train_set/objective_ablation.md")
    parser.add_argument("--config", default="fixed_work_transformer")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aux-class-lambdas",
        default="0.0,0.05,0.1,0.2,0.3",
        help="Comma separated CE weights",
    )
    parser.add_argument(
        "--direction-lambdas",
        default="0.0",
        help="Comma separated direction BCE weights",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy best model/eval to train_set/model_transformer.*",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("empty float list")
    return values


def run_command(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def metric(eval_data: dict[str, Any], split: str, name: str) -> float | None:
    return finite((eval_data.get("results", {}).get(split, {}) or {}).get(name))


def per_pair_metric(eval_data: dict[str, Any], pair: str, name: str) -> float | None:
    return finite((eval_data.get("per_pair", {}).get(pair, {}) or {}).get(name))


def score_trial(eval_data: dict[str, Any]) -> float:
    test = eval_data.get("results", {}).get("test", {}) or {}
    o23 = eval_data.get("per_pair", {}).get("O2-O3", {}) or {}
    val = eval_data.get("results", {}).get("val", {}) or {}

    score = 0.0
    score += 2.0 * float(test.get("aux_acc_3cls", 0.0) or 0.0)
    score += 1.5 * float(test.get("acc_3cls", 0.0) or 0.0)
    score += 1.0 * float(o23.get("aux_acc_3cls", 0.0) or 0.0)
    score += 0.8 * float(o23.get("aux_tie_recall", 0.0) or 0.0)
    score += 0.5 * float(test.get("dir_acc", 0.0) or 0.0)
    score -= 0.2 * float(test.get("mae", 0.0) or 0.0)
    score -= 0.1 * float(val.get("mae", 0.0) or 0.0)
    return score


def fmt(value: Any, digits: int = 4) -> str:
    value_f = finite(value)
    if value_f is None:
        return "-"
    return f"{value_f:.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    rows = summary["trials"]
    best = summary["best_trial"]
    lines: list[str] = [
        "# Transformer objective ablation",
        "",
        f"> Generated: {summary['generated_at']}",
        "",
        "## Conclusion",
        "",
        (
            "Best objective: "
            f"`{best['objective_name']}` "
            f"with `aux_class_lambda={best['aux_class_lambda']}` and "
            f"`direction_lambda={best['direction_lambda']}`."
        ),
        "",
        "The regression head remains the primary output for continuous log-ratio scoring. "
        "The auxiliary CE head is evaluated by `aux_acc_3cls`, `aux_tie_recall`, and hard-pair behavior.",
        "",
        "## Trial table",
        "",
        "| objective | aux λ | dir λ | test MAE | test R2 | test dir | test 3cls | aux 3cls | aux tie recall | O2-O3 3cls | O2-O3 aux | O2-O3 tie recall | score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {objective_name} | {aux:.2f} | {direction:.2f} | {mae} | {r2} | {dir_acc} | "
            "{acc_3cls} | {aux_acc} | {tie_recall} | {o23_acc} | {o23_aux} | {o23_tie} | {score} |".format(
                objective_name=row["objective_name"],
                aux=float(row["aux_class_lambda"]),
                direction=float(row["direction_lambda"]),
                mae=fmt(row["test_mae"]),
                r2=fmt(row["test_r2"]),
                dir_acc=fmt(row["test_dir_acc"]),
                acc_3cls=fmt(row["test_acc_3cls"]),
                aux_acc=fmt(row["test_aux_acc_3cls"]),
                tie_recall=fmt(row["test_aux_tie_recall"]),
                o23_acc=fmt(row["o2_o3_acc_3cls"]),
                o23_aux=fmt(row["o2_o3_aux_acc_3cls"]),
                o23_tie=fmt(row["o2_o3_aux_tie_recall"]),
                score=fmt(row["selection_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The ranking favors higher auxiliary three-class accuracy, regression-derived three-class accuracy, "
            "O2-O3 auxiliary accuracy, O2-O3 tie recall, and direction accuracy, with a small penalty for MAE. "
            "This is a model-selection aid; final single-program scoring must still be checked against proxy and time scores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    aux_lambdas = parse_float_list(args.aux_class_lambdas)
    direction_lambdas = parse_float_list(args.direction_lambdas)

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    for aux_lambda in aux_lambdas:
        for direction_lambda in direction_lambdas:
            objective_name = "reg_only" if aux_lambda == 0.0 and direction_lambda == 0.0 else "reg_ce"
            if direction_lambda > 0.0:
                objective_name += "_dir"
            trial_name = f"aux_{aux_lambda:.2f}_dir_{direction_lambda:.2f}".replace(".", "p")
            trial_dir = output_dir / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "scripts/train_transformer.py",
                "--config",
                args.config,
                "--pairs",
                args.pairs,
                "--output",
                display_path(trial_dir),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--aux-class-lambda",
                str(aux_lambda),
                "--direction-lambda",
                str(direction_lambda),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            run_command(cmd)

            eval_path = trial_dir / "model_transformer_eval.json"
            eval_data = load_json(eval_path)
            trial = {
                "objective_name": objective_name,
                "trial_name": trial_name,
                "output_dir": display_path(trial_dir),
                "eval_path": display_path(eval_path),
                "model_path": display_path(trial_dir / "model_transformer.pt"),
                "aux_class_lambda": aux_lambda,
                "direction_lambda": direction_lambda,
                "test_mae": metric(eval_data, "test", "mae"),
                "test_rmse": metric(eval_data, "test", "rmse"),
                "test_r2": metric(eval_data, "test", "r2"),
                "test_dir_acc": metric(eval_data, "test", "dir_acc"),
                "test_acc_3cls": metric(eval_data, "test", "acc_3cls"),
                "test_aux_acc_3cls": metric(eval_data, "test", "aux_acc_3cls"),
                "test_aux_tie_recall": metric(eval_data, "test", "aux_tie_recall"),
                "val_mae": metric(eval_data, "val", "mae"),
                "val_r2": metric(eval_data, "val", "r2"),
                "val_aux_acc_3cls": metric(eval_data, "val", "aux_acc_3cls"),
                "o2_o3_acc_3cls": per_pair_metric(eval_data, "O2-O3", "acc_3cls"),
                "o2_o3_aux_acc_3cls": per_pair_metric(eval_data, "O2-O3", "aux_acc_3cls"),
                "o2_o3_aux_tie_recall": per_pair_metric(eval_data, "O2-O3", "aux_tie_recall"),
                "training_objective": eval_data.get("training_objective", {}),
                "tie_strategy": eval_data.get("tie_strategy", {}),
            }
            trial["selection_score"] = round(score_trial(eval_data), 6)
            trials.append(trial)

    best_trial = max(trials, key=lambda row: float(row["selection_score"]))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "pairs": args.pairs,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": args.device,
        "trials": trials,
        "best_trial": best_trial,
    }

    summary_path = (REPO_ROOT / args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = (REPO_ROOT / args.markdown).resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")

    if args.promote_best:
        best_eval = pathlib.Path(best_trial["eval_path"])
        best_model = pathlib.Path(best_trial["model_path"])
        if not best_eval.is_absolute():
            best_eval = REPO_ROOT / best_eval
        if not best_model.is_absolute():
            best_model = REPO_ROOT / best_model
        shutil.copy2(best_eval, REPO_ROOT / "train_set" / "model_transformer_eval.json")
        shutil.copy2(best_model, REPO_ROOT / "train_set" / "model_transformer.pt")

    print(f"[ok] summary: {summary_path}")
    print(f"[ok] markdown: {markdown_path}")
    print(
        "[ok] best: "
        f"{best_trial['trial_name']} "
        f"aux={best_trial['aux_class_lambda']} "
        f"dir={best_trial['direction_lambda']} "
        f"score={best_trial['selection_score']}"
    )


if __name__ == "__main__":
    main()
