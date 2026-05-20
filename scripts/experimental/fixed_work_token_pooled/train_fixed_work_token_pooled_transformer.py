#!/usr/bin/env python3
"""
Fixed-work bucket token experiment with side pooling.

Hypothesis:
The current fixed-work branch already learns useful local direction signals, but its
head only consumes summary-token outputs. This pooled variant adds a mean-pooled
side representation over bucket tokens to test whether local evidence can improve
tie / near-tie calibration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
FIXED_WORK_ROOT = SCRIPTS_ROOT / "experimental" / "fixed_work_token"
for candidate in (SCRIPTS_ROOT, FIXED_WORK_ROOT):
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
    TIE_THRESHOLD,
    class_weight_dict,
    compute_aux_metrics,
    compute_metrics,
    magnitude_bin_counts,
    naive_rank_baseline,
    select_device,
    split_by_program,
)

from fixed_work_token_schema import (
    NUM_BUCKETS_DEFAULT,
    WORK_BUCKET_FEATURE_NAMES,
    apply_fixed_work_token_scaler,
    build_fixed_work_token_map,
    build_fixed_work_token_schema,
    dump_schema_json,
    fit_fixed_work_token_scaler,
)
from train_fixed_work_token_transformer import (
    _collect_run_keys,
    _load_pairs_table,
    _load_run_features,
    predict_with_aux_np_fixed_work,
    train_fixed_work_model,
)


EXPERIMENT_CONFIGS: dict[str, dict[str, float | int]] = {
    "fixed_work_bucket_pooled_base": {
        "num_buckets": NUM_BUCKETS_DEFAULT,
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


class FixedWorkBucketPooledTokenPairTransformer(nn.Module):
    def __init__(
        self,
        summary_dim: int,
        bucket_feat_dim: int,
        num_buckets: int = NUM_BUCKETS_DEFAULT,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.bucket_feat_dim = int(bucket_feat_dim)
        self.tokens_per_side = 1 + self.num_buckets

        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.bucket_proj = nn.Sequential(
            nn.Linear(bucket_feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.token_role_emb = nn.Embedding(self.tokens_per_side, d_model)
        self.side_emb = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # summary + pooled local state for each side, plus both diffs
        pair_dim = 6 * d_model
        self.head = nn.Sequential(
            nn.Linear(pair_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(pair_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 3),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _encode_side(self, summary: torch.Tensor, buckets: torch.Tensor, side_id: int) -> torch.Tensor:
        summary_tok = self.summary_proj(summary).unsqueeze(1)
        bucket_flat = buckets.reshape(-1, self.bucket_feat_dim)
        bucket_tok = self.bucket_proj(bucket_flat).reshape(summary.size(0), self.num_buckets, -1)
        seq = torch.cat([summary_tok, bucket_tok], dim=1)
        role_ids = torch.arange(self.tokens_per_side, device=summary.device)
        seq = seq + self.token_role_emb(role_ids).unsqueeze(0)
        seq = seq + self.side_emb.weight[side_id].view(1, 1, -1)
        return seq

    def _encode_pair(self, x_i: torch.Tensor, x_j: torch.Tensor, b_i: torch.Tensor, b_j: torch.Tensor) -> torch.Tensor:
        side_i = self._encode_side(x_i, b_i, side_id=0)
        side_j = self._encode_side(x_j, b_j, side_id=1)
        seq = torch.cat([side_i, side_j], dim=1)
        out = self.encoder(seq)

        side_i_out = out[:, : self.tokens_per_side, :]
        side_j_out = out[:, self.tokens_per_side :, :]

        summary_i = side_i_out[:, 0, :]
        summary_j = side_j_out[:, 0, :]
        pooled_i = side_i_out[:, 1:, :].mean(dim=1)
        pooled_j = side_j_out[:, 1:, :].mean(dim=1)

        summary_diff = summary_i - summary_j
        pooled_diff = pooled_i - pooled_j
        return torch.cat([summary_i, pooled_i, summary_j, pooled_j, summary_diff, pooled_diff], dim=-1)

    def forward_with_aux(self, x_i: torch.Tensor, x_j: torch.Tensor, b_i: torch.Tensor, b_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_repr = self._encode_pair(x_i, x_j, b_i, b_j)
        reg = self.head(pair_repr).squeeze(-1)
        cls = self.cls_head(pair_repr)
        return reg, cls

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, b_i: torch.Tensor, b_j: torch.Tensor) -> torch.Tensor:
        reg, _ = self.forward_with_aux(x_i, x_j, b_i, b_j)
        return reg


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    preset = EXPERIMENT_CONFIGS.get(pre_args.config, {}) if pre_args.config else {}
    if pre_args.config and pre_args.config not in EXPERIMENT_CONFIGS:
        sys.exit("[error] unknown preset '" + str(pre_args.config) + "', available: " + ", ".join(EXPERIMENT_CONFIGS))

    parser = argparse.ArgumentParser(
        description="Experimental fixed-work-bucket-token transformer with side pooling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--run-features", default="train_set/run_features.parquet")
    parser.add_argument("--output", default="train_set/fixed_work_token_pooled_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=NUM_BUCKETS_DEFAULT, dest="num_buckets")
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
    run_features_path = (REPO_ROOT / args.run_features).resolve()
    out_dir = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    df_pairs = _load_pairs_table(pairs_path)
    df_run = _load_run_features(run_features_path)
    if DROPPED_INPUT_FEATURES:
        print(f"[info] dropped dead input features: {', '.join(DROPPED_INPUT_FEATURES)}")

    xi_cols = [f"xi_{col}" for col in NON_TIME_COLS]
    xj_cols = [f"xj_{col}" for col in NON_TIME_COLS]
    missing = [col for col in xi_cols + xj_cols if col not in df_pairs.columns]
    if missing:
        sys.exit(f"[error] missing pair feature columns: {missing[:5]}...")

    sep = "=" * 72
    print(sep)
    print("Step 0: Split train / val / test by program")
    print(sep)
    df_train, df_val, df_test = split_by_program(df_pairs, seed=args.seed)
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

    all_keys = _collect_run_keys(df_pairs)
    train_keys = _collect_run_keys(df_train)
    run_key_df = df_run[
        [(str(program), str(variant)) in all_keys for program, variant in zip(df_run["program"], df_run["variant"])]
    ].copy()
    if len(run_key_df) != len(all_keys):
        sys.exit("[error] fixed-work pooled run mapping is incomplete for the active pairs table")

    schema = build_fixed_work_token_schema(num_buckets=args.num_buckets)
    schema["pooled_head"] = {
        "enabled": True,
        "description": "Head consumes summary token outputs plus side mean pooling over encoded bucket tokens.",
    }
    schema_path = out_dir / "fixed_work_token_pooled_schema.json"
    dump_schema_json(schema_path, schema)

    print("\n" + sep)
    print("Step 1: Build fixed-work token cache")
    print(sep)
    raw_token_map, token_summary = build_fixed_work_token_map(run_key_df, num_buckets=args.num_buckets)
    scaler = fit_fixed_work_token_scaler(raw_token_map, train_keys=train_keys)
    token_map = apply_fixed_work_token_scaler(raw_token_map, scaler)
    print(
        f"  runs={token_summary['n_runs']}  buckets={token_summary['num_buckets']}  bucket_features={len(WORK_BUCKET_FEATURE_NAMES)}  "
        f"window_count(median)={token_summary['window_count']['median']:.1f}  active_windows(median)={token_summary['active_window_count']['median']:.1f}"
    )
    print(
        f"  fitted fixed-work scaler on {scaler['n_tokens']} train bucket tokens  mean_assigned_instruction_share(mean)={token_summary['mean_assigned_instruction_share']['mean']:.4f}"
    )

    print("\n" + sep)
    print("Step 2: Naive rank baseline")
    print(sep)
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        metrics = naive_rank_baseline(part)
        print(
            f"  {name:5s} | MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}  dir_acc={metrics['dir_acc']:.4f}  acc_3cls={metrics['acc_3cls']:.3f}"
        )

    print("\n" + sep)
    print(
        "Step 3: Build FixedWorkBucketPooledTokenPairTransformer  "
        f"num_buckets={args.num_buckets}  d_model={args.d_model}  nhead={args.nhead}  nlayers={args.nlayers}  ffn={args.ffn_dim}"
    )
    print(sep)
    model = FixedWorkBucketPooledTokenPairTransformer(
        summary_dim=len(NON_TIME_COLS),
        bucket_feat_dim=len(WORK_BUCKET_FEATURE_NAMES),
        num_buckets=args.num_buckets,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.nlayers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"  trainable params: {n_params:,}")
    print(f"  output dir: {out_dir}")
    print(f"  schema file: {schema_path.name}")
    print(
        f"  HuberLoss(delta={args.huber_delta})  dir_lambda={args.direction_lambda}  aux_cls_lambda={args.aux_class_lambda}  "
        f"near_tie<={args.near_tie_threshold}  w_tie={args.tie_reg_weight}  w_near={args.near_tie_reg_weight}  noise={args.noise_std}  "
        f"AdamW lr={args.lr}  wd={args.wd}"
    )

    history = train_fixed_work_model(
        model=model,
        device=device,
        df_train=df_train,
        df_val=df_val,
        token_map=token_map,
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

    print("\n" + sep)
    print("Step 4: Evaluation")
    print(sep)
    header = (
        f"  {'split':5s} | {'MAE':>7} {'RMSE':>7} {'R2':>7} {'dir_acc':>8} {'acc_3cls':>9} {'aux_3cls':>9} {'tie_rec':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[str, dict[str, int | float]] = {}
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        y_pred, cls_logits = predict_with_aux_np_fixed_work(model, part, token_map, device)
        y_true = part["log_ratio"].values.astype(np.float32)
        metrics = compute_metrics(y_true, y_pred)
        metrics.update(compute_aux_metrics(y_true, cls_logits))
        results[name] = metrics
        print(
            f"  {name:5s} | {metrics['mae']:7.4f} {metrics['rmse']:7.4f} {metrics['r2']:7.4f} {metrics['dir_acc']:>8} {metrics['acc_3cls']:9.3f} {metrics['aux_acc_3cls']:9.3f} {metrics['aux_tie_recall']:8.3f}"
        )

    print("\n[info] test split by variant pair:")
    per_pair: dict[str, dict[str, int | float]] = {}
    for variant_i in ("O0", "O1", "O2"):
        for variant_j in ("O1", "O2", "O3"):
            if variant_i >= variant_j:
                continue
            mask = (df_test["variant_i"] == variant_i) & (df_test["variant_j"] == variant_j)
            if mask.sum() == 0:
                continue
            subset = df_test[mask]
            y_pred, cls_logits = predict_with_aux_np_fixed_work(model, subset, token_map, device)
            y_true = subset["log_ratio"].values.astype(np.float32)
            metrics = compute_metrics(y_true, y_pred)
            metrics.update(compute_aux_metrics(y_true, cls_logits))
            key = f"{variant_i}-{variant_j}"
            per_pair[key] = metrics
            print(
                f"  {key}: n={metrics['n']:3d}  dir_acc={metrics['dir_acc']}  reg_acc_3cls={metrics['acc_3cls']:.3f}  aux_acc_3cls={metrics['aux_acc_3cls']:.3f}"
            )

    model_path = out_dir / "model_fixed_work_token_pooled_transformer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hparams": {
                "summary_dim": len(NON_TIME_COLS),
                "bucket_feat_dim": len(WORK_BUCKET_FEATURE_NAMES),
                "num_buckets": args.num_buckets,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "nlayers": args.nlayers,
                "num_layers": args.nlayers,
                "ffn_dim": args.ffn_dim,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
                "head_hidden": args.head_hidden,
                "pooled_head": True,
            },
            "non_time_cols": NON_TIME_COLS,
            "bucket_feature_names": list(WORK_BUCKET_FEATURE_NAMES),
            "fixed_work_schema": schema,
            "fixed_work_scaler": scaler,
        },
        model_path,
    )

    eval_result = {
        "model": "FixedWorkBucketPooledTokenPairTransformer",
        "config": args.config,
        "seed": args.seed,
        "pairs_path": str(pairs_path.relative_to(REPO_ROOT)),
        "run_features_path": str(run_features_path.relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "architecture": (
            f"[summary+{args.num_buckets} fixed_work_bucket tokens]x2->"
            f"TransformerEncoder({args.nlayers}L,{args.nhead}H,ffn={args.ffn_dim})->"
            "[summary_i;pool_i;summary_j;pool_j;summary_diff;pool_diff]->reg_head+cls_head"
        ),
        "source_isolation": {
            "baseline_script": "scripts/train_transformer.py",
            "experimental_script": "scripts/experimental/fixed_work_token_pooled/train_fixed_work_token_pooled_transformer.py",
            "schema_script": "scripts/experimental/fixed_work_token/fixed_work_token_schema.py",
            "overwrites_baseline_outputs": False,
        },
        "token_schema": schema,
        "token_cache_summary": token_summary,
        "token_scaler": scaler,
        "n_params": n_params,
        "device": str(device),
        "log_ratio_clip": args.clip,
        "training_objective": {
            "regression_loss": "weighted_huber",
            "classification_loss": "weighted_cross_entropy" if args.aux_class_lambda > 0.0 else "disabled",
            "direction_loss": "disabled" if args.direction_lambda <= 0.0 else "binary_cross_entropy",
            "loss_formula": "L = L_reg + aux_class_lambda * L_CE + direction_lambda * L_dir",
            "aux_class_enabled": bool(args.aux_class_lambda > 0.0),
            "aux_class_lambda": args.aux_class_lambda,
            "class_balance_power": args.class_balance_power,
            "class_weights": class_weight_dict(df_train["label_int"].values.astype(np.int64), power=args.class_balance_power),
            "final_train_loss": round(float(history["train_loss"][-1]), 6) if history["train_loss"] else None,
            "final_val_loss": round(float(history["val_loss"][-1]), 6) if history["val_loss"] else None,
            "final_val_reg_loss": round(float(history["val_reg_loss"][-1]), 6) if history["val_reg_loss"] else None,
            "final_val_aux_class_loss": round(float(history["val_aux_class_loss"][-1]), 6) if history["val_aux_class_loss"] else None,
        },
        "tie_strategy": {
            "tie_threshold": TIE_THRESHOLD,
            "near_tie_threshold": args.near_tie_threshold,
            "tie_reg_weight": args.tie_reg_weight,
            "near_tie_reg_weight": args.near_tie_reg_weight,
            "aux_class_lambda": args.aux_class_lambda,
            "class_balance_power": args.class_balance_power,
            "direction_lambda": args.direction_lambda,
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
            name: magnitude_bin_counts(part["log_ratio"].values.astype(np.float32), near_tie_threshold=args.near_tie_threshold)
            for name, part in (("train", df_train), ("val", df_val), ("test", df_test))
        },
        "results": results,
        "per_pair": per_pair,
        "history": {key: [round(value, 6) for value in values] for key, values in history.items()},
    }
    eval_path = out_dir / "model_fixed_work_token_pooled_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    print(f"\n[ok] model saved: {model_path}")
    print(f"[ok] eval saved: {eval_path}")
    print(f"[ok] schema saved: {schema_path}")


if __name__ == "__main__":
    main()