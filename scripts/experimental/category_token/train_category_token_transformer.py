#!/usr/bin/env python3
"""
Minimal experimental branch for a category-token pair transformer.

This script is intentionally isolated from the main training entrypoint:
  - it reuses the existing pairs table and evaluation utilities;
  - it writes outputs into a separate directory;
  - it does not change the default single-token baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from feature_columns import DROPPED_INPUT_FEATURES, NON_TIME_COLS
from train_transformer import (
    CLASS_LABELS,
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
    predict_with_aux_np,
    select_device,
    split_by_program,
    train,
)

from category_schema import build_category_feature_map, build_category_index_map, schema_metadata


EXPERIMENT_CONFIGS: dict[str, dict[str, float | int]] = {
    "category_token_base": {
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


class CategoryTokenPairTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        category_index_map: OrderedDict[str, list[int]],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.category_names = list(category_index_map.keys())
        self.category_index_map = category_index_map
        self.tokens_per_side = 1 + len(self.category_names)

        self.summary_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.category_projs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(len(indices), d_model),
                    nn.LayerNorm(d_model),
                )
                for name, indices in category_index_map.items()
            }
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

        self.head = nn.Sequential(
            nn.Linear(3 * d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(3 * d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, len(CLASS_LABELS)),
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

    def _encode_side(self, x: torch.Tensor, side_id: int) -> torch.Tensor:
        tokens: list[torch.Tensor] = [self.summary_proj(x)]
        for name in self.category_names:
            indices = self.category_index_map[name]
            tokens.append(self.category_projs[name](x[:, indices]))

        side_tokens = torch.stack(tokens, dim=1)
        role_ids = torch.arange(self.tokens_per_side, device=x.device)
        side_tokens = side_tokens + self.token_role_emb(role_ids).unsqueeze(0)
        side_tokens = side_tokens + self.side_emb.weight[side_id].view(1, 1, -1)
        return side_tokens

    def _encode_pair(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        side_i = self._encode_side(x_i, side_id=0)
        side_j = self._encode_side(x_j, side_id=1)
        seq = torch.cat([side_i, side_j], dim=1)
        out = self.encoder(seq)

        out_i = out[:, 0, :]
        out_j = out[:, self.tokens_per_side, :]
        diff = out_i - out_j
        return torch.cat([out_i, out_j, diff], dim=-1)

    def forward_with_aux(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_repr = self._encode_pair(x_i, x_j)
        reg = self.head(pair_repr).squeeze(-1)
        cls = self.cls_head(pair_repr)
        return reg, cls

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        reg, _ = self.forward_with_aux(x_i, x_j)
        return reg


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    preset = EXPERIMENT_CONFIGS.get(pre_args.config, {}) if pre_args.config else {}
    if pre_args.config and pre_args.config not in EXPERIMENT_CONFIGS:
        sys.exit(
            "[error] unknown preset '"
            + str(pre_args.config)
            + "', available: "
            + ", ".join(EXPERIMENT_CONFIGS)
        )

    parser = argparse.ArgumentParser(
        description="Experimental category-token transformer training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--output", default="train_set/category_token_transformer")
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
    parser.add_argument(
        "--aux-class-lambda",
        type=float,
        default=DEFAULT_AUX_CLASS_LAMBDA,
        dest="aux_class_lambda",
    )
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=NEAR_TIE_THRESHOLD,
        dest="near_tie_threshold",
    )
    parser.add_argument(
        "--tie-reg-weight",
        type=float,
        default=DEFAULT_TIE_REG_WEIGHT,
        dest="tie_reg_weight",
    )
    parser.add_argument(
        "--near-tie-reg-weight",
        type=float,
        default=DEFAULT_NEAR_TIE_REG_WEIGHT,
        dest="near_tie_reg_weight",
    )
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=DEFAULT_CLASS_BALANCE_POWER,
        dest="class_balance_power",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, dest="noise_std")
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip", type=float, default=LOG_RATIO_CLIP)

    if preset:
        parser.set_defaults(**preset)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs_path = (REPO_ROOT / args.pairs).resolve()
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

    category_feature_map = build_category_feature_map(NON_TIME_COLS)
    category_index_map = build_category_index_map(NON_TIME_COLS)
    schema = schema_metadata(NON_TIME_COLS, include_summary_token=True)
    schema_path = out_dir / "category_token_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    sep = "=" * 68
    print(sep)
    print("Step 0: Category-token schema")
    print(sep)
    print(
        "  tokens/program = "
        f"{schema['n_tokens_per_program']}  "
        f"(summary + {schema['n_category_tokens']} semantic tokens)"
    )
    for token in schema["tokens"]:
        if token["token"] == "summary":
            print(f"  summary: {token['n_features']:2d} features (global projection)")
            continue
        print(f"  {token['token']:7s}: {token['n_features']:2d} features")

    print("\n" + sep)
    print("Step 1: Split train / val / test by program")
    print(sep)
    df_train, df_val, df_test = split_by_program(df, seed=args.seed)
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        dist = "  ".join(
            f"{label}={part['label_class'].value_counts().get(label, 0)}"
            for label in ("i_better", "tie", "j_better")
        )
        mag_bins = magnitude_bin_counts(
            part["log_ratio"].values.astype(np.float32),
            near_tie_threshold=args.near_tie_threshold,
        )
        mag_dist = "  ".join(f"{name_}={mag_bins.get(name_, 0)}" for name_ in mag_bins)
        print(
            f"  {name:5s}: {part['program'].nunique():3d} programs  {len(part):5d} pairs  "
            f"labels: {dist}  |  |log_ratio| bins: {mag_dist}"
        )

    print("\n" + sep)
    print("Step 2: Naive rank baseline")
    print(sep)
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        metrics = naive_rank_baseline(part)
        print(
            f"  {name:5s} | MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}  "
            f"dir_acc={metrics['dir_acc']:.4f}  acc_3cls={metrics['acc_3cls']:.3f}"
        )

    print("\n" + sep)
    print(
        "Step 3: Build CategoryTokenPairTransformer  "
        f"d_model={args.d_model}  nhead={args.nhead}  "
        f"nlayers={args.nlayers}  ffn={args.ffn_dim}"
    )
    print(sep)
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
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"  trainable params: {n_params:,}")
    print(f"  output dir: {out_dir}")
    print(f"  schema file: {schema_path.name}")
    print(
        "  HuberLoss(δ="
        f"{args.huber_delta})  dir_lambda={args.direction_lambda}  "
        f"aux_cls_lambda={args.aux_class_lambda}  near_tie<={args.near_tie_threshold}  "
        f"w_tie={args.tie_reg_weight}  w_near={args.near_tie_reg_weight}  "
        f"noise={args.noise_std}  AdamW lr={args.lr}  wd={args.wd}"
    )

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

    print("\n" + sep)
    print("Step 4: Evaluation")
    print(sep)
    header = (
        f"  {'split':5s} | {'MAE':>7} {'RMSE':>7} {'R2':>7} "
        f"{'dir_acc':>8} {'acc_3cls':>9} {'aux_3cls':>9} {'tie_rec':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[str, dict[str, int | float]] = {}
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        y_pred, cls_logits = predict_with_aux_np(model, part, device)
        y_true = part["log_ratio"].values.astype(np.float32)
        metrics = compute_metrics(y_true, y_pred)
        metrics.update(compute_aux_metrics(y_true, cls_logits))
        results[name] = metrics
        print(
            f"  {name:5s} | {metrics['mae']:7.4f} {metrics['rmse']:7.4f} {metrics['r2']:7.4f} "
            f"{metrics['dir_acc']:>8} {metrics['acc_3cls']:9.3f} "
            f"{metrics['aux_acc_3cls']:9.3f} {metrics['aux_tie_recall']:8.3f}"
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
            y_pred, cls_logits = predict_with_aux_np(model, subset, device)
            y_true = subset["log_ratio"].values.astype(np.float32)
            metrics = compute_metrics(y_true, y_pred)
            metrics.update(compute_aux_metrics(y_true, cls_logits))
            key = f"{variant_i}-{variant_j}"
            per_pair[key] = metrics
            print(
                f"  {key}: n={metrics['n']:3d}  dir_acc={metrics['dir_acc']}  "
                f"reg_acc_3cls={metrics['acc_3cls']:.3f}  aux_acc_3cls={metrics['aux_acc_3cls']:.3f}"
            )

    model_path = out_dir / "model_category_token_transformer.pt"
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
                "tokens_per_side": schema["n_tokens_per_program"],
            },
            "non_time_cols": NON_TIME_COLS,
            "category_feature_map": category_feature_map,
            "token_schema": schema,
        },
        model_path,
    )

    eval_result = {
        "model": "CategoryTokenPairTransformer",
        "config": args.config,
        "seed": args.seed,
        "pairs_path": str(pairs_path.relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "architecture": (
            f"[summary+{len(category_feature_map)} category tokens]x2->"
            f"TransformerEncoder({args.nlayers}L,{args.nhead}H,ffn={args.ffn_dim})->"
            "[summary_i;summary_j;summary_i-summary_j]->reg_head+cls_head"
        ),
        "source_isolation": {
            "baseline_script": "scripts/train_transformer.py",
            "experimental_script": "scripts/experimental/category_token/train_category_token_transformer.py",
            "category_schema": "scripts/experimental/category_token/category_schema.py",
            "overwrites_baseline_outputs": False,
        },
        "token_schema": schema,
        "n_params": n_params,
        "device": str(device),
        "log_ratio_clip": args.clip,
        "training_objective": {
            "regression_loss": "weighted_huber",
            "classification_loss": (
                "weighted_cross_entropy" if args.aux_class_lambda > 0.0 else "disabled"
            ),
            "direction_loss": (
                "binary_cross_entropy" if args.direction_lambda > 0.0 else "disabled"
            ),
            "loss_formula": "L = L_reg + aux_class_lambda * L_CE + direction_lambda * L_dir",
            "aux_class_enabled": bool(args.aux_class_lambda > 0.0),
            "aux_class_lambda": args.aux_class_lambda,
            "class_balance_power": args.class_balance_power,
            "class_weights": class_weight_dict(
                df_train["label_int"].values.astype(np.int64),
                power=args.class_balance_power,
            ),
            "final_train_loss": round(float(history["train_loss"][-1]), 6) if history["train_loss"] else None,
            "final_val_loss": round(float(history["val_loss"][-1]), 6) if history["val_loss"] else None,
            "final_val_reg_loss": round(float(history["val_reg_loss"][-1]), 6) if history["val_reg_loss"] else None,
            "final_val_aux_class_loss": (
                round(float(history["val_aux_class_loss"][-1]), 6)
                if history["val_aux_class_loss"]
                else None
            ),
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
            name: magnitude_bin_counts(
                part["log_ratio"].values.astype(np.float32),
                near_tie_threshold=args.near_tie_threshold,
            )
            for name, part in (("train", df_train), ("val", df_val), ("test", df_test))
        },
        "results": results,
        "per_pair": per_pair,
        "history": {key: [round(value, 6) for value in values] for key, values in history.items()},
    }
    eval_path = out_dir / "model_category_token_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    print(f"\n[ok] model saved: {model_path}")
    print(f"[ok] eval saved: {eval_path}")
    print(f"[ok] schema saved: {schema_path}")


if __name__ == "__main__":
    main()