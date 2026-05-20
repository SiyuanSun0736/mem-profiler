#!/usr/bin/env python3
"""
Hybrid fixed-work + hotspot token experiment.

Each run is represented by one shared summary token, plus two local token families:
fixed-work buckets and hotspot windows. Both families are fed into the same pair
encoder so the summary token can absorb complementary local evidence without
introducing an additional pooling head.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
FIXED_WORK_ROOT = SCRIPTS_ROOT / "experimental" / "fixed_work_token"
HOTSPOT_ROOT = SCRIPTS_ROOT / "experimental" / "hotspot_token"
for candidate in (SCRIPTS_ROOT, FIXED_WORK_ROOT, HOTSPOT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
    balanced_class_weights,
    class_weight_dict,
    compute_aux_metrics,
    compute_metrics,
    direction_bce_loss,
    magnitude_bin_counts,
    naive_rank_baseline,
    regression_sample_weights,
    select_device,
    split_by_program,
    to_3class,
    weighted_mean,
)
from fixed_work_token_schema import (
    NUM_BUCKETS_DEFAULT,
    WORK_BUCKET_FEATURE_NAMES,
    apply_fixed_work_token_scaler,
    build_fixed_work_token_map,
    build_fixed_work_token_schema,
    fit_fixed_work_token_scaler,
)
from hotspot_token_schema import (
    TOP_K_DEFAULT,
    WINDOW_FEATURE_NAMES,
    apply_hotspot_token_scaler,
    build_hotspot_token_map,
    build_hotspot_token_schema,
    fit_hotspot_token_scaler,
)


EXPERIMENT_CONFIGS: dict[str, dict[str, float | int]] = {
    "fixed_work_hotspot_token_base": {
        "num_buckets": NUM_BUCKETS_DEFAULT,
        "top_k": TOP_K_DEFAULT,
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


def _load_run_features(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] missing run_features table: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    sys.exit(f"[error] unsupported run_features table format: {path.suffix}")


def _collect_run_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    keys_i = {(str(program), str(variant)) for program, variant in zip(df["program"], df["variant_i"])}
    keys_j = {(str(program), str(variant)) for program, variant in zip(df["program"], df["variant_j"])}
    return keys_i | keys_j


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_fixed_work_hotspot_schema(num_buckets: int, top_k: int) -> dict[str, Any]:
    return {
        "token_families": {
            "summary": {
                "token": "summary",
                "description": "Shared global run-level summary token built from z-score run features.",
            },
            "fixed_work": build_fixed_work_token_schema(num_buckets=num_buckets),
            "hotspot": build_hotspot_token_schema(top_k=top_k),
        },
        "tokens_per_program": int(1 + num_buckets + top_k),
        "integration": {
            "encoder": "shared TransformerEncoder over summary + fixed-work buckets + hotspot windows",
            "pair_head": "summary-only head to isolate token-family mixing from extra pooling changes",
        },
    }


class FixedWorkHotspotTokenPairTransformer(nn.Module):
    def __init__(
        self,
        summary_dim: int,
        bucket_feat_dim: int,
        window_feat_dim: int,
        num_buckets: int = NUM_BUCKETS_DEFAULT,
        top_k: int = TOP_K_DEFAULT,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.top_k = int(top_k)
        self.bucket_feat_dim = int(bucket_feat_dim)
        self.window_feat_dim = int(window_feat_dim)
        self.tokens_per_side = 1 + self.num_buckets + self.top_k

        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.bucket_proj = nn.Sequential(
            nn.Linear(bucket_feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.window_proj = nn.Sequential(
            nn.Linear(window_feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.token_family_emb = nn.Embedding(3, d_model)
        self.bucket_role_emb = nn.Embedding(self.num_buckets, d_model)
        self.window_role_emb = nn.Embedding(self.top_k, d_model)
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

    def _encode_side(
        self,
        summary: torch.Tensor,
        fixed_work_tokens: torch.Tensor,
        hotspot_tokens: torch.Tensor,
        side_id: int,
    ) -> torch.Tensor:
        summary_tok = self.summary_proj(summary).unsqueeze(1)
        summary_tok = summary_tok + self.token_family_emb.weight[0].view(1, 1, -1)

        bucket_flat = fixed_work_tokens.reshape(-1, self.bucket_feat_dim)
        bucket_tok = self.bucket_proj(bucket_flat).reshape(summary.size(0), self.num_buckets, -1)
        bucket_ids = torch.arange(self.num_buckets, device=summary.device)
        bucket_tok = bucket_tok + self.bucket_role_emb(bucket_ids).unsqueeze(0)
        bucket_tok = bucket_tok + self.token_family_emb.weight[1].view(1, 1, -1)

        window_flat = hotspot_tokens.reshape(-1, self.window_feat_dim)
        window_tok = self.window_proj(window_flat).reshape(summary.size(0), self.top_k, -1)
        window_ids = torch.arange(self.top_k, device=summary.device)
        window_tok = window_tok + self.window_role_emb(window_ids).unsqueeze(0)
        window_tok = window_tok + self.token_family_emb.weight[2].view(1, 1, -1)

        seq = torch.cat([summary_tok, bucket_tok, window_tok], dim=1)
        seq = seq + self.side_emb.weight[side_id].view(1, 1, -1)
        return seq

    def _encode_pair(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        b_i: torch.Tensor,
        b_j: torch.Tensor,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
    ) -> torch.Tensor:
        side_i = self._encode_side(x_i, b_i, h_i, side_id=0)
        side_j = self._encode_side(x_j, b_j, h_j, side_id=1)
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
        b_i: torch.Tensor,
        b_j: torch.Tensor,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_repr = self._encode_pair(x_i, x_j, b_i, b_j, h_i, h_j)
        reg = self.head(pair_repr).squeeze(-1)
        cls = self.cls_head(pair_repr)
        return reg, cls

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        b_i: torch.Tensor,
        b_j: torch.Tensor,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
    ) -> torch.Tensor:
        reg, _ = self.forward_with_aux(x_i, x_j, b_i, b_j, h_i, h_j)
        return reg


def make_fixed_work_hotspot_tensors(
    df: pd.DataFrame,
    fixed_work_token_map: dict[tuple[str, str], np.ndarray],
    hotspot_token_map: dict[tuple[str, str], np.ndarray],
    device: torch.device,
    clip: float = LOG_RATIO_CLIP,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    xi = df[[f"xi_{col}" for col in NON_TIME_COLS]].values.astype(np.float32)
    xj = df[[f"xj_{col}" for col in NON_TIME_COLS]].values.astype(np.float32)

    keys_i = list(zip(df["program"].astype(str), df["variant_i"].astype(str)))
    keys_j = list(zip(df["program"].astype(str), df["variant_j"].astype(str)))
    bi = np.stack([fixed_work_token_map[key] for key in keys_i], axis=0).astype(np.float32)
    bj = np.stack([fixed_work_token_map[key] for key in keys_j], axis=0).astype(np.float32)
    hi = np.stack([hotspot_token_map[key] for key in keys_i], axis=0).astype(np.float32)
    hj = np.stack([hotspot_token_map[key] for key in keys_j], axis=0).astype(np.float32)

    y_raw = df["log_ratio"].values.astype(np.float32)
    y = np.clip(y_raw, -clip, clip)
    y_cls = df["label_int"].values.astype(np.int64) if "label_int" in df.columns else to_3class(y_raw).astype(np.int64)
    reg_weight = regression_sample_weights(y_raw)
    return (
        torch.from_numpy(xi).to(device),
        torch.from_numpy(xj).to(device),
        torch.from_numpy(bi).to(device),
        torch.from_numpy(bj).to(device),
        torch.from_numpy(hi).to(device),
        torch.from_numpy(hj).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(y_cls).to(device),
        torch.from_numpy(reg_weight).to(device),
    )


def train_fixed_work_hotspot_model(
    model: FixedWorkHotspotTokenPairTransformer,
    device: torch.device,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    fixed_work_token_map: dict[tuple[str, str], np.ndarray],
    hotspot_token_map: dict[tuple[str, str], np.ndarray],
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 25,
    huber_delta: float = 1.0,
    noise_std: float = 0.0,
    direction_lambda: float = 0.0,
    aux_class_lambda: float = DEFAULT_AUX_CLASS_LAMBDA,
    near_tie_threshold: float = NEAR_TIE_THRESHOLD,
    tie_reg_weight: float = DEFAULT_TIE_REG_WEIGHT,
    near_tie_reg_weight: float = DEFAULT_NEAR_TIE_REG_WEIGHT,
    class_balance_power: float = DEFAULT_CLASS_BALANCE_POWER,
) -> dict[str, list[float]]:
    xi_tr, xj_tr, bi_tr, bj_tr, hi_tr, hj_tr, y_tr, ycls_tr, regw_tr = make_fixed_work_hotspot_tensors(
        df_train,
        fixed_work_token_map,
        hotspot_token_map,
        device,
    )
    xi_va, xj_va, bi_va, bj_va, hi_va, hj_va, y_va, ycls_va, regw_va = make_fixed_work_hotspot_tensors(
        df_val,
        fixed_work_token_map,
        hotspot_token_map,
        device,
    )

    regw_tr = torch.from_numpy(
        regression_sample_weights(
            df_train["log_ratio"].values.astype(np.float32),
            near_tie_threshold=near_tie_threshold,
            tie_reg_weight=tie_reg_weight,
            near_tie_reg_weight=near_tie_reg_weight,
        )
    ).to(device)
    regw_va = torch.from_numpy(
        regression_sample_weights(
            df_val["log_ratio"].values.astype(np.float32),
            near_tie_threshold=near_tie_threshold,
            tie_reg_weight=tie_reg_weight,
            near_tie_reg_weight=near_tie_reg_weight,
        )
    ).to(device)

    loader = DataLoader(
        TensorDataset(xi_tr, xj_tr, bi_tr, bj_tr, hi_tr, hj_tr, y_tr, ycls_tr, regw_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    reg_criterion = nn.HuberLoss(delta=huber_delta, reduction="none")
    class_weights = torch.from_numpy(
        balanced_class_weights(ycls_tr.detach().cpu().numpy(), power=class_balance_power)
    ).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    no_improve = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_reg_loss": [],
        "val_reg_loss": [],
        "train_aux_class_loss": [],
        "val_aux_class_loss": [],
    }

    print(f"\n  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValR2':>7}  {'ValDir':>7}  {'Val3Cls':>8}  {'LR':>8}  {'Time':>6}")
    print("  " + "-" * 76)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses: list[float] = []
        train_reg_losses: list[float] = []
        train_aux_losses: list[float] = []
        for xi_b, xj_b, bi_b, bj_b, hi_b, hj_b, y_b, ycls_b, regw_b in loader:
            optimizer.zero_grad()
            if noise_std > 0.0:
                xi_b = xi_b + torch.randn_like(xi_b) * noise_std
                xj_b = xj_b + torch.randn_like(xj_b) * noise_std
                bi_b = bi_b + torch.randn_like(bi_b) * noise_std
                bj_b = bj_b + torch.randn_like(bj_b) * noise_std
                hi_b = hi_b + torch.randn_like(hi_b) * noise_std
                hj_b = hj_b + torch.randn_like(hj_b) * noise_std

            pred, cls_logits = model.forward_with_aux(xi_b, xj_b, bi_b, bj_b, hi_b, hj_b)
            reg_loss = weighted_mean(reg_criterion(pred, y_b), regw_b)
            loss = reg_loss
            aux_cls_loss = pred.new_tensor(0.0)
            if aux_class_lambda > 0.0:
                aux_cls_loss = cls_criterion(cls_logits, ycls_b)
                loss = loss + aux_class_lambda * aux_cls_loss
            if direction_lambda > 0.0:
                loss = loss + direction_lambda * direction_bce_loss(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_reg_losses.append(reg_loss.item())
            train_aux_losses.append(aux_cls_loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred, val_cls_logits = model.forward_with_aux(xi_va, xj_va, bi_va, bj_va, hi_va, hj_va)
            val_reg_loss = weighted_mean(reg_criterion(val_pred, y_va), regw_va)
            val_aux_cls_loss = cls_criterion(val_cls_logits, ycls_va) if aux_class_lambda > 0.0 else val_pred.new_tensor(0.0)
            val_loss = val_reg_loss + aux_class_lambda * val_aux_cls_loss
            if direction_lambda > 0.0:
                val_loss = val_loss + direction_lambda * direction_bce_loss(val_pred, y_va)

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss.item()))
        history["train_reg_loss"].append(float(np.mean(train_reg_losses)))
        history["val_reg_loss"].append(float(val_reg_loss.item()))
        history["train_aux_class_loss"].append(float(np.mean(train_aux_losses)))
        history["val_aux_class_loss"].append(float(val_aux_cls_loss.item()))

        if epoch % 10 == 0 or epoch == 1:
            val_pred_np = val_pred.detach().cpu().float().numpy()
            y_true_np = y_va.detach().cpu().float().numpy()
            metrics = compute_metrics(y_true_np, val_pred_np)
            elapsed = time.time() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"  {epoch:5d}  {train_loss:10.4f}  {float(val_loss.item()):10.4f}  "
                f"{metrics['r2']:7.4f}  {metrics['dir_acc']:7.4f}  {metrics['acc_3cls']:8.4f}  {cur_lr:8.2e}  {elapsed:5.1f}s"
            )

        val_loss_scalar = float(val_loss.item())
        if val_loss_scalar < best_val - 1e-5:
            best_val = val_loss_scalar
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [early stop] epoch={epoch}, val_loss stalled for {patience} rounds")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  best val_loss = {best_val:.4f}")
    return history


@torch.no_grad()
def predict_with_aux_np_fixed_work_hotspot(
    model: FixedWorkHotspotTokenPairTransformer,
    df: pd.DataFrame,
    fixed_work_token_map: dict[tuple[str, str], np.ndarray],
    hotspot_token_map: dict[tuple[str, str], np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    xi, xj, bi, bj, hi, hj, _, _, _ = make_fixed_work_hotspot_tensors(
        df,
        fixed_work_token_map,
        hotspot_token_map,
        device,
    )
    model.eval()
    pred, cls_logits = model.forward_with_aux(xi, xj, bi, bj, hi, hj)
    return pred.detach().cpu().float().numpy(), cls_logits.detach().cpu().float().numpy()


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    preset = EXPERIMENT_CONFIGS.get(pre_args.config, {}) if pre_args.config else {}
    if pre_args.config and pre_args.config not in EXPERIMENT_CONFIGS:
        sys.exit("[error] unknown preset '" + str(pre_args.config) + "', available: " + ", ".join(EXPERIMENT_CONFIGS))

    parser = argparse.ArgumentParser(
        description="Experimental fixed-work+hotspot-token transformer training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, choices=list(EXPERIMENT_CONFIGS))
    parser.add_argument("--pairs", default="train_set/pairs.parquet")
    parser.add_argument("--run-features", default="train_set/run_features.parquet")
    parser.add_argument("--output", default="train_set/fixed_work_hotspot_token_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=NUM_BUCKETS_DEFAULT, dest="num_buckets")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, dest="top_k")
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
        sys.exit("[error] hybrid token run mapping is incomplete for the active pairs table")

    schema = build_fixed_work_hotspot_schema(num_buckets=args.num_buckets, top_k=args.top_k)
    schema_path = out_dir / "fixed_work_hotspot_token_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    print("\n" + sep)
    print("Step 1: Build fixed-work + hotspot token caches")
    print(sep)
    raw_fixed_work_token_map, fixed_work_summary = build_fixed_work_token_map(run_key_df, num_buckets=args.num_buckets)
    raw_hotspot_token_map, hotspot_summary = build_hotspot_token_map(run_key_df, top_k=args.top_k)
    fixed_work_scaler = fit_fixed_work_token_scaler(raw_fixed_work_token_map, train_keys=train_keys)
    hotspot_scaler = fit_hotspot_token_scaler(raw_hotspot_token_map, train_keys=train_keys)
    fixed_work_token_map = apply_fixed_work_token_scaler(raw_fixed_work_token_map, fixed_work_scaler)
    hotspot_token_map = apply_hotspot_token_scaler(raw_hotspot_token_map, hotspot_scaler)
    print(
        f"  fixed_work runs={fixed_work_summary['n_runs']}  buckets={fixed_work_summary['num_buckets']}  bucket_features={len(WORK_BUCKET_FEATURE_NAMES)}  "
        f"window_count(median)={fixed_work_summary['window_count']['median']:.1f}  active_windows(median)={fixed_work_summary['active_window_count']['median']:.1f}"
    )
    print(
        f"  hotspot   runs={hotspot_summary['n_runs']}  top_k={hotspot_summary['top_k']}  window_features={len(WINDOW_FEATURE_NAMES)}  "
        f"window_count(median)={hotspot_summary['window_count']['median']:.1f}  active_windows(median)={hotspot_summary['active_window_count']['median']:.1f}"
    )
    print(
        f"  fitted fixed_work scaler on {fixed_work_scaler['n_tokens']} train bucket tokens  "
        f"mean_assigned_instruction_share(mean)={fixed_work_summary['mean_assigned_instruction_share']['mean']:.4f}"
    )
    print(
        f"  fitted hotspot scaler on {hotspot_scaler['n_tokens']} train tokens  "
        f"padding(mean)={hotspot_summary['padding_token_count']['mean']:.2f}"
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
        "Step 3: Build FixedWorkHotspotTokenPairTransformer  "
        f"num_buckets={args.num_buckets}  top_k={args.top_k}  d_model={args.d_model}  "
        f"nhead={args.nhead}  nlayers={args.nlayers}  ffn={args.ffn_dim}"
    )
    print(sep)
    model = FixedWorkHotspotTokenPairTransformer(
        summary_dim=len(NON_TIME_COLS),
        bucket_feat_dim=len(WORK_BUCKET_FEATURE_NAMES),
        window_feat_dim=len(WINDOW_FEATURE_NAMES),
        num_buckets=args.num_buckets,
        top_k=args.top_k,
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

    history = train_fixed_work_hotspot_model(
        model=model,
        device=device,
        df_train=df_train,
        df_val=df_val,
        fixed_work_token_map=fixed_work_token_map,
        hotspot_token_map=hotspot_token_map,
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
        y_pred, cls_logits = predict_with_aux_np_fixed_work_hotspot(
            model,
            part,
            fixed_work_token_map,
            hotspot_token_map,
            device,
        )
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
            y_pred, cls_logits = predict_with_aux_np_fixed_work_hotspot(
                model,
                subset,
                fixed_work_token_map,
                hotspot_token_map,
                device,
            )
            y_true = subset["log_ratio"].values.astype(np.float32)
            metrics = compute_metrics(y_true, y_pred)
            metrics.update(compute_aux_metrics(y_true, cls_logits))
            key = f"{variant_i}-{variant_j}"
            per_pair[key] = metrics
            print(
                f"  {key}: n={metrics['n']:3d}  dir_acc={metrics['dir_acc']}  reg_acc_3cls={metrics['acc_3cls']:.3f}  aux_acc_3cls={metrics['aux_acc_3cls']:.3f}"
            )

    model_path = out_dir / "model_fixed_work_hotspot_token_transformer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hparams": {
                "summary_dim": len(NON_TIME_COLS),
                "bucket_feat_dim": len(WORK_BUCKET_FEATURE_NAMES),
                "window_feat_dim": len(WINDOW_FEATURE_NAMES),
                "num_buckets": args.num_buckets,
                "top_k": args.top_k,
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
            "bucket_feature_names": list(WORK_BUCKET_FEATURE_NAMES),
            "window_feature_names": list(WINDOW_FEATURE_NAMES),
            "hybrid_schema": schema,
            "fixed_work_scaler": fixed_work_scaler,
            "hotspot_scaler": hotspot_scaler,
        },
        model_path,
    )

    eval_result = {
        "model": "FixedWorkHotspotTokenPairTransformer",
        "config": args.config,
        "seed": args.seed,
        "pairs_path": _display_path(pairs_path),
        "run_features_path": _display_path(run_features_path),
        "output_dir": _display_path(out_dir),
        "architecture": (
            f"[summary+{args.num_buckets} fixed_work_bucket+top{args.top_k} hotspot_window tokens]x2->"
            f"TransformerEncoder({args.nlayers}L,{args.nhead}H,ffn={args.ffn_dim})->"
            "[summary_i;summary_j;summary_i-summary_j]->reg_head+cls_head"
        ),
        "source_isolation": {
            "baseline_script": "scripts/train_transformer.py",
            "experimental_script": "scripts/experimental/fixed_work_hotspot_token/train_fixed_work_hotspot_token_transformer.py",
            "schema_scripts": [
                "scripts/experimental/fixed_work_token/fixed_work_token_schema.py",
                "scripts/experimental/hotspot_token/hotspot_token_schema.py",
            ],
            "overwrites_baseline_outputs": False,
        },
        "token_schema": schema,
        "token_cache_summary": {
            "fixed_work": fixed_work_summary,
            "hotspot": hotspot_summary,
        },
        "token_scaler": {
            "fixed_work": fixed_work_scaler,
            "hotspot": hotspot_scaler,
        },
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
    eval_path = out_dir / "model_fixed_work_hotspot_token_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    print(f"\n[ok] model saved: {model_path}")
    print(f"[ok] eval saved: {eval_path}")
    print(f"[ok] schema saved: {schema_path}")


if __name__ == "__main__":
    main()