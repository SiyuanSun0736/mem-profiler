#!/usr/bin/env python3
"""
train_transformer.py — 基于 Transformer Encoder 的成对优化级别预测

架构：
    每对 (x_i, x_j) 各为一个 token，经共享线性投影后输入 TransformerEncoder，
    最后拼接两个输出 token 与差向量做回归：
        [x_i ; x_j]  →  投影层  →  TransformerEncoder  →
        [out_i ; out_j ; out_i − out_j]  →  回归头  →  log_ratio

标签：log(total_cycles_j / total_cycles_i)
设备优先级：DirectML (WSL2/Windows) → CUDA → CPU

用法：
    python scripts/train_transformer.py
    python scripts/train_transformer.py --d-model 64 --nhead 4 --nlayers 3 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIE_THRESHOLD = 0.05
LOG_RATIO_CLIP = 6.0          # 裁剪极端 log_ratio（±6σ ≈ 400x cycles 差异）

# ── 微调预设（参考 Siamese-MicroPerf tuned_configs.py）──────────────────────
# 机制：fixed_work — log_ratio = log(cycles_j / cycles_i)，clip ±6
# 模型：PairTransformer（双 token Transformer Encoder）
#
# 参数说明
# --------
# huber_delta     : Huber loss 的 δ，越小越不敏感于极端 log_ratio
# direction_lambda: 方向辅助损失权重（BCE on sign），0 表示纯回归
# noise_std       : 输入高斯噪声标准差（训练增广），0 表示关闭
# patience        : 早停等待轮数
TUNED_CONFIGS: dict[str, dict] = {
    "fixed_work_transformer": {
        # 模型超参
        "d_model":    64,
        "nhead":       2,
        "nlayers":     3,
        "ffn_dim":   256,
        "dropout":  0.10,
        # 训练超参
        "lr":               1.5e-4,
        "wd":               1e-4,
        "epochs":           300,
        "batch":             64,
        "patience":          55,
        "clip":             6.0,
        "huber_delta":       0.5,
        "direction_lambda": 0.15,
        "noise_std":       0.008,
    },
    "fixed_work_transformer_strong": {
        # 更深、更大 FFN，适合数据量增大后使用
        "d_model":   128,
        "nhead":       4,
        "nlayers":     4,
        "ffn_dim":   512,
        "dropout":  0.15,
        "lr":               8e-5,
        "wd":               2e-4,
        "epochs":           400,
        "batch":             32,
        "patience":          60,
        "clip":             6.0,
        "huber_delta":       0.5,
        "direction_lambda": 0.20,
        "noise_std":       0.010,
    },
}

# ── 特征列（与 build_pair_table.py / train_model.py 保持一致）───────────────
NON_TIME_COLS: list[str] = [
    # 效率指标
    "ipc", "cpi",
    # LLC
    "llc_load_miss_rate", "llc_store_miss_rate",
    "llc_mpki", "llc_store_mpki",
    # dTLB
    "dtlb_miss_rate", "dtlb_mpki",
    # iTLB
    "itlb_mpki",
    # page fault
    "fault_per_ki", "fault_per_ms",
    "minor_fault_ratio",
    # 采样密度
    "samples_per_ms",
    # 窗口分布 — IPC
    "win_ipc_mean", "win_ipc_std", "win_ipc_p95", "win_ipc_peak_share", "win_ipc_min",
    # 窗口分布 — LLC MPKI
    "win_llc_mpki_mean", "win_llc_mpki_std", "win_llc_mpki_p95",
    "win_llc_mpki_peak_share", "win_llc_mpki_min",
    # 窗口分布 — dTLB MPKI
    "win_dtlb_mpki_mean", "win_dtlb_mpki_std", "win_dtlb_mpki_p95",
    "win_dtlb_mpki_peak_share", "win_dtlb_mpki_min",
    # 窗口分布 — iTLB MPKI
    "win_itlb_mpki_mean", "win_itlb_mpki_std", "win_itlb_mpki_p95",
    "win_itlb_mpki_peak_share", "win_itlb_mpki_min",
    # 窗口分布 — page fault
    "win_fault_mean", "win_fault_std", "win_fault_p95",
    "win_fault_peak_share", "win_fault_min",
    # Fault 子类型比例（bounded [0,1]）
    "anon_fault_ratio",
    "file_fault_ratio",
    "write_fault_ratio",
    "instruction_fault_ratio",
    # MM syscall 密度
    "mmap_per_ms",
    "munmap_per_ms",
    "brk_per_ms",
    "mm_syscall_per_ms",
    "mmap_bytes_per_ms",
    # 阶段特征（warmup / steady-state）
    "warmup_ipc",
    "steady_ipc",
    "phase_ipc_ratio",
    "warmup_llc_mpki",
    "steady_llc_mpki",
    "phase_llc_ratio",
    "phase_fault_ratio",
]
F = len(NON_TIME_COLS)        # 54 个每运行特征（38 原有 + 16 新扩展字段）


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  设备选择：DirectML → CUDA → CPU                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def select_device(force: Optional[str] = None) -> torch.device:
    """
    按优先级选择计算设备：
      1. force 参数指定的设备（如 'cpu', 'cuda:0', 'privateuseone:0'）
      2. torch_directml（WSL2 / Windows DirectX12 GPU）
      3. torch.cuda（NVIDIA GPU）
      4. CPU 兜底

    DirectML 设备标识符为 'privateuseone:0'（内部名称），
    对外打印时显示为 'directml:0' 以便阅读。
    """
    if force:
        dev = torch.device(force)
        print(f"[device] 强制使用: {dev}")
        return dev

    # 1. DirectML（WSL2 + Windows GPU）
    try:
        import torch_directml
        dml_dev = torch_directml.device()
        # 做一次简单运算确认真正可用
        _ = (torch.ones(1).to(dml_dev) + 1).item()
        print(f"[device] DirectML (privateuseone:0)  ✓")
        return dml_dev
    except Exception as e:
        print(f"[device] DirectML 不可用: {e}")

    # 2. CUDA
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}  ✓")
        return dev

    # 3. CPU 兜底
    print("[device] CPU  (无 GPU 加速)")
    return torch.device("cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  模型定义                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PairTransformer(nn.Module):
    """
    成对 Transformer Encoder 回归模型。

    输入：
        x_i : (batch, F)  变体 i 的 z-score 特征
        x_j : (batch, F)  变体 j 的 z-score 特征

    流程：
        1. 共享线性投影：F → d_model
        2. 加入可学习的 token-type embedding（i=0, j=1）
        3. TransformerEncoder (num_layers 层, nhead 头)
        4. 取两个输出 token，拼接 [out_i ; out_j ; out_i−out_j]
        5. 回归头 → 标量 log_ratio

    输出：
        log_ratio : (batch,)
    """

    def __init__(
        self,
        feat_dim:       int   = F,
        d_model:        int   = 64,
        nhead:          int   = 4,
        num_layers:     int   = 3,
        dim_feedforward:int   = 256,
        dropout:        float = 0.1,
        head_hidden:    int   = 64,
    ) -> None:
        super().__init__()

        # 1. 输入投影（xi 和 xj 共享权重）
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. 可学习 token-type embedding：0=i，1=j
        self.token_type_emb = nn.Embedding(2, d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (batch, seq, d_model)
            norm_first=True,     # Pre-LN：训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 回归头：输入 3 * d_model（out_i, out_j, diff）
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # 投影：(batch, F) → (batch, d_model)
        h_i = self.proj(x_i)
        h_j = self.proj(x_j)

        # token-type embedding
        device = x_i.device
        type_ids = torch.zeros(x_i.size(0), dtype=torch.long, device=device)
        h_i = h_i + self.token_type_emb(type_ids)
        type_ids = torch.ones(x_i.size(0), dtype=torch.long, device=device)
        h_j = h_j + self.token_type_emb(type_ids)

        # 拼成序列：(batch, 2, d_model)
        seq = torch.stack([h_i, h_j], dim=1)

        # Transformer Encoder
        out = self.encoder(seq)            # (batch, 2, d_model)

        out_i = out[:, 0, :]               # (batch, d_model)
        out_j = out[:, 1, :]               # (batch, d_model)
        diff  = out_i - out_j              # (batch, d_model)

        # 回归头
        cat   = torch.cat([out_i, out_j, diff], dim=-1)   # (batch, 3*d_model)
        return self.head(cat).squeeze(-1)                  # (batch,)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  评估指标（与 train_model.py 保持一致）                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def to_3class(log_ratio: np.ndarray) -> np.ndarray:
    cls = np.ones(len(log_ratio), dtype=np.int8)
    cls[log_ratio >  TIE_THRESHOLD] = 0
    cls[log_ratio < -TIE_THRESHOLD] = 2
    return cls


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_pred - y_true
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    non_tie = np.abs(y_true) > TIE_THRESHOLD
    if non_tie.sum() > 0:
        dir_acc = float(np.mean(
            np.sign(y_pred[non_tie]) == np.sign(y_true[non_tie])
        ))
    else:
        dir_acc = float("nan")

    acc_3cls = float(np.mean(to_3class(y_pred) == to_3class(y_true)))

    return {
        "n":        int(len(y_true)),
        "mae":      round(mae,      4),
        "rmse":     round(rmse,     4),
        "r2":       round(r2,       4),
        "dir_acc":  round(dir_acc,  4),
        "acc_3cls": round(acc_3cls, 4),
    }


def naive_rank_baseline(df: pd.DataFrame) -> dict:
    rank_diff = df["variant_rank_diff"].values
    y_pred = np.where(rank_diff > 0, -0.1, np.where(rank_diff < 0, 0.1, 0.0))
    return compute_metrics(df["log_ratio"].values, y_pred)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  数据准备                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def split_by_program(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    programs = np.array(sorted(df["program"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(programs)
    n = len(programs)
    n_train = int(train_frac * n)
    n_val   = int((train_frac + val_frac) * n)
    return (
        df[df["program"].isin(set(programs[:n_train]))].copy(),
        df[df["program"].isin(set(programs[n_train:n_val]))].copy(),
        df[df["program"].isin(set(programs[n_val:]))].copy(),
    )


def make_tensors(
    df: pd.DataFrame,
    device: torch.device,
    clip: float = LOG_RATIO_CLIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (xi_tensor, xj_tensor, y_tensor)，y 被裁剪到 ±clip。"""
    xi = df[[f"xi_{c}" for c in NON_TIME_COLS]].values.astype(np.float32)
    xj = df[[f"xj_{c}" for c in NON_TIME_COLS]].values.astype(np.float32)
    y  = np.clip(df["log_ratio"].values.astype(np.float32), -clip, clip)
    return (
        torch.from_numpy(xi).to(device),
        torch.from_numpy(xj).to(device),
        torch.from_numpy(y).to(device),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  训练循环                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(
    model:          PairTransformer,
    device:         torch.device,
    df_train:       pd.DataFrame,
    df_val:         pd.DataFrame,
    epochs:         int   = 200,
    batch_size:     int   = 64,
    lr:             float = 3e-4,
    weight_decay:   float = 1e-4,
    patience:       int   = 25,
    huber_delta:    float = 1.0,
    noise_std:      float = 0.0,
    direction_lambda: float = 0.0,
) -> dict:
    """
    训练循环，返回 {'train_loss': [...], 'val_loss': [...]} 历史。

    huber_delta     : Huber loss δ（越小对极端样本越不敏感）
    noise_std       : 输入高斯噪声标准差（训练增广，0 = 关闭）
    direction_lambda: 方向辅助 BCE loss 权重（0 = 纯回归）
    """
    xi_tr, xj_tr, y_tr = make_tensors(df_train, device)
    xi_va, xj_va, y_va = make_tensors(df_val,   device)

    loader = DataLoader(
        TensorDataset(xi_tr, xj_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.HuberLoss(delta=huber_delta)

    best_val  = float("inf")
    best_state: dict = {}
    no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    print(f"\n  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValR²':>7}  "
          f"{'ValDir':>7}  {'LR':>8}  {'Time':>6}")
    print("  " + "─" * 66)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses: list[float] = []
        for xi_b, xj_b, y_b in loader:
            optimizer.zero_grad()
            # 输入噪声增广
            if noise_std > 0.0:
                xi_b = xi_b + torch.randn_like(xi_b) * noise_std
                xj_b = xj_b + torch.randn_like(xj_b) * noise_std
            pred = model(xi_b, xj_b)
            loss = criterion(pred, y_b)
            # 方向辅助 BCE loss（仅在 |y_true| > TIE_THRESHOLD 的样本上）
            if direction_lambda > 0.0:
                non_tie = y_b.abs() > TIE_THRESHOLD
                if non_tie.sum() > 0:
                    dir_target = (y_b[non_tie] > 0).float()
                    dir_logit  = pred[non_tie]
                    dir_loss   = nn.functional.binary_cross_entropy_with_logits(
                        dir_logit, dir_target
                    )
                    loss = loss + direction_lambda * dir_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(xi_va, xj_va)
            val_loss = criterion(val_pred, y_va).item()

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 每 10 个 epoch 打印一次
        if epoch % 10 == 0 or epoch == 1:
            vp_np = val_pred.cpu().float().numpy()
            yt_np = y_va.cpu().float().numpy()
            m = compute_metrics(yt_np, vp_np)
            elapsed = time.time() - t0
            cur_lr  = scheduler.get_last_lr()[0]
            print(
                f"  {epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  "
                f"{m['r2']:7.4f}  {m['dir_acc']:7.4f}  {cur_lr:8.2e}  {elapsed:5.1f}s"
            )

        # Early stopping
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [早停] epoch={epoch}，val_loss {patience} 轮无改善")
                break

    # 恢复最佳权重
    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  最佳 val_loss = {best_val:.4f}")
    return history


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  推理工具                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_np(
    model: PairTransformer,
    df: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    xi = torch.from_numpy(
        df[[f"xi_{c}" for c in NON_TIME_COLS]].values.astype(np.float32)
    ).to(device)
    xj = torch.from_numpy(
        df[[f"xj_{c}" for c in NON_TIME_COLS]].values.astype(np.float32)
    ).to(device)
    return model(xi, xj).cpu().float().numpy()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  主入口                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    # ── 两阶段解析：先检测 --config，再以预设覆盖 argparse 默认值 ────────
    # 这样显式传入的 CLI 参数仍可覆盖预设。
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()
    _cfg: dict = TUNED_CONFIGS.get(_pre_args.config, {}) if _pre_args.config else {}
    if _pre_args.config and _pre_args.config not in TUNED_CONFIGS:
        sys.exit(
            f"[错误] 未知预设 '{_pre_args.config}'，可用: "
            + ", ".join(TUNED_CONFIGS)
        )

    parser = argparse.ArgumentParser(
        description="Transformer Encoder 成对回归训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",   default=None,
                        choices=list(TUNED_CONFIGS),
                        help="加载微调预设（其余 CLI 参数仍可覆盖）")
    parser.add_argument("--pairs",    default="train_set/pairs.parquet")
    parser.add_argument("--output",   default="train_set")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--d-model",  type=int,   default=64,   dest="d_model")
    parser.add_argument("--nhead",    type=int,   default=4)
    parser.add_argument("--nlayers",  type=int,   default=3)
    parser.add_argument("--ffn-dim",  type=int,   default=256,  dest="ffn_dim")
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--wd",       type=float, default=1e-4)
    parser.add_argument("--patience", type=int,   default=25)
    parser.add_argument("--huber-delta",      type=float, default=1.0,  dest="huber_delta",
                        help="Huber loss delta")
    parser.add_argument("--direction-lambda", type=float, default=0.0,  dest="direction_lambda",
                        help="方向辅助 BCE loss 权重（0=关闭）")
    parser.add_argument("--noise-std",        type=float, default=0.0,  dest="noise_std",
                        help="输入高斯噪声标准差（训练增广，0=关闭）")
    parser.add_argument("--device",   default=None,
                        help="强制设备，如 cpu / cuda:0 / privateuseone:0")
    parser.add_argument("--clip",     type=float, default=LOG_RATIO_CLIP,
                        help="log_ratio 裁剪范围（±clip）")

    # 将预设值注入为 argparse 默认值（显式 CLI 参数优先级更高）
    if _cfg:
        parser.set_defaults(**_cfg)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs_path = (REPO_ROOT / args.pairs).resolve()
    out_dir    = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        sys.exit(f"[错误] 找不到 {pairs_path}，请先运行 build_pair_table.py")

    # ── 设备选择 ──────────────────────────────────────────────────────────
    device = select_device(args.device)

    # ── 数据加载 ──────────────────────────────────────────────────────────
    df = pd.read_parquet(pairs_path)

    xi_cols   = [f"xi_{c}"   for c in NON_TIME_COLS]
    xj_cols   = [f"xj_{c}"   for c in NON_TIME_COLS]
    diff_cols = [f"diff_{c}" for c in NON_TIME_COLS]
    missing = [c for c in xi_cols + xj_cols + diff_cols if c not in df.columns]
    if missing:
        sys.exit(f"[错误] 缺少特征列: {missing[:5]}...")

    # ── Step 1: 数据划分 ──────────────────────────────────────────────────
    sep = "=" * 62
    print(sep)
    print("Step 1: 按程序划分 train / val / test")
    print(sep)
    df_train, df_val, df_test = split_by_program(df, seed=args.seed)
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = "  ".join(
            f"{c}={part['label_class'].value_counts().get(c, 0)}"
            for c in ["i_better", "tie", "j_better"]
        )
        print(
            f"  {name:5s}: {part['program'].nunique():3d} 程序  "
            f"{len(part):5d} 对  label分布: {dist}"
        )

    # ── Step 2: 朴素基准 ──────────────────────────────────────────────────
    print("\n" + sep)
    print("Step 2: 朴素基准（variant 名义排名）")
    print(sep)
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        m = naive_rank_baseline(part)
        print(
            f"  {name:5s} | MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
            f"dir_acc={m['dir_acc']:.4f}  acc_3cls={m['acc_3cls']:.3f}"
        )

    # ── Step 3: 模型构建与训练 ────────────────────────────────────────────
    print("\n" + sep)
    print(
        f"Step 3: 构建 PairTransformer  "
        f"d_model={args.d_model}  nhead={args.nhead}  "
        f"nlayers={args.nlayers}  ffn={args.ffn_dim}"
    )
    print(sep)

    model = PairTransformer(
        feat_dim=F,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.nlayers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数量: {n_params:,}")
    print(f"  log_ratio 裁剪范围: ±{args.clip}")
    config_label = f"[预设: {args.config}]  " if args.config else ""
    print(
        f"  {config_label}HuberLoss(δ={args.huber_delta})  "
        f"dir_λ={args.direction_lambda}  noise={args.noise_std}  "
        f"AdamW lr={args.lr}  wd={args.wd}"
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
    )

    # ── Step 4: 评估 ──────────────────────────────────────────────────────
    print("\n" + sep)
    print("Step 4: 评估结果")
    print(sep)

    header = (
        f"  {'split':5s} | {'MAE':>7} {'RMSE':>7} {'R²':>7} "
        f"{'dir_acc':>8} {'acc_3cls':>9}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    results: dict[str, dict] = {}
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        y_pred = predict_np(model, part, device)
        y_true = part["log_ratio"].values.astype(np.float32)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(
            f"  {name:5s} | {m['mae']:7.4f} {m['rmse']:7.4f} {m['r2']:7.4f} "
            f"{m['dir_acc']:>8} {m['acc_3cls']:9.3f}"
        )

    # ── Step 5: 按 variant 对细分 ─────────────────────────────────────────
    print("\n[info] test 集按 variant 对的方向准确率：")
    per_pair: dict[str, dict] = {}
    for vi in ["O0", "O1", "O2"]:
        for vj in ["O1", "O2", "O3"]:
            if vi >= vj:
                continue
            mask = (df_test["variant_i"] == vi) & (df_test["variant_j"] == vj)
            if mask.sum() == 0:
                continue
            sub = df_test[mask]
            yp  = predict_np(model, sub, device)
            yt  = sub["log_ratio"].values.astype(np.float32)
            m   = compute_metrics(yt, yp)
            per_pair[f"{vi}-{vj}"] = m
            print(
                f"  {vi}-{vj}: n={m['n']:3d}  "
                f"dir_acc={m['dir_acc']}  acc_3cls={m['acc_3cls']:.3f}"
            )

    # ── 保存模型 ──────────────────────────────────────────────────────────
    model_path = out_dir / "model_transformer.pt"
    # 保存 state_dict + 超参数（便于后续加载推理）
    torch.save(
        {
            "model_state": model.state_dict(),
            "hparams": {
                "feat_dim":       F,
                "d_model":        args.d_model,
                "nhead":          args.nhead,
                "num_layers":     args.nlayers,
                "dim_feedforward":args.ffn_dim,
                "dropout":        args.dropout,
            },
            "non_time_cols": NON_TIME_COLS,
        },
        model_path,
        # DirectML 设备上的 tensor 需先 .cpu() 再保存，state_dict 已在 CPU
    )

    eval_result = {
        "model":          "PairTransformer",
        "architecture":   (
            f"[xi;xj]→proj(F→{args.d_model})→TokenTypeEmb→"
            f"TransformerEncoder({args.nlayers}L,{args.nhead}H,ffn={args.ffn_dim})"
            f"→[out_i;out_j;out_i-out_j]→head({3*args.d_model}→{64}→1)"
        ),
        "n_params":       n_params,
        "device":         str(device),
        "log_ratio_clip": args.clip,
        "splits": {
            "train_programs": int(df_train["program"].nunique()),
            "val_programs":   int(df_val["program"].nunique()),
            "test_programs":  int(df_test["program"].nunique()),
            "train_pairs":    len(df_train),
            "val_pairs":      len(df_val),
            "test_pairs":     len(df_test),
        },
        "results":   results,
        "per_pair":  per_pair,
        "history":   {k: [round(v, 6) for v in vs] for k, vs in history.items()},
    }
    eval_path = out_dir / "model_transformer_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    print(f"\n[ok] 模型已保存: {model_path}")
    print(f"[ok] 评估已保存: {eval_path}")


if __name__ == "__main__":
    main()
