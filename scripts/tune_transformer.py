"""
tune_transformer.py — PairTransformer 超参搜索空间 & 固定配置
==============================================================
只包含参数定义，供手动或脚本调用使用。
"""

from __future__ import annotations

from typing import Any

# ── 网格搜索空间 ──────────────────────────────────────────────────────────────
# 所有组合的笛卡尔积（nhead 必须整除 d_model，无效组合在运行时过滤）
GRID_SPACE: dict[str, list[Any]] = {
    "d_model":          [32, 64],
    "nhead":            [2, 4],
    "nlayers":          [2, 3],
    "ffn_dim":          [128, 256],
    "dropout":          [0.1, 0.15],
    "lr":               [1e-4, 3e-4],
    "huber_delta":      [0.5, 1.0],
    "direction_lambda": [0.0, 0.15],
    "noise_std":        [0.0, 0.008],
}

# ── 随机搜索空间 ──────────────────────────────────────────────────────────────
# tuple → 连续区间（lr/wd 用对数均匀分布，其余线性均匀）
# list  → 离散候选值
RANDOM_SPACE: dict[str, Any] = {
    "d_model":          [32, 64, 128],
    "nhead":            [2, 4],
    "nlayers":          [2, 3, 4],
    "ffn_dim":          [128, 256, 512],
    "dropout":          (0.05, 0.25),
    "lr":               (5e-5, 5e-4),    # log-uniform
    "wd":               (1e-5, 5e-4),    # log-uniform
    "huber_delta":      (0.3, 1.5),
    "direction_lambda": (0.0, 0.5),
    "noise_std":        (0.0, 0.02),
    "batch":            [32, 64, 128],
}

# ── 调参阶段固定不搜索的训练参数 ──────────────────────────────────────────────
FIXED_TRAINING: dict[str, Any] = {
    "epochs":   150,   # 调参用较少 epoch，找到方向后再用完整轮数
    "patience":  30,
    "clip":      6.0,
    "wd":        1e-4,
}
