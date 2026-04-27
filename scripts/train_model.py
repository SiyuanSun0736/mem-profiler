#!/usr/bin/env python3
"""
train_model.py — Phase 1 成对 MLP 模型训练与评估

架构（直接拼接版，与 04-model-plan Phase 1 对应）：
    [x_i ; x_j ; x_i − x_j]  →  MLP(128 → 64 → 32)  →  log_ratio 回归

标签：log(total_cycles_j / total_cycles_i)
  > 0 → i 更优（i 执行用时更短）
  < 0 → j 更优

数据划分：按程序（program）划分 train/val/test，防止同程序数据泄漏。

评估指标：
  - MAE / RMSE（log_ratio 回归）
  - R²
  - 方向准确率：sign(pred) == sign(true)（排除 tie 区间）
  - 3 分类准确率：{i_better, tie, j_better}
  - 朴素基准：仅凭 variant 名义排名预测方向

用法：
    python scripts/train_model.py
    python scripts/train_model.py --pairs train_set/pairs.parquet --output train_set
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

from feature_columns import NON_TIME_COLS


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIE_THRESHOLD = 0.05


def feat_cols() -> list[str]:
    """返回 [x_i; x_j; diff] 的列名列表，共 len(NON_TIME_COLS) * 3 列。"""
    return (
        [f"xi_{c}" for c in NON_TIME_COLS]
        + [f"xj_{c}" for c in NON_TIME_COLS]
        + [f"diff_{c}" for c in NON_TIME_COLS]
    )


# ── 标签工具 ─────────────────────────────────────────────────────────────────

def to_3class(log_ratio: np.ndarray) -> np.ndarray:
    cls = np.ones(len(log_ratio), dtype=np.int8)   # 1 = tie
    cls[log_ratio >  TIE_THRESHOLD] = 0             # 0 = i_better
    cls[log_ratio < -TIE_THRESHOLD] = 2             # 2 = j_better
    return cls


# ── 评估指标 ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_pred - y_true
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # 方向准确率（仅在 true 非 tie 的样本上算）
    non_tie = np.abs(y_true) > TIE_THRESHOLD
    if non_tie.sum() > 0:
        dir_acc = float(np.mean(
            np.sign(y_pred[non_tie]) == np.sign(y_true[non_tie])
        ))
    else:
        dir_acc = float("nan")

    # 3 分类准确率
    pred_cls = to_3class(y_pred)
    true_cls = to_3class(y_true)
    acc_3cls = float(np.mean(pred_cls == true_cls))

    return {
        "n":        int(len(y_true)),
        "mae":      round(mae,      4),
        "rmse":     round(rmse,     4),
        "r2":       round(r2,       4),
        "dir_acc":  round(dir_acc,  4) if not isinstance(dir_acc, float) or not (dir_acc != dir_acc) else None,
        "acc_3cls": round(acc_3cls, 4),
    }


# ── 朴素基准：仅凭 variant 名义排名预测 ──────────────────────────────────────

def naive_rank_baseline(df: pd.DataFrame) -> dict:
    """
    朴素基准：若 variant_rank_diff > 0（j 名义上优化程度更高），预测 j_better；
    若 < 0，预测 i_better；= 0，预测 tie。
    """
    rank_diff = df["variant_rank_diff"].values
    y_pred_naive = np.where(rank_diff > 0, -0.1, np.where(rank_diff < 0, 0.1, 0.0))
    y_true = df["log_ratio"].values
    return compute_metrics(y_true, y_pred_naive)


# ── 线性基线：Ridge 回归 ────────────────────────────────────────────────────

def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> object:
    """
    Baseline 1：带 L2 正则的线性回归（Ridge）。
    用与 MLP 完全相同的 [xi; xj; xi-xj] 特征矩阵，确保比较公平。
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        sys.exit(
            "[错误] 缺少 scikit-learn，请运行：\n"
            "  .venv/bin/pip install scikit-learn"
        )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(X_train, y_train)
    return model


# ── 数据划分 ─────────────────────────────────────────────────────────────────

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
    train_p = set(programs[:n_train])
    val_p   = set(programs[n_train:n_val])
    test_p  = set(programs[n_val:])
    return (
        df[df["program"].isin(train_p)].copy(),
        df[df["program"].isin(val_p)].copy(),
        df[df["program"].isin(test_p)].copy(),
    )


# ── 模型训练 ─────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden: tuple[int, ...] = (128, 64, 32),
) -> object:
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        sys.exit(
            "[错误] 缺少 scikit-learn，请运行：\n"
            "  .venv/bin/pip install scikit-learn"
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )),
    ])
    model.fit(X_train, y_train)
    return model


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 成对 MLP 训练")
    parser.add_argument("--pairs",  default="train_set/pairs.parquet")
    parser.add_argument("--output", default="train_set")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--hidden", default="128,64,32",
        help="MLP 隐层大小，逗号分隔，默认 128,64,32",
    )
    args = parser.parse_args()

    pairs_path = (REPO_ROOT / args.pairs).resolve()
    out_dir    = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = tuple(int(x) for x in args.hidden.split(","))

    if not pairs_path.exists():
        sys.exit(f"[错误] 找不到 {pairs_path}，请先运行 build_pair_table.py")

    df = pd.read_parquet(pairs_path)
    _feat_cols = feat_cols()

    missing = [c for c in _feat_cols if c not in df.columns]
    if missing:
        sys.exit(f"[错误] 缺少特征列: {missing[:5]}...")

    # ── Step 1: 数据划分 ──────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Step 1: 按程序划分 train / val / test")
    print("=" * 60, flush=True)

    df_train, df_val, df_test = split_by_program(df, seed=args.seed)
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(
            f"  {name:5s}: {part['program'].nunique():3d} 程序  "
            f"{len(part):5d} 对  "
            f"label分布: "
            + "  ".join(
                f"{c}={part['label_class'].value_counts().get(c, 0)}"
                for c in ["i_better", "tie", "j_better"]
            )
        )

    X_train = df_train[_feat_cols].values.astype(np.float32)
    y_train = df_train["log_ratio"].values.astype(np.float32)
    X_val   = df_val[_feat_cols].values.astype(np.float32)
    y_val   = df_val["log_ratio"].values.astype(np.float32)
    X_test  = df_test[_feat_cols].values.astype(np.float32)
    y_test  = df_test["log_ratio"].values.astype(np.float32)

    # ── Step 2: 朴素基准 ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Step 2: 朴素基准（variant 名义排名）")
    print("=" * 60, flush=True)

    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        m = naive_rank_baseline(part)
        print(
            f"  {name:5s} | MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
            f"dir_acc={m['dir_acc']}  acc_3cls={m['acc_3cls']:.3f}"
        )

    # ── Step 2b: 线性基线（Ridge 回归）────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"Step 2b: 线性基线（Ridge α=1.0）  输入维度={len(_feat_cols)}")
    print("=" * 60, flush=True)

    ridge_model = train_ridge(X_train, y_train)

    header_r = f"  {'split':5s} | {'MAE':>7} {'RMSE':>7} {'R²':>7} {'dir_acc':>8} {'acc_3cls':>9}"
    print(header_r)
    print("  " + "-" * (len(header_r) - 2))
    ridge_results: dict[str, dict] = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        y_pred_r = ridge_model.predict(X).astype(np.float32)
        m_r = compute_metrics(y, y_pred_r)
        ridge_results[name] = m_r
        print(
            f"  {name:5s} | {m_r['mae']:7.4f} {m_r['rmse']:7.4f} {m_r['r2']:7.4f} "
            f"{str(m_r['dir_acc']):>8} {m_r['acc_3cls']:9.3f}"
        )

    # ── Step 3: 训练 MLP ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"Step 3: 训练 Phase 1 MLP  隐层={hidden}  输入维度={len(_feat_cols)}")
    print("=" * 60, flush=True)

    model = train_mlp(X_train, y_train, hidden=hidden)

    # ── Step 4: 评估 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Step 4: 评估结果")
    print("=" * 60, flush=True)

    header = f"  {'split':5s} | {'MAE':>7} {'RMSE':>7} {'R²':>7} {'dir_acc':>8} {'acc_3cls':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[str, dict] = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        y_pred = model.predict(X).astype(np.float32)
        m = compute_metrics(y, y_pred)
        results[name] = m
        print(
            f"  {name:5s} | {m['mae']:7.4f} {m['rmse']:7.4f} {m['r2']:7.4f} "
            f"{str(m['dir_acc']):>8} {m['acc_3cls']:9.3f}"
        )

    # ── Step 5: 每类 variant 对的准确率 ──────────────────────────────────
    print("\n[info] test 集按 variant 对的方向准确率：")
    for vi in ["O0", "O1", "O2"]:
        for vj in ["O1", "O2", "O3"]:
            if vi >= vj:
                continue
            mask = (df_test["variant_i"] == vi) & (df_test["variant_j"] == vj)
            if mask.sum() == 0:
                continue
            X_sub  = df_test.loc[mask, _feat_cols].values.astype(np.float32)
            y_sub  = df_test.loc[mask, "log_ratio"].values.astype(np.float32)
            yp_sub = model.predict(X_sub).astype(np.float32)
            m_sub  = compute_metrics(y_sub, yp_sub)
            print(
                f"  {vi}-{vj}: n={m_sub['n']:3d}  "
                f"dir_acc={m_sub['dir_acc']}  acc_3cls={m_sub['acc_3cls']:.3f}"
            )

    # ── 保存 ──────────────────────────────────────────────────────────────
    model_path = out_dir / "model_phase1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    eval_result = {
        "model":        "Phase1_MLP_concat",
        "architecture": f"[xi; xj; xi-xj] -> MLP{list(hidden)} -> log_ratio",
        "feature_view": "non_time",
        "feature_dim":  len(NON_TIME_COLS),
        "input_dim":    len(_feat_cols),
        "tie_threshold": TIE_THRESHOLD,
        "splits": {
            "train_programs": int(df_train["program"].nunique()),
            "val_programs":   int(df_val["program"].nunique()),
            "test_programs":  int(df_test["program"].nunique()),
            "train_pairs":    len(df_train),
            "val_pairs":      len(df_val),
            "test_pairs":     len(df_test),
        },
        "metrics":        results,
        "ridge_baseline": ridge_results,
    }

    eval_path = out_dir / "model_eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    print(f"\n[ok] 模型已保存:   {model_path}")
    print(f"[ok] 评估已保存:   {eval_path}")


if __name__ == "__main__":
    main()
