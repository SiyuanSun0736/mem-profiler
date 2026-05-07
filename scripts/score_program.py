#!/usr/bin/env python3
"""
score_program.py — 单程序优化分数推断（方案 A：参考锚点法 · Fixed Work）
=========================================================================

原理
----
已知锚点 r_k 的绝对分数 S_{r_k} = log(T_O0 / T_{r_k})，
成对模型预测：  model(query, r_k) = log(cycles_{r_k} / cycles_query)
则：           S_query = S_{r_k} + model(query, r_k)
               = log(T_O0/T_{r_k}) + log(T_{r_k}/T_query)
               = log(T_O0 / T_query)                    ← 与 O0 基准的对数性能提升

多锚点平均：   S_query = mean_k [ S_{r_k} + model(query, r_k) ]

0-100 分数：   使用训练集中所有变体的 score_gt 分布做百分位归一化。

档位：
  poor   [ 0, 25)  —— 接近 O0，基本无优化
  medium [25, 50)  —— 有一定优化，但不显著
  good   [50, 75)  —— 明显优化
  strong [75,100]  —— 接近或超过 O3

瓶颈归因（Level 1 规则打分）
  将 z-score 特征按 4 类瓶颈聚合为 severity 分数（越高越严重）：
    cache_bound : llc_mpki, llc_load_miss_rate, llc_store_miss_rate
    tlb_bound   : dtlb_mpki, dtlb_miss_rate, itlb_mpki
    fault_heavy : fault_per_ki, fault_per_ms
    low_ipc     : -ipc, cpi   （ipc 取反：低 IPC = 高压力）

输出
----
  train_set/scores.parquet     — 每次 run 的分数 + 归因
  train_set/score_eval.json    — 汇总评估指标（预测 vs 真值 S）

用法
----
  # 对全部 580 runs 评分
  python scripts/score_program.py

  # 对特定程序+变体打印诊断报告
  python scripts/score_program.py --program aha --variant O2

  # 强制 CPU
  python scripts/score_program.py --device cpu

    # 若更关注时间外部验证而不是 proxy 评分，可切回 time-first tuned 参数
    python scripts/score_program.py --tuned-selection-objective time

    # 将评估 JSON 写到自定义路径，便于并存多套结果
    python scripts/score_program.py \
            --output train_set/scores_time_first.parquet \
            --eval-output train_set/score_eval_time_first.json \
            --tuned-selection-objective time
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS  = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from train_transformer import (  # noqa: E402
    DROPPED_INPUT_FEATURES,
    NON_TIME_COLS,
    PairTransformer,
    select_device,
)

F = len(NON_TIME_COLS)
VARIANT_RANK = {"O0": 0, "O1": 1, "O2": 2, "O3": 3}
CLASS_LABELS = ("i_better", "tie", "j_better")
DEFAULT_TIE_GATE_THRESHOLD = 0.55
DEFAULT_TIE_SHRINK_POWER = 1.0
DEFAULT_TIE_MARGIN_WEIGHT_ALPHA = 0.30
DEFAULT_MIN_ANCHOR_QUALITY = 0.30
DEFAULT_OUTLIER_MAD_SCALE = 3.0
DEFAULT_OUTLIER_MIN_DELTA = 0.35
DEFAULT_TUNED_DEFAULTS_JSON = "train_set/score_tune_fine_variant_best.json"
DEFAULT_MIN_RELIABLE_TUNE_SCORE_VALID = 32
DEFAULT_MIN_RELIABLE_TUNE_TIME_VALID = 32
DEFAULT_TUNED_SELECTION_OBJECTIVE = "score"

# ── 瓶颈归因特征分组 ───────────────────────────────────────────────────────────
# 每组特征的 z-score 均值作为该类瓶颈的 severity（值越高压力越大）
BOTTLENECK_GROUPS: dict[str, list[str]] = {
    "cache_bound": [
        "llc_mpki", "llc_load_miss_rate", "llc_store_miss_rate", "llc_store_mpki",
        "win_llc_mpki_mean", "win_llc_mpki_peak_share",
    ],
    "tlb_bound": [
        "dtlb_mpki", "dtlb_miss_rate", "itlb_mpki",
        "win_dtlb_mpki_mean", "win_dtlb_mpki_peak_share",
        "win_itlb_mpki_mean", "win_itlb_mpki_peak_share",
    ],
    "fault_heavy": [
        "fault_per_ki", "fault_per_ms",
        "win_fault_mean", "win_fault_peak_share", "win_fault_p95",
    ],
    "low_ipc": [
        "cpi",          # cpi 越高越差，直接用 z-score
        "win_ipc_std",  # IPC 波动越大越不稳
    ],
}
# low_ipc 中 ipc 需要取反（低 IPC → 高压力）
_IPC_INVERT = {"ipc", "win_ipc_mean", "win_ipc_min", "win_ipc_p95"}

# ── 热点窗口辅助 ───────────────────────────────────────────────────────────────

def _load_hotspot_windows(output_dir: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    从 output_dir/window_metrics.jsonl 加载原始窗口数据，
    按热点分数（LLC MPKI + dTLB MPKI + IPC 惩罚）降序返回 top-K 窗口。

    每个返回的窗口包含：
      window_id, ipc, llc_mpki, dtlb_mpki, faults, hotspot_score
    """
    wm_path = pathlib.Path(output_dir) / "window_metrics.jsonl"
    if not wm_path.exists():
        return []
    try:
        raw_windows = [
            json.loads(line)
            for line in wm_path.read_text().splitlines()
            if line.strip()
        ]
    except Exception:
        return []
    if not raw_windows:
        return []

    scored: list[tuple[float, dict]] = []
    for w in raw_windows:
        instr  = max(float(w.get("instructions", 0)), 1)
        cycles = max(float(w.get("cycles", 0)), 1)
        ipc       = instr / cycles
        llc_mpki  = float(w.get("llc_load_misses", 0)) / instr * 1000
        dtlb_mpki = float(w.get("dtlb_misses", 0))     / instr * 1000
        faults    = float(w.get("minor_faults", 0) + w.get("major_faults", 0))
        # 热点分数：cache 压力 + TLB 压力 + IPC 惩罚（高 IPC = 低压力）
        hs = llc_mpki + 0.3 * dtlb_mpki + max(0.0, 2.0 - ipc) * 5.0
        scored.append((hs, {
            "window_id":   w.get("window_id", -1),
            "ipc":         round(ipc,       3),
            "llc_mpki":    round(llc_mpki,  2),
            "dtlb_mpki":   round(dtlb_mpki, 2),
            "faults":      int(faults),
            "hotspot_score": round(hs, 2),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:top_k]]


def _bottleneck_scores(feat_z: dict[str, float]) -> list[dict[str, Any]]:
    """
    根据 z-score 特征计算各类瓶颈的 severity（0~1 软剪裁）。

    返回按 severity 降序排列的列表，每项含 category / severity / support_metrics。
    """
    results = []
    for category, cols in BOTTLENECK_GROUPS.items():
        avail = [c for c in cols if c in feat_z]
        if not avail:
            continue
        vals = []
        support: dict[str, float] = {}
        for c in avail:
            z = feat_z[c]
            if c in _IPC_INVERT:
                z = -z
            vals.append(z)
            support[c] = round(feat_z[c], 3)
        severity = float(np.clip(np.mean(vals), 0.0, None))  # 负值截为 0
        # 归一化到 [0,1]：z≈2 视为严重
        severity_norm = float(min(severity / 2.0, 1.0))
        results.append({
            "category":       category,
            "severity":       round(severity_norm, 3),
            "rank":           0,  # 后续填充
            "support_metrics": support,
        })

    results.sort(key=lambda x: x["severity"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def _percentile_score(s_value: float, all_scores: np.ndarray) -> float:
    """将 log-time-ratio 映射为 0-100 百分位分。"""
    return float(np.mean(all_scores <= s_value)) * 100.0


def _band(pct: float) -> str:
    if pct >= 75:
        return "strong"
    if pct >= 50:
        return "good"
    if pct >= 25:
        return "medium"
    return "poor"


def _variant_distance_weight(query_variant: str, anchor_variant: str) -> float:
    q_rank = VARIANT_RANK.get(query_variant)
    a_rank = VARIANT_RANK.get(anchor_variant)
    if q_rank is None or a_rank is None:
        return 1.0
    distance = abs(q_rank - a_rank)
    return max(0.55, 1.0 - 0.15 * distance)


def _pair_vote_confidence(
    class_probs: list[float],
    decoded_class: str,
    tie_margin_weight_alpha: float,
) -> tuple[float, float, float]:
    i_prob, tie_prob, j_prob = [float(p) for p in class_probs]
    cls_conf = max(i_prob, tie_prob, j_prob)
    direction_margin = abs(i_prob - j_prob)
    alpha = float(np.clip(tie_margin_weight_alpha, 0.0, 1.0))
    if decoded_class == "tie":
        return cls_conf, cls_conf, direction_margin
    effective_conf = (1.0 - alpha) * cls_conf + alpha * direction_margin
    return effective_conf, cls_conf, direction_margin


def _decode_pair_log_ratio(
    pred_log_ratio: float,
    cls_logits: torch.Tensor,
    tie_gate_threshold: float,
    tie_shrink_power: float,
) -> dict[str, float | str | list[float]]:
    probs = torch.softmax(cls_logits, dim=-1).detach().cpu().numpy().reshape(-1)
    cls_idx = int(np.argmax(probs))
    decoded_class = CLASS_LABELS[cls_idx]
    tie_prob = float(probs[1])
    cls_conf = float(probs[cls_idx])
    gated_scale = max(0.0, 1.0 - tie_prob) ** tie_shrink_power
    gated_log_ratio = float(pred_log_ratio) * gated_scale

    if decoded_class == "tie" and tie_prob >= tie_gate_threshold:
        gated_log_ratio = 0.0
    elif decoded_class == "i_better":
        gated_log_ratio = abs(gated_log_ratio)
    elif decoded_class == "j_better":
        gated_log_ratio = -abs(gated_log_ratio)

    return {
        "decoded_class": decoded_class,
        "tie_prob": round(tie_prob, 6),
        "class_confidence": round(cls_conf, 6),
        "gated_scale": round(gated_scale, 6),
        "raw_log_ratio": round(float(pred_log_ratio), 6),
        "gated_log_ratio": round(float(gated_log_ratio), 6),
        "class_probs": [round(float(p), 6) for p in probs.tolist()],
    }


def _filter_anchor_estimates(
    anchor_estimates: list[dict[str, Any]],
    mad_scale: float,
    min_delta: float,
) -> list[dict[str, Any]]:
    if len(anchor_estimates) < 3:
        for item in anchor_estimates:
            item["used"] = item["weight"] > 0.0
            item["outlier_delta"] = 0.0
        return anchor_estimates

    scores = np.array([float(item["score_estimate_raw"]) for item in anchor_estimates], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    tol = max(float(min_delta), float(mad_scale) * mad)

    kept = 0
    for item in anchor_estimates:
        delta = abs(float(item["score_estimate_raw"]) - median)
        item["outlier_delta"] = round(delta, 6)
        item["used"] = (delta <= tol) and (float(item["weight"]) > 0.0)
        kept += int(item["used"])

    if kept == 0:
        for item in anchor_estimates:
            item["used"] = float(item["weight"]) > 0.0

    return anchor_estimates


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_reliable_tuned_best(
    best: dict[str, Any],
    shared_best: dict[str, Any] | None,
) -> tuple[bool, str]:
    n_score_valid = int(best.get("n_score_valid", 0) or 0)
    n_time_valid = int(best.get("n_time_valid", 0) or 0)
    score_corr = best.get("score_corr")
    time_corr = best.get("time_corr")

    if n_score_valid < DEFAULT_MIN_RELIABLE_TUNE_SCORE_VALID:
        return False, f"n_score_valid<{DEFAULT_MIN_RELIABLE_TUNE_SCORE_VALID}"
    if not _is_finite_number(score_corr):
        return False, "score_corr_not_finite"
    if n_time_valid < DEFAULT_MIN_RELIABLE_TUNE_TIME_VALID:
        return False, f"n_time_valid<{DEFAULT_MIN_RELIABLE_TUNE_TIME_VALID}"
    if not _is_finite_number(time_corr):
        return False, "time_corr_not_finite"
    if float(time_corr) < 0.0:
        return False, "time_corr_negative"

    if not shared_best:
        return True, "ok"

    shared_score_corr = shared_best.get("score_corr")
    if _is_finite_number(shared_score_corr):
        # 若局部 score_corr 明显落后于共享参数，则优先回退到 ALL。
        if float(score_corr) + 1e-6 < float(shared_score_corr) - 0.02:
            return False, "score_corr_much_worse_than_all"
    return True, "ok"


def _load_tuned_variant_defaults(
    path: pathlib.Path,
    selection_objective: str = DEFAULT_TUNED_SELECTION_OBJECTIVE,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] 读取 tuned defaults 失败: {path} ({exc})")
        return {}

    if selection_objective == "score":
        best_by_variant = data.get("best_for_score_by_variant") or data.get("best_by_variant")
    else:
        best_by_variant = data.get("best_by_variant")
    if not isinstance(best_by_variant, dict):
        return {}

    param_keys = {
        "tie_gate_threshold",
        "tie_shrink_power",
        "tie_margin_weight_alpha",
        "min_anchor_quality",
        "anchor_outlier_mad_scale",
        "anchor_outlier_min_delta",
    }
    raw_best_by_variant: dict[str, dict[str, Any]] = {}
    for variant, payload in best_by_variant.items():
        if not isinstance(payload, dict):
            continue
        best = payload.get("best")
        if not isinstance(best, dict):
            continue
        raw_best_by_variant[str(variant)] = best

    shared_best = raw_best_by_variant.get("ALL")
    variant_defaults: dict[str, dict[str, Any]] = {}
    for variant, best in raw_best_by_variant.items():
        picked = {
            key: float(best[key])
            for key in param_keys
            if key in best and best[key] is not None
        }
        if picked:
            is_reliable, reliability_reason = _is_reliable_tuned_best(best, shared_best)
            variant_defaults[str(variant)] = {
                "params": picked,
                "reliable": bool(is_reliable),
                "reason": reliability_reason,
                "metrics": {
                    "n_runs": int(best.get("n_runs", 0) or 0),
                    "n_time_valid": int(best.get("n_time_valid", 0) or 0),
                    "n_score_valid": int(best.get("n_score_valid", 0) or 0),
                    "time_corr": (
                        float(best.get("time_corr"))
                        if _is_finite_number(best.get("time_corr"))
                        else None
                    ),
                    "score_corr": (
                        float(best.get("score_corr"))
                        if _is_finite_number(best.get("score_corr"))
                        else None
                    ),
                },
            }
    return variant_defaults


def _resolve_scoring_params(
    query_variant: str,
    cli_overrides: dict[str, float | None],
    tuned_defaults: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    fallback_defaults = {
        "tie_gate_threshold": DEFAULT_TIE_GATE_THRESHOLD,
        "tie_shrink_power": DEFAULT_TIE_SHRINK_POWER,
        "tie_margin_weight_alpha": DEFAULT_TIE_MARGIN_WEIGHT_ALPHA,
        "min_anchor_quality": DEFAULT_MIN_ANCHOR_QUALITY,
        "anchor_outlier_mad_scale": DEFAULT_OUTLIER_MAD_SCALE,
        "anchor_outlier_min_delta": DEFAULT_OUTLIER_MIN_DELTA,
    }
    variant_entry = tuned_defaults.get(query_variant, {})
    shared_entry = tuned_defaults.get("ALL", {})
    variant_defaults = variant_entry.get("params", {}) if isinstance(variant_entry, dict) else {}
    shared_defaults = shared_entry.get("params", {}) if isinstance(shared_entry, dict) else {}
    variant_reliable = bool(variant_entry.get("reliable", False)) if isinstance(variant_entry, dict) else False

    resolution_meta = {
        "query_variant": query_variant,
        "variant_tuned_available": bool(variant_defaults),
        "variant_tuned_reliable": variant_reliable,
        "variant_tuned_reason": variant_entry.get("reason") if isinstance(variant_entry, dict) else None,
        "shared_tuned_available": bool(shared_defaults),
        "source_by_key": {},
    }

    resolved: dict[str, float] = {}
    for key, fallback in fallback_defaults.items():
        override = cli_overrides.get(key)
        if override is not None:
            resolved[key] = float(override)
            resolution_meta["source_by_key"][key] = "cli"
        elif variant_reliable and key in variant_defaults:
            resolved[key] = float(variant_defaults[key])
            resolution_meta["source_by_key"][key] = f"variant:{query_variant}"
        elif key in shared_defaults:
            resolved[key] = float(shared_defaults[key])
            resolution_meta["source_by_key"][key] = "variant:ALL"
        else:
            resolved[key] = float(fallback)
            resolution_meta["source_by_key"][key] = "builtin"
    return resolved, resolution_meta


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model(model_path: pathlib.Path, device: torch.device) -> PairTransformer:
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    hparams = data.get("hparams", {})
    model = PairTransformer(
        feat_dim        = F,
        d_model         = hparams.get("d_model",         64),
        nhead           = hparams.get("nhead",             2),
        num_layers      = hparams.get("nlayers", hparams.get("num_layers", 3)),
        dim_feedforward = hparams.get("ffn_dim", hparams.get("dim_feedforward", 256)),
        dropout         = 0.0,  # 推理阶段关闭 dropout
    )
    missing, unexpected = model.load_state_dict(data["model_state"], strict=False)
    allowed_missing = {k for k in missing if k.startswith("cls_head.")}
    unexpected = list(unexpected)
    if unexpected or (set(missing) - allowed_missing):
        raise RuntimeError(
            f"模型权重与当前结构不兼容: missing={list(missing)} unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def _to_tensor(feat_dict: dict[str, float], device: torch.device) -> torch.Tensor:
    """将特征字典转为 (1, F) tensor。"""
    arr = np.array([feat_dict.get(c, 0.0) for c in NON_TIME_COLS], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


# ── 单次推断 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_score(
    query_feat:    dict[str, float],
    query_variant: str,
    anchors:       list[dict[str, Any]],   # anchor records with score_gt + features
    model:         PairTransformer,
    device:        torch.device,
    tie_gate_threshold: float = DEFAULT_TIE_GATE_THRESHOLD,
    tie_shrink_power: float = DEFAULT_TIE_SHRINK_POWER,
    tie_margin_weight_alpha: float = DEFAULT_TIE_MARGIN_WEIGHT_ALPHA,
    min_anchor_quality: float = DEFAULT_MIN_ANCHOR_QUALITY,
    outlier_mad_scale: float = DEFAULT_OUTLIER_MAD_SCALE,
    outlier_min_delta: float = DEFAULT_OUTLIER_MIN_DELTA,
) -> dict[str, Any]:
    """
    利用参考锚点法估算单程序优化分数。

    S_query_k = S_{anchor_k} + model(query, anchor_k)
    final_S   = mean over all available anchors

    返回：{score_log, score_100, band, anchor_details}
    """
    xi = _to_tensor(query_feat, device)

    anchor_estimates = []
    for anc in anchors:
        xj = _to_tensor({c: float(anc.get(c, 0.0)) for c in NON_TIME_COLS}, device)
        pred_log_ratio, cls_logits = model.forward_with_aux(xi, xj)
        pred_log_ratio = float(pred_log_ratio.cpu().item())
        decoded = _decode_pair_log_ratio(
            pred_log_ratio,
            cls_logits,
            tie_gate_threshold=tie_gate_threshold,
            tie_shrink_power=tie_shrink_power,
        )

        anchor_quality = float(anc.get("anchor_quality", 1.0) or 0.0)
        distance_weight = _variant_distance_weight(query_variant, str(anc.get("variant", "")))
        effective_confidence, class_confidence, direction_margin = _pair_vote_confidence(
            list(decoded["class_probs"]),
            str(decoded["decoded_class"]),
            tie_margin_weight_alpha=tie_margin_weight_alpha,
        )
        base_weight = anchor_quality * distance_weight * effective_confidence
        if anchor_quality < min_anchor_quality:
            base_weight = 0.0

        gated_log_ratio = float(decoded["gated_log_ratio"])
        s_est = float(anc["score_gt"]) + gated_log_ratio
        anchor_estimates.append({
            "anchor_variant":    anc["variant"],
            "anchor_score_gt":   round(float(anc["score_gt"]), 4),
            "anchor_quality":    round(anchor_quality, 4),
            "distance_weight":   round(distance_weight, 4),
            "class_confidence":  round(class_confidence, 4),
            "effective_confidence": round(effective_confidence, 4),
            "direction_margin":  round(direction_margin, 4),
            "tie_prob":          round(float(decoded["tie_prob"]), 4),
            "decoded_class":     str(decoded["decoded_class"]),
            "model_log_ratio":   round(pred_log_ratio, 4),
            "gated_log_ratio":   round(gated_log_ratio, 4),
            "score_estimate_raw": round(s_est, 6),
            "score_estimate":    round(s_est, 4),
            "weight":            round(base_weight, 6),
            "used":              base_weight > 0.0,
        })

    if not anchor_estimates:
        return {"score_log": 0.0, "score_100": 50.0, "band": "medium",
                "anchor_details": []}

    anchor_estimates = _filter_anchor_estimates(
        anchor_estimates,
        mad_scale=outlier_mad_scale,
        min_delta=outlier_min_delta,
    )
    kept = [e for e in anchor_estimates if e.get("used")]
    if not kept:
        kept = anchor_estimates

    weights = np.array([float(e["weight"]) for e in kept], dtype=np.float64)
    estimates = np.array([float(e["score_estimate_raw"]) for e in kept], dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
        final_score = float(estimates.mean())
    else:
        final_score = float(np.average(estimates, weights=weights))

    return {
        "score_log":     round(final_score, 4),
        "anchor_details": anchor_estimates,
        "n_anchors_used": len(kept),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="单程序优化分数推断（锚点法 · Fixed Work）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default="train_set/model_transformer.pt")
    parser.add_argument("--anchor-set", default="train_set/anchor_set.parquet")
    parser.add_argument("--zscore",     default="train_set/run_features_zscore.parquet")
    parser.add_argument("--output",     default="train_set/scores.parquet")
    parser.add_argument("--eval-output", default=None,
                        help="评估 JSON 输出路径；默认写到 train_set/score_eval.json")
    parser.add_argument("--tuned-defaults-json", default=DEFAULT_TUNED_DEFAULTS_JSON,
                        help="按 variant 覆盖评分默认值的 tuned JSON；CLI 显式传参优先")
    parser.add_argument("--tuned-selection-objective", choices=["score", "time"],
                        default=DEFAULT_TUNED_SELECTION_OBJECTIVE,
                        help="从 tuned JSON 中选择 score-first 还是 time-first 的最优参数")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--tie-gate-threshold", type=float, default=None,
                        help="辅助分类头判为 tie 时将回归输出压到 0 的阈值")
    parser.add_argument("--tie-shrink-power", type=float, default=None,
                        help="按 (1 - p_tie)^power 缩放回归输出的幂次")
    parser.add_argument("--tie-margin-weight-alpha", type=float, default=None,
                        help="对非 tie pair，将锚点投票权重从纯 class confidence 向 direction margin 混合的比例")
    parser.add_argument("--min-anchor-quality", type=float, default=None,
                        help="低于该质量分数的锚点不参与聚合")
    parser.add_argument("--anchor-outlier-mad-scale", type=float, default=None,
                        help="按中位数绝对偏差过滤锚点估计时的倍率")
    parser.add_argument("--anchor-outlier-min-delta", type=float, default=None,
                        help="锚点估计最小离群剔除阈值")
    parser.add_argument("--program",    default=None, help="只对指定程序打印诊断报告")
    parser.add_argument("--variant",    default=None, help="与 --program 配合使用")
    args = parser.parse_args()

    model_path  = (REPO_ROOT / args.model).resolve()
    anchor_path = (REPO_ROOT / args.anchor_set).resolve()
    zscore_path = (REPO_ROOT / args.zscore).resolve()
    out_path    = (REPO_ROOT / args.output).resolve()
    tuned_defaults_path = (REPO_ROOT / args.tuned_defaults_json).resolve()

    for p in (model_path, anchor_path, zscore_path):
        if not p.exists():
            sys.exit(f"[错误] 找不到 {p}\n  提示：先运行 build_anchor_set.py")

    device = select_device(args.device)
    model  = load_model(model_path, device)
    tuned_defaults = _load_tuned_variant_defaults(
        tuned_defaults_path,
        selection_objective=args.tuned_selection_objective,
    )
    cli_overrides = {
        "tie_gate_threshold": args.tie_gate_threshold,
        "tie_shrink_power": args.tie_shrink_power,
        "tie_margin_weight_alpha": args.tie_margin_weight_alpha,
        "min_anchor_quality": args.min_anchor_quality,
        "anchor_outlier_mad_scale": args.anchor_outlier_mad_scale,
        "anchor_outlier_min_delta": args.anchor_outlier_min_delta,
    }

    df_anchors = pd.read_parquet(anchor_path)
    df_queries = pd.read_parquet(zscore_path)
    if DROPPED_INPUT_FEATURES:
        print(f"[info] 已从评分输入中剔除死特征: {', '.join(DROPPED_INPUT_FEATURES)}")
    if tuned_defaults:
        available_variants = ", ".join(sorted(tuned_defaults.keys()))
        print(
            f"[info] 已加载 variant 默认参数: {available_variants} "
            f"(objective={args.tuned_selection_objective})"
        )

    # 读取 anchor stats（用于 0-100 归一化）
    stats_path = anchor_path.with_suffix(".stats.json")
    anchor_stats: dict = {}
    if stats_path.exists():
        anchor_stats = json.loads(stats_path.read_text())

    # 所有 score_gt 用于百分位归一化
    all_scores = df_anchors["score_gt"].values.astype(np.float64)

    # 建立 anchor 查找：program → list[anchor_record]
    anchor_map: dict[str, list[dict]] = {}
    for _, row in df_anchors.iterrows():
        prog = row["program"]
        anchor_map.setdefault(prog, []).append(row.to_dict())

    print(f"\n{'='*60}")
    print("  单程序优化分数推断（Fixed Work · 参考锚点法）")
    print(f"{'='*60}")
    print(f"  锚点数: {len(df_anchors)}  |  待评分 runs: {len(df_queries)}\n")

    records: list[dict] = []

    for _, qrow in df_queries.iterrows():
        prog    = qrow["program"]
        variant = qrow["variant"]

        # 提取 query z-score 特征
        query_feat = {c: float(qrow.get(c, 0.0)) for c in NON_TIME_COLS}

        # 获取该程序的锚点（排除 query 自身）
        anchors = [
            a for a in anchor_map.get(prog, [])
            if a["variant"] != variant
        ]

        # 若无任何外部锚点（如该程序只有一个变体），退化为使用 O0
        if not anchors:
            anchors = anchor_map.get(prog, [])

        scoring_params, scoring_param_meta = _resolve_scoring_params(
            query_variant=variant,
            cli_overrides=cli_overrides,
            tuned_defaults=tuned_defaults,
        )

        # 推断分数
        result = predict_score(
            query_feat,
            query_variant=variant,
            anchors=anchors,
            model=model,
            device=device,
            tie_gate_threshold=scoring_params["tie_gate_threshold"],
            tie_shrink_power=scoring_params["tie_shrink_power"],
            tie_margin_weight_alpha=scoring_params["tie_margin_weight_alpha"],
            min_anchor_quality=scoring_params["min_anchor_quality"],
            outlier_mad_scale=scoring_params["anchor_outlier_mad_scale"],
            outlier_min_delta=scoring_params["anchor_outlier_min_delta"],
        )

        score_log = result["score_log"]
        score_100 = _percentile_score(score_log, all_scores)
        band      = _band(score_100)

        # 瓶颈归因
        bottlenecks = _bottleneck_scores(query_feat)

        rec = {
            "program":     prog,
            "variant":     variant,
            "score_log":   score_log,
            "score_100":   round(score_100, 1),
            "band":        band,
            "n_anchors":   len(anchors),
            "n_anchors_used": int(result.get("n_anchors_used", len(anchors))),
            "scoring_param_resolution": json.dumps(scoring_param_meta, ensure_ascii=False),
            # 真值（用于评估）
            "score_gt":    float(
                df_anchors.loc[
                    (df_anchors["program"] == prog) &
                    (df_anchors["variant"] == variant),
                    "score_gt",
                ].values[0]
            ) if len(
                df_anchors.loc[
                    (df_anchors["program"] == prog) &
                    (df_anchors["variant"] == variant)
                ]
            ) > 0 else float("nan"),
            # top 瓶颈
            "top_bottleneck":       bottlenecks[0]["category"] if bottlenecks else "",
            "top_bottleneck_sev":   bottlenecks[0]["severity"] if bottlenecks else 0.0,
            "second_bottleneck":    bottlenecks[1]["category"] if len(bottlenecks) > 1 else "",
        }
        records.append(rec)

    df_scores = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_parquet(out_path, index=False)

    # ── 汇总评估（预测分 vs 真值分）──
    valid = df_scores.dropna(subset=["score_gt"])
    mae   = float(np.abs(valid["score_log"] - valid["score_gt"]).mean())
    corr  = float(valid["score_log"].corr(valid["score_gt"]))

    # 方向准确率（预测分 > 0 ↔ 真值分 > 0，即是否比 O0 更优）
    dir_correct = ((valid["score_log"] > 0) == (valid["score_gt"] > 0)).mean()

    # 档位准确率（预测档位 == 真值档位）
    valid = valid.copy()
    valid["band_gt"] = valid["score_gt"].apply(
        lambda s: _band(_percentile_score(s, all_scores))
    )
    band_acc = (valid["band"] == valid["band_gt"]).mean()

    eval_report = {
        "n_runs":          len(df_scores),
        "n_with_gt":       len(valid),
        "mae_score_log":   round(mae,      4),
        "corr_score_log":  round(corr,     4),
        "dir_accuracy":    round(float(dir_correct), 4),
        "band_accuracy":   round(float(band_acc),    4),
    }
    eval_path = pathlib.Path(args.eval_output).resolve() \
        if args.eval_output else (REPO_ROOT / "train_set" / "score_eval.json").resolve()
    eval_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False))

    # ── 打印汇总 ──
    sep = "─" * 58
    print(f"  {'program':20s} {'var':4s}  {'score_log':>9}  {'score_100':>9}  band")
    print(f"  {sep}")
    for _, row in df_scores.sort_values(["program", "variant"]).head(24).iterrows():
        gt_str = f" (gt={row['score_gt']:+.3f})" if not math.isnan(row["score_gt"]) else ""
        print(f"  {row['program']:20s} {row['variant']:4s}  "
              f"{row['score_log']:+9.3f}  {row['score_100']:9.1f}  "
              f"{row['band']:6s}{gt_str}")
    if len(df_scores) > 24:
        print(f"  ... （共 {len(df_scores)} 条，仅显示前 24）")

    print(f"\n{'='*58}")
    print("  评估汇总")
    print(f"  {'MAE (log-score)':25s} {eval_report['mae_score_log']:.4f}")
    print(f"  {'Pearson r':25s} {eval_report['corr_score_log']:.4f}")
    print(f"  {'方向准确率':25s} {eval_report['dir_accuracy']:.4f}")
    print(f"  {'档位准确率':25s} {eval_report['band_accuracy']:.4f}")
    print(f"{'='*58}\n")

    print(f"[ok] 分数表:   {out_path}")
    print(f"[ok] 评估报告: {eval_path}")

    # ── 若指定了 --program，打印完整诊断卡片 ──
    if args.program:
        variant = args.variant or "O3"
        qrow_matches = df_queries[
            (df_queries["program"] == args.program) &
            (df_queries["variant"] == variant)
        ]
        if qrow_matches.empty:
            print(f"\n[警告] 未找到 {args.program} / {variant}")
            return

        qrow = qrow_matches.iloc[0]
        query_feat  = {c: float(qrow.get(c, 0.0)) for c in NON_TIME_COLS}
        anchors     = [a for a in anchor_map.get(args.program, []) if a["variant"] != variant]
        scoring_params = _resolve_scoring_params(
            query_variant=variant,
            cli_overrides=cli_overrides,
            tuned_defaults=tuned_defaults,
        )
        result      = predict_score(
            query_feat,
            query_variant=variant,
            anchors=anchors,
            model=model,
            device=device,
            tie_gate_threshold=scoring_params["tie_gate_threshold"],
            tie_shrink_power=scoring_params["tie_shrink_power"],
            tie_margin_weight_alpha=scoring_params["tie_margin_weight_alpha"],
            min_anchor_quality=scoring_params["min_anchor_quality"],
            outlier_mad_scale=scoring_params["anchor_outlier_mad_scale"],
            outlier_min_delta=scoring_params["anchor_outlier_min_delta"],
        )
        score_log   = result["score_log"]
        score_100   = _percentile_score(score_log, all_scores)
        band_label  = _band(score_100)
        bottlenecks = _bottleneck_scores(query_feat)

        print(f"\n{'='*60}")
        print(f"  诊断卡片：{args.program}  /  {variant}")
        print(f"{'='*60}")
        print(
            "  实际评分参数      : "
            f"gate={scoring_params['tie_gate_threshold']:.3f}  "
            f"shrink={scoring_params['tie_shrink_power']:.3f}  "
            f"alpha={scoring_params['tie_margin_weight_alpha']:.3f}"
        )
        print(f"  优化分数 (log)  : {score_log:+.3f}")
        print(f"  优化分数 (0-100): {score_100:.1f}")
        print(f"  优化档位        : {band_label}")
        print(f"  使用锚点数      : {result.get('n_anchors_used', len(anchors))} / {len(anchors)}")
        print(f"\n  锚点比较明细：")
        for anc in result["anchor_details"]:
            used_mark = "*" if anc.get("used") else "x"
            print(f"    [{used_mark}] vs {anc['anchor_variant']:3s}  "
                  f"raw={anc['model_log_ratio']:+.4f}  gated={anc['gated_log_ratio']:+.4f}  "
                  f"cls={anc['decoded_class']:8s}  p_tie={anc['tie_prob']:.3f}  "
                  f"w={anc['weight']:.3f}  anchor_S={anc['anchor_score_gt']:+.4f}  "
                  f"→ score_est={anc['score_estimate']:+.4f}")
        print(f"\n  瓶颈归因（Top 3）：")
        for bt in bottlenecks[:3]:
            top_metrics = sorted(bt["support_metrics"].items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:3]
            metrics_str = "  ".join(f"{k}={v:+.2f}" for k, v in top_metrics)
            print(f"    [{bt['rank']}] {bt['category']:12s}  severity={bt['severity']:.3f}"
                  f"   {metrics_str}")

        # ── 热点窗口证据 ──────────────────────────────────────────────────
        output_dir = str(qrow.get("output_dir", ""))
        hotspot_windows = _load_hotspot_windows(output_dir, top_k=5)
        if hotspot_windows:
            print(f"\n  热点窗口证据（Top {len(hotspot_windows)}，按热点分数降序）：")
            print(f"    {'win_id':>6}  {'ipc':>6}  {'llc_mpki':>9}  "
                  f"{'dtlb_mpki':>10}  {'faults':>7}  {'hs_score':>9}")
            for hw in hotspot_windows:
                print(f"    {hw['window_id']:6d}  {hw['ipc']:6.3f}  "
                      f"{hw['llc_mpki']:9.2f}  {hw['dtlb_mpki']:10.2f}  "
                      f"{hw['faults']:7d}  {hw['hotspot_score']:9.2f}")
        else:
            print(f"\n  [info] 未找到原始窗口数据: {output_dir or '(output_dir 不可用)'}")

        # ── 阶段特征摘要（warmup vs steady）──────────────────────────────
        phase_keys = [
            ("warmup_ipc", "steady_ipc", "phase_ipc_ratio", "IPC"),
            ("warmup_llc_mpki", "steady_llc_mpki", "phase_llc_ratio", "LLC_MPKI"),
        ]
        phase_avail = all(k in qrow.index for k in ["warmup_ipc", "steady_ipc"])
        if phase_avail:
            print(f"\n  阶段特征（warmup vs steady-state，z-score）：")
            for wk, sk, rk, label in phase_keys:
                wv = float(qrow.get(wk, 0.0))
                sv = float(qrow.get(sk, 0.0))
                rv = float(qrow.get(rk, 0.0))
                print(f"    {label:12s}  warmup={wv:+.3f}  steady={sv:+.3f}  "
                      f"ratio={rv:+.3f}")

        print("")


if __name__ == "__main__":
    main()
