# score-first vs time-first 默认口径对比

> 生成时间：2026-05-07 09:42:55 UTC  
> 生成脚本：scripts/compare_selection_objectives.py

## 结论

当前建议默认口径：**score-first**。

- proxy Pearson r 提升 +0.0011
- proxy MAE 改善 +0.0030
- strict time Pearson r 仅变化 -0.0016
- strict time Spearman 仅变化 -0.0018

## 一页总表

| 指标 | score-first | time-first | score-first - time-first | 更优口径 |
| --- | ---: | ---: | ---: | --- |
| proxy.corr_score_log | 0.8996 | 0.8985 | +0.0011 更好 | score-first |
| proxy.mae_score_log | 0.3174 | 0.3204 | -0.0030 更好 | score-first |
| proxy.dir_accuracy | 0.7567 | 0.7567 | +0.0000 | tie |
| proxy.band_accuracy | 0.8075 | 0.8075 | +0.0000 | tie |
| time.corr_model_time | 0.4321 | 0.4337 | -0.0016 | time-first |
| time.spearman_model | 0.5174 | 0.5192 | -0.0018 | time-first |
| time.mae_model_time | 1.0031 | 0.9993 | +0.0038 | time-first |
| time.dir_acc_model | 0.8632 | 0.8632 | +0.0000 | tie |
| time.band_acc_model | 0.6769 | 0.6837 | -0.0068 | time-first |
| coverage.n_valid_strict | 294.0000 | 294.0000 | +0.0000 | tie |

## ALL 共享参数对比

| 参数 | score-first | time-first |
| --- | ---: | ---: |
| tie_gate_threshold | 0.60 | 0.50 |
| tie_shrink_power | 0.75 | 0.75 |
| tie_margin_weight_alpha | 0.15 | 0.50 |
| min_anchor_quality | 0.30 | 0.30 |
| anchor_outlier_mad_scale | 3.00 | 3.00 |
| anchor_outlier_min_delta | 0.35 | 0.35 |

## Variant-local tuned 可靠性

| 口径 | variant | reliable | reason | n_score_valid | n_time_valid | score_corr | time_corr | gate | shrink | alpha |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score_first | O0 | no | score_corr_not_finite | 128 | 104 | - | - | 0.50 | 0.75 | 0.50 |
| score_first | O1 | no | n_score_valid<32 | 0 | 96 | - | -0.0876 | 0.50 | 0.75 | 0.50 |
| score_first | O2 | yes | ok | 124 | 95 | 0.9112 | 0.2262 | 0.60 | 1.25 | 0.15 |
| score_first | O3 | yes | ok | 122 | 95 | 0.9406 | 0.1704 | 0.55 | 1.25 | 0.15 |
| time_first | O0 | no | score_corr_not_finite | 128 | 104 | - | - | 0.50 | 0.75 | 0.50 |
| time_first | O1 | no | n_score_valid<32 | 0 | 96 | - | -0.0876 | 0.50 | 0.75 | 0.50 |
| time_first | O2 | yes | ok | 124 | 95 | 0.9095 | 0.2301 | 0.60 | 1.25 | 0.50 |
| time_first | O3 | yes | ok | 122 | 95 | 0.9391 | 0.1729 | 0.55 | 1.25 | 0.50 |

## 解释

score-first 看的是单程序评分对 proxy 真值的恢复能力；time-first 看的是 strict 时间外部验证。当前这两套口径的时间指标差距很小，但 score-first 在 proxy 侧更稳，因此默认更适合作为主线口径。

## 复现命令

```bash
/home/ssy/mem-profiler/.venv/bin/python scripts/compare_selection_objectives.py --device cpu
```

