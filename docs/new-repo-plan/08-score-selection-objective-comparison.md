# score-first / time-first / time-aware 默认口径对比

> 生成时间：2026-05-11 14:39:48 UTC
> 生成脚本：scripts/compare_selection_objectives.py

## 结论

当前建议默认口径：**score-first**。

- proxy Pearson r 提升 +0.0026
- proxy MAE 改善 +0.0072
- strict time Pearson r 仅变化 -0.0012
- strict time Spearman 仅变化 +0.0000

time-aware 是可选的外部时间优先口径：strict time Pearson 相对 score-first +0.0013 更好，MAE -0.0069 更好，repeat-backed Pearson +0.0064 更好；代价是 proxy Pearson -0.0013。

## 一页总表

| 指标 | score-first | time-first | time-aware | score-first - time-first | time-aware - score-first | score/time 更优 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| proxy.corr_score_log | 0.9072 | 0.9046 | 0.9059 | +0.0026 更好 | -0.0013 | score-first |
| proxy.mae_score_log | 0.2723 | 0.2795 | 0.2766 | -0.0072 更好 | +0.0043 | score-first |
| proxy.dir_accuracy | 0.7807 | 0.7807 | 0.7834 | +0.0000 | +0.0027 更好 | tie |
| proxy.band_accuracy | 0.8048 | 0.7995 | 0.8021 | +0.0053 更好 | -0.0027 | score-first |
| time.corr_model_time | 0.3980 | 0.3991 | 0.3993 | -0.0012 | +0.0013 更好 | time-first |
| time.spearman_model | 0.5220 | 0.5220 | 0.5233 | +0.0000 更好 | +0.0013 更好 | score-first |
| time.mae_model_time | 1.0947 | 1.0878 | 1.0878 | +0.0069 | -0.0069 更好 | time-first |
| time.dir_acc_model | 0.8583 | 0.8583 | 0.8583 | +0.0000 | +0.0000 | tie |
| time.band_acc_model | 0.6150 | 0.6150 | 0.6150 | +0.0000 | +0.0000 | tie |
| coverage.n_valid_strict | 361.0000 | 361.0000 | 361.0000 | +0.0000 | +0.0000 | tie |

## ALL 共享参数对比

| 参数 | score-first | time-first | time-aware |
| --- | ---: | ---: | ---: |
| tie_gate_threshold | 0.62 | 0.60 | 0.60 |
| tie_shrink_power | 0.65 | 1.25 | 0.65 |
| tie_margin_weight_alpha | 0.10 | 0.55 | 0.55 |
| min_anchor_quality | 0.30 | 0.25 | 0.25 |
| anchor_outlier_mad_scale | 2.50 | 2.50 | 2.50 |
| anchor_outlier_min_delta | 0.35 | 0.35 | 0.35 |

## time-aware 结果

time-aware proxy: corr=0.9059, MAE=0.2766, band=0.8021.
time-aware strict time: corr=0.3993, spearman=0.5233, MAE=1.0878, band=0.6150.
time-aware repeat-backed: n=75, corr=0.7945, spearman=0.7376.

## Variant-local tuned 可靠性

| 口径 | variant | reliable | reason | n_score_valid | n_time_valid | score_corr | time_corr | gate | shrink | alpha |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score_first | O0 | no | score_corr_not_finite | 128 | 121 | - | - | 0.62 | 0.65 | 0.55 |
| score_first | O1 | no | n_score_valid<32 | 0 | 120 | - | -0.0077 | 0.55 | 1.25 | 0.55 |
| score_first | O2 | yes | ok | 124 | 120 | 0.9136 | 0.2042 | 0.62 | 0.65 | 0.10 |
| score_first | O3 | yes | ok | 122 | 120 | 0.9502 | 0.1855 | 0.60 | 1.25 | 0.10 |
| time_first | O0 | no | score_corr_not_finite | 128 | 121 | - | - | 0.62 | 0.65 | 0.55 |
| time_first | O1 | no | n_score_valid<32 | 0 | 120 | - | -0.0077 | 0.55 | 1.25 | 0.55 |
| time_first | O2 | yes | ok | 124 | 120 | 0.9121 | 0.2072 | 0.60 | 1.25 | 0.55 |
| time_first | O3 | yes | ok | 122 | 120 | 0.9445 | 0.1891 | 0.60 | 1.25 | 0.55 |
| time_aware | O0 | no | score_corr_not_finite | 128 | 121 | - | - | 0.62 | 0.65 | 0.55 |
| time_aware | O1 | no | n_score_valid<32 | 0 | 120 | - | -0.0080 | 0.55 | 1.25 | 0.55 |
| time_aware | O2 | yes | ok | 124 | 120 | 0.9112 | 0.2051 | 0.60 | 1.25 | 0.55 |
| time_aware | O3 | yes | ok | 122 | 120 | 0.9486 | 0.1872 | 0.48 | 1.25 | 0.55 |

## 解释

score-first 看的是单程序评分对 proxy 真值的恢复能力；time-first 看的是 strict 时间外部验证；time-aware 同时纳入 proxy、strict time、band 和 repeat-backed 子集。当前 score-first 仍更适合作为默认主线口径；若一次实验更看重真实时间外部一致性，可以显式切到 time-aware。

## 复现命令

```bash
/home/ssy/mem-profiler/.venv/bin/python scripts/compare_selection_objectives.py --device cpu
```
