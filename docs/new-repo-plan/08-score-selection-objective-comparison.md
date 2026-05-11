# score-first vs time-first 默认口径对比

> 生成时间：2026-05-11 13:13:49 UTC
> 生成脚本：scripts/compare_selection_objectives.py

## 结论

当前建议默认口径：**score-first**。

- proxy Pearson r 提升 +0.0011
- proxy MAE 改善 +0.0037
- strict time Pearson r 仅变化 -0.0008
- strict time Spearman 仅变化 -0.0010

## 一页总表

| 指标 | score-first | time-first | score-first - time-first | 更优口径 |
| --- | ---: | ---: | ---: | --- |
| proxy.corr_score_log | 0.9072 | 0.9061 | +0.0011 更好 | score-first |
| proxy.mae_score_log | 0.2726 | 0.2763 | -0.0037 更好 | score-first |
| proxy.dir_accuracy | 0.7807 | 0.7834 | -0.0027 | time-first |
| proxy.band_accuracy | 0.8021 | 0.7995 | +0.0026 更好 | score-first |
| time.corr_model_time | 0.3982 | 0.3990 | -0.0008 | time-first |
| time.spearman_model | 0.5219 | 0.5230 | -0.0010 | time-first |
| time.mae_model_time | 1.0935 | 1.0905 | +0.0030 | time-first |
| time.dir_acc_model | 0.8583 | 0.8583 | +0.0000 | tie |
| time.band_acc_model | 0.6150 | 0.6150 | +0.0000 | tie |
| coverage.n_valid_strict | 361.0000 | 361.0000 | +0.0000 | tie |

## ALL 共享参数对比

| 参数 | score-first | time-first |
| --- | ---: | ---: |
| tie_gate_threshold | 0.62 | 0.48 |
| tie_shrink_power | 0.65 | 0.65 |
| tie_margin_weight_alpha | 0.10 | 0.55 |
| min_anchor_quality | 0.30 | 0.25 |
| anchor_outlier_mad_scale | 3.50 | 2.50 |
| anchor_outlier_min_delta | 0.35 | 0.35 |

## Variant-local tuned 可靠性

| 口径 | variant | reliable | reason | n_score_valid | n_time_valid | score_corr | time_corr | gate | shrink | alpha |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score_first | O0 | no | score_corr_not_finite | 128 | 104 | - | - | 0.48 | 0.65 | 0.55 |
| score_first | O1 | no | n_score_valid<32 | 0 | 96 | - | -0.0804 | 0.48 | 0.65 | 0.55 |
| score_first | O2 | yes | ok | 124 | 95 | 0.9115 | 0.2257 | 0.60 | 1.25 | 0.10 |
| score_first | O3 | yes | ok | 122 | 95 | 0.9426 | 0.1721 | 0.62 | 1.25 | 0.10 |
| time_first | O0 | no | score_corr_not_finite | 128 | 104 | - | - | 0.48 | 0.65 | 0.55 |
| time_first | O1 | no | n_score_valid<32 | 0 | 96 | - | -0.0804 | 0.48 | 0.65 | 0.55 |
| time_first | O2 | yes | ok | 124 | 95 | 0.9091 | 0.2308 | 0.62 | 1.25 | 0.55 |
| time_first | O3 | yes | ok | 122 | 95 | 0.9407 | 0.1757 | 0.55 | 1.25 | 0.55 |

## 解释

score-first 看的是单程序评分对 proxy 真值的恢复能力；time-first 看的是 strict 时间外部验证。当前这两套口径的时间指标差距很小，但 score-first 在 proxy 侧更稳，因此默认更适合作为主线口径。

## 复现命令

```bash
/home/ssy/mem-profiler/.venv/bin/python scripts/compare_selection_objectives.py --device cpu
```
