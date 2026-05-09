# score-first vs time-first 默认口径对比

> 生成时间：2026-05-07 12:13:34 UTC  
> 生成脚本：scripts/compare_selection_objectives.py

## 结论

当前建议默认口径：**score-first**。

- proxy Pearson r 提升 +0.0015
- proxy MAE 改善 +0.0034
- strict time Pearson r 仅变化 -0.0024
- strict time Spearman 仅变化 -0.0028

## 一页总表

| 指标 | score-first | time-first | score-first - time-first | 更优口径 |
| --- | ---: | ---: | ---: | --- |
| proxy.corr_score_log | 0.9005 | 0.8990 | +0.0015 更好 | score-first |
| proxy.mae_score_log | 0.3160 | 0.3194 | -0.0034 更好 | score-first |
| proxy.dir_accuracy | 0.7567 | 0.7567 | +0.0000 | tie |
| proxy.band_accuracy | 0.8075 | 0.8075 | +0.0000 | tie |
| time.corr_model_time | 0.4325 | 0.4349 | -0.0024 | time-first |
| time.spearman_model | 0.5181 | 0.5209 | -0.0028 | time-first |
| time.mae_model_time | 1.0025 | 0.9975 | +0.0050 | time-first |
| time.dir_acc_model | 0.8632 | 0.8632 | +0.0000 | tie |
| time.band_acc_model | 0.6769 | 0.6837 | -0.0068 | time-first |
| coverage.n_valid_strict | 294.0000 | 294.0000 | +0.0000 | tie |

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

这份对比有一个边界要说清楚：它比较的是同一套评分链路上两种 tuned-default selection objective，而不是在重选锚点策略。

当前默认底座已经是 P4 的 O0/O2/O3 加权锚点，而不是旧的 O0/O3 简单平均：

- anchor set 默认使用 O0/O2/O3。
- 聚合时会同时使用 anchor_quality、variant 距离权重和辅助分类置信度。
- 同一程序的多个 anchor estimate 会先做中位数离群过滤，再进入最终聚合。

因此，这里 score-first 相比 time-first 的差异，主要来自 P3 评分层 tuned 参数与回退策略的选择，而不是推翻或替换了 P4 锚点策略。更准确地说：P4 提供的是当前单程序评分成立的公共底座，08 里的对比是在这个底座之上比较默认口径该优先偏向 proxy 恢复还是 strict 时间外部验证。

## 复现命令

```bash
/home/ssy/mem-profiler/.venv/bin/python scripts/compare_selection_objectives.py --device cpu
```

