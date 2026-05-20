# Hotspot Token vs Baseline Comparison

Generated: 2026-05-19 17:06:24 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: HotspotWindowTokenPairTransformer (train_set/tmp_hotspot_token_smoke/model_hotspot_token_transformer.pt)

## Overall Test

| Metric | Baseline | HotspotToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 1.5287 | 0.9609 |
| rmse | 0.8685 | 2.0703 | 1.2018 |
| r2 | 0.8069 | -0.0970 | -0.9039 |
| dir_acc | 0.9020 | 0.6029 | -0.2991 |
| acc_3cls | 0.7958 | 0.4917 | -0.3041 |
| aux_acc_3cls | 0.8417 | 0.4292 | -0.4125 |
| aux_tie_recall | 0.6667 | 0.3056 | -0.3611 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.3056 |
| near_tie | 40 | 0.7000 | 0.5250 | -0.1750 | - | - |
| O2-O3 | 40 | 0.6667 | 0.6667 | 0.0000 | 0.5455 | 0.2727 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.0000 | -0.3889 | 0.6667 | 0.3056 | -0.3611 |
| near_tie | 0.6500 | 0.5000 | -0.1500 | 0.5750 | 0.3500 | -0.2250 |
| O2-O3 | 0.4750 | 0.2750 | -0.2000 | 0.5750 | 0.3750 | -0.2000 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
