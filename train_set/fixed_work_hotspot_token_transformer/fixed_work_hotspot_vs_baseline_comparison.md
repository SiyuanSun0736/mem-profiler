# Fixed-Work + Hotspot Token vs Baseline Comparison

Generated: 2026-05-20 05:34:06 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkHotspotTokenPairTransformer (train_set/fixed_work_hotspot_token_transformer/model_fixed_work_hotspot_token_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkHotspot | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 0.6237 | 0.0559 |
| rmse | 0.8685 | 0.9519 | 0.0834 |
| r2 | 0.8069 | 0.7681 | -0.0388 |
| dir_acc | 0.9020 | 0.8971 | -0.0049 |
| acc_3cls | 0.7958 | 0.8000 | 0.0042 |
| aux_acc_3cls | 0.8417 | 0.8333 | -0.0084 |
| aux_tie_recall | 0.6667 | 0.6667 | 0.0000 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.6667 |
| near_tie | 40 | 0.7000 | 0.6750 | -0.0250 | - | - |
| O2-O3 | 40 | 0.6667 | 0.6111 | -0.0556 | 0.5455 | 0.5000 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.3889 | 0.0000 | 0.6667 | 0.6667 | 0.0000 |
| near_tie | 0.6500 | 0.6250 | -0.0250 | 0.5750 | 0.5500 | -0.0250 |
| O2-O3 | 0.4750 | 0.4500 | -0.0250 | 0.5750 | 0.5750 | 0.0000 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
