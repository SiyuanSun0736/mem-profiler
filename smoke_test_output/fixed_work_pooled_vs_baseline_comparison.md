# Fixed-Work Bucket Pooled Token vs Baseline Comparison

Generated: 2026-05-20 05:17:28 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkBucketPooledTokenPairTransformer (smoke_test_output/model_fixed_work_token_pooled_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkPooled | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 1.1622 | 0.5944 |
| rmse | 0.8685 | 1.5991 | 0.7306 |
| r2 | 0.8069 | 0.3455 | -0.4614 |
| dir_acc | 0.9020 | 0.6814 | -0.2206 |
| acc_3cls | 0.7958 | 0.5917 | -0.2041 |
| aux_acc_3cls | 0.8417 | 0.4375 | -0.4042 |
| aux_tie_recall | 0.6667 | 0.1667 | -0.5000 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.1667 |
| near_tie | 40 | 0.7000 | 0.5250 | -0.1750 | - | - |
| O2-O3 | 40 | 0.6667 | 0.4444 | -0.2223 | 0.5455 | 0.2273 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.1111 | -0.2778 | 0.6667 | 0.1667 | -0.5000 |
| near_tie | 0.6500 | 0.5250 | -0.1250 | 0.5750 | 0.4500 | -0.1250 |
| O2-O3 | 0.4750 | 0.2750 | -0.2000 | 0.5750 | 0.3000 | -0.2750 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
