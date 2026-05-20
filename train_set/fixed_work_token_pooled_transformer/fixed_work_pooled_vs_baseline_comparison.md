# Fixed-Work Bucket Pooled Token vs Baseline Comparison

Generated: 2026-05-20 05:19:47 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkBucketPooledTokenPairTransformer (train_set/fixed_work_token_pooled_transformer/model_fixed_work_token_pooled_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkPooled | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 0.6479 | 0.0801 |
| rmse | 0.8685 | 0.9895 | 0.1210 |
| r2 | 0.8069 | 0.7494 | -0.0575 |
| dir_acc | 0.9020 | 0.8578 | -0.0442 |
| acc_3cls | 0.7958 | 0.7292 | -0.0666 |
| aux_acc_3cls | 0.8417 | 0.7958 | -0.0459 |
| aux_tie_recall | 0.6667 | 0.6389 | -0.0278 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.6389 |
| near_tie | 40 | 0.7000 | 0.6750 | -0.0250 | - | - |
| O2-O3 | 40 | 0.6667 | 0.8889 | 0.2222 | 0.5455 | 0.5000 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.1944 | -0.1945 | 0.6667 | 0.6389 | -0.0278 |
| near_tie | 0.6500 | 0.6000 | -0.0500 | 0.5750 | 0.4000 | -0.1750 |
| O2-O3 | 0.4750 | 0.4250 | -0.0500 | 0.5750 | 0.5500 | -0.0250 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
