# Fixed-Work Bucket Token vs Baseline Comparison

Generated: 2026-05-19 17:44:39 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkBucketTokenPairTransformer (train_set/tmp_fixed_work_token_smoke/model_fixed_work_token_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 1.3778 | 0.8100 |
| rmse | 0.8685 | 1.8412 | 0.9727 |
| r2 | 0.8069 | 0.1323 | -0.6746 |
| dir_acc | 0.9020 | 0.6765 | -0.2255 |
| acc_3cls | 0.7958 | 0.5542 | -0.2416 |
| aux_acc_3cls | 0.8417 | 0.4042 | -0.4375 |
| aux_tie_recall | 0.6667 | 0.3056 | -0.3611 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.3056 |
| near_tie | 40 | 0.7000 | 0.6750 | -0.0250 | - | - |
| O2-O3 | 40 | 0.6667 | 0.7222 | 0.0555 | 0.5455 | 0.2727 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.0556 | -0.3333 | 0.6667 | 0.3056 | -0.3611 |
| near_tie | 0.6500 | 0.6500 | 0.0000 | 0.5750 | 0.3750 | -0.2000 |
| O2-O3 | 0.4750 | 0.3500 | -0.1250 | 0.5750 | 0.3750 | -0.2000 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
