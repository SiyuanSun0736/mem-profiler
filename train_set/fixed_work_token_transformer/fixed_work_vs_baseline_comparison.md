# Fixed-Work Bucket Token vs Baseline Comparison

Generated: 2026-05-19 17:49:47 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkBucketTokenPairTransformer (train_set/fixed_work_token_transformer/model_fixed_work_token_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 0.5977 | 0.0299 |
| rmse | 0.8685 | 0.9255 | 0.0570 |
| r2 | 0.8069 | 0.7808 | -0.0261 |
| dir_acc | 0.9020 | 0.9216 | 0.0196 |
| acc_3cls | 0.7958 | 0.7792 | -0.0166 |
| aux_acc_3cls | 0.8417 | 0.7875 | -0.0542 |
| aux_tie_recall | 0.6667 | 0.5278 | -0.1389 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.5278 |
| near_tie | 40 | 0.7000 | 0.7000 | 0.0000 | - | - |
| O2-O3 | 40 | 0.6667 | 0.7778 | 0.1111 | 0.5455 | 0.4545 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.1667 | -0.2222 | 0.6667 | 0.5278 | -0.1389 |
| near_tie | 0.6500 | 0.6000 | -0.0500 | 0.5750 | 0.4000 | -0.1750 |
| O2-O3 | 0.4750 | 0.3500 | -0.1250 | 0.5750 | 0.5000 | -0.0750 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
