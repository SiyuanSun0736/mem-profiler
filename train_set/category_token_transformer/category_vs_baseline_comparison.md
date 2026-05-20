# Category Token vs Baseline Comparison

Generated: 2026-05-19 16:53:50 UTC

## Setup

- pairs: train_set/pairs.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: CategoryTokenPairTransformer (train_set/category_token_transformer/model_category_token_transformer.pt)

## Overall Test

| Metric | Baseline | CategoryToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 0.5733 | 0.0055 |
| rmse | 0.8685 | 0.9159 | 0.0474 |
| r2 | 0.8069 | 0.7853 | -0.0216 |
| dir_acc | 0.9020 | 0.8578 | -0.0442 |
| acc_3cls | 0.7958 | 0.7625 | -0.0333 |
| aux_acc_3cls | 0.8417 | 0.8167 | -0.0250 |
| aux_tie_recall | 0.6667 | 0.6667 | 0.0000 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.6667 |
| near_tie | 40 | 0.7000 | 0.6250 | -0.0750 | - | - |
| O2-O3 | 40 | 0.6667 | 0.5000 | -0.1667 | 0.5455 | 0.5909 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.3333 | -0.0556 | 0.6667 | 0.6667 | 0.0000 |
| near_tie | 0.6500 | 0.5750 | -0.0750 | 0.5750 | 0.5000 | -0.0750 |
| O2-O3 | 0.4750 | 0.3750 | -0.1000 | 0.5750 | 0.5500 | -0.0250 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
