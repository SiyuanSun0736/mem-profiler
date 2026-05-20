# Category Token vs Baseline Comparison

Generated: 2026-05-19 16:51:28 UTC

## Setup

- pairs: train_set/pairs.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: CategoryTokenPairTransformer (train_set/tmp_category_token_smoke/model_category_token_transformer.pt)

## Overall Test

| Metric | Baseline | CategoryToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 1.2426 | 0.6748 |
| rmse | 0.8685 | 1.7715 | 0.9030 |
| r2 | 0.8069 | 0.1968 | -0.6101 |
| dir_acc | 0.9020 | 0.6912 | -0.2108 |
| acc_3cls | 0.7958 | 0.6000 | -0.1958 |
| aux_acc_3cls | 0.8417 | 0.3875 | -0.4542 |
| aux_tie_recall | 0.6667 | 0.1111 | -0.5556 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.1111 |
| near_tie | 40 | 0.7000 | 0.6250 | -0.0750 | - | - |
| O2-O3 | 40 | 0.6667 | 0.6667 | 0.0000 | 0.5455 | 0.1364 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.1111 | -0.2778 | 0.6667 | 0.1111 | -0.5556 |
| near_tie | 0.6500 | 0.6250 | -0.0250 | 0.5750 | 0.4250 | -0.1500 |
| O2-O3 | 0.4750 | 0.3500 | -0.1250 | 0.5750 | 0.2500 | -0.3250 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
