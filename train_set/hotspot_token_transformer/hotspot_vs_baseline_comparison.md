# Hotspot Token vs Baseline Comparison

Generated: 2026-05-19 17:08:54 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: HotspotWindowTokenPairTransformer (train_set/hotspot_token_transformer/model_hotspot_token_transformer.pt)

## Overall Test

| Metric | Baseline | HotspotToken | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 0.6162 | 0.0484 |
| rmse | 0.8685 | 0.9475 | 0.0790 |
| r2 | 0.8069 | 0.7702 | -0.0367 |
| dir_acc | 0.9020 | 0.8725 | -0.0295 |
| acc_3cls | 0.7958 | 0.7625 | -0.0333 |
| aux_acc_3cls | 0.8417 | 0.7625 | -0.0792 |
| aux_tie_recall | 0.6667 | 0.6389 | -0.0278 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.6389 |
| near_tie | 40 | 0.7000 | 0.6750 | -0.0250 | - | - |
| O2-O3 | 40 | 0.6667 | 0.7778 | 0.1111 | 0.5455 | 0.5909 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.3056 | -0.0833 | 0.6667 | 0.6389 | -0.0278 |
| near_tie | 0.6500 | 0.6000 | -0.0500 | 0.5750 | 0.3000 | -0.2750 |
| O2-O3 | 0.4750 | 0.4500 | -0.0250 | 0.5750 | 0.5250 | -0.0500 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
