# Fixed-Work + Hotspot Token vs Baseline Comparison

Generated: 2026-05-20 05:29:09 UTC

## Setup

- pairs: train_set/pairs.parquet
- run_features: train_set/run_features.parquet
- seed: 42
- test programs: 20
- test pairs: 240
- baseline model: PairTransformer (train_set/model_transformer.pt)
- experiment model: FixedWorkHotspotTokenPairTransformer (output_smoke_test/model_fixed_work_hotspot_token_transformer.pt)

## Overall Test

| Metric | Baseline | FixedWorkHotspot | Delta(exp-base) |
| --- | --- | --- | --- |
| mae | 0.5678 | 1.3937 | 0.8259 |
| rmse | 0.8685 | 1.9430 | 1.0745 |
| r2 | 0.8069 | 0.0337 | -0.7732 |
| dir_acc | 0.9020 | 0.6618 | -0.2402 |
| acc_3cls | 0.7958 | 0.5583 | -0.2375 |
| aux_acc_3cls | 0.8417 | 0.4208 | -0.4209 |
| aux_tie_recall | 0.6667 | 0.3056 | -0.3611 |

## Focused Slices

| Slice | n | dir_acc(base) | dir_acc(exp) | delta | tie_rec(base) | tie_rec(exp) |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 36 | - | - | - | 0.6667 | 0.3056 |
| near_tie | 40 | 0.7000 | 0.6250 | -0.0750 | - | - |
| O2-O3 | 40 | 0.6667 | 0.5556 | -0.1111 | 0.5455 | 0.3182 |

| Slice | acc_3cls(base) | acc_3cls(exp) | delta | aux_3cls(base) | aux_3cls(exp) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| tie | 0.3889 | 0.0556 | -0.3333 | 0.6667 | 0.3056 | -0.3611 |
| near_tie | 0.6500 | 0.6000 | -0.0500 | 0.5750 | 0.4250 | -0.1500 |
| O2-O3 | 0.4750 | 0.2750 | -0.2000 | 0.5750 | 0.3500 | -0.2250 |

## Notes

- tie uses |log_ratio| <= 0.05
- near_tie uses 0.05 < |log_ratio| <= 0.25
- O2-O3 is computed as an unordered variant pair, so both directions are included.
