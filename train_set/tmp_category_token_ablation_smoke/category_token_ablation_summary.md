# Category Token Ablation Summary

Generated: 2026-05-19 17:43:35 UTC

## Setup

- pairs: train_set/pairs.parquet
- seed: 42
- baseline model: train_set/model_transformer.pt
- objective: separate token grouping effects from extra-token effects

## Overall Test

| Variant | MAE | Delta | dir_acc | Delta | acc_3cls | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| summary_only | 1.5147 | 0.9469 | 0.5980 | -0.3040 | 0.5042 | -0.2916 |
| coarse_2way | 1.3061 | 0.7383 | 0.6324 | -0.2696 | 0.5375 | -0.2583 |
| no_mm_phase_4way | 1.3224 | 0.7546 | 0.6422 | -0.2598 | 0.5458 | -0.2500 |
| semantic_full_reference | 0.5733 | 0.0055 | 0.8578 | -0.0442 | 0.7625 | -0.0333 |

## Focused Slices

| Variant | near_dir | Delta | O2-O3 dir | Delta | tie_rec | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| summary_only | 0.5250 | -0.1750 | 0.5000 | -0.1667 | 0.2222 | -0.4445 |
| coarse_2way | 0.5500 | -0.1500 | 0.5556 | -0.1111 | 0.0556 | -0.6111 |
| no_mm_phase_4way | 0.5750 | -0.1250 | 0.4444 | -0.2223 | 0.5278 | -0.1389 |
| semantic_full_reference | 0.6250 | -0.0750 | 0.5000 | -0.1667 | 0.6667 | 0.0000 |

## Diagnosis

- No grouping recovered a clear gain over baseline; extra semantic tokens are still not helping under the current setup.

## Variants

- summary_only: Only keep the global summary token; no semantic category tokens.
- coarse_2way: Two extra tokens: execution (core+phase) and memory (cache+tlb+fault+mm).
- no_mm_phase_4way: Keep core/cache/tlb/fault tokens and drop mm/phase tokens.
- semantic_full_reference: Reference to the existing 6-way semantic category-token experiment.
