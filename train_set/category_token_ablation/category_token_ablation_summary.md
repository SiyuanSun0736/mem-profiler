# Category Token Ablation Summary

Generated: 2026-05-19 17:47:17 UTC

## Setup

- pairs: train_set/pairs.parquet
- seed: 42
- baseline model: train_set/model_transformer.pt
- objective: separate token grouping effects from extra-token effects

## Overall Test

| Variant | MAE | Delta | dir_acc | Delta | acc_3cls | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| summary_only | 0.6416 | 0.0738 | 0.8873 | -0.0147 | 0.7833 | -0.0125 |
| coarse_2way | 0.5572 | -0.0106 | 0.8971 | -0.0049 | 0.7833 | -0.0125 |
| no_mm_phase_4way | 0.6516 | 0.0838 | 0.8725 | -0.0295 | 0.7417 | -0.0541 |
| semantic_full_reference | 0.5733 | 0.0055 | 0.8578 | -0.0442 | 0.7625 | -0.0333 |

## Focused Slices

| Variant | near_dir | Delta | O2-O3 dir | Delta | tie_rec | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| summary_only | 0.7750 | 0.0750 | 0.7222 | 0.0555 | 0.5833 | -0.0834 |
| coarse_2way | 0.6750 | -0.0250 | 0.6667 | 0.0000 | 0.7222 | 0.0555 |
| no_mm_phase_4way | 0.6750 | -0.0250 | 0.5556 | -0.1111 | 0.6111 | -0.0556 |
| semantic_full_reference | 0.6250 | -0.0750 | 0.5000 | -0.1667 | 0.6667 | 0.0000 |

## Diagnosis

- Reduced semantic groupings recover part of the loss relative to full 6-way, so grouping matters, but extra tokens still appear costly overall.

## Variants

- summary_only: Only keep the global summary token; no semantic category tokens.
- coarse_2way: Two extra tokens: execution (core+phase) and memory (cache+tlb+fault+mm).
- no_mm_phase_4way: Keep core/cache/tlb/fault tokens and drop mm/phase tokens.
- semantic_full_reference: Reference to the existing 6-way semantic category-token experiment.
