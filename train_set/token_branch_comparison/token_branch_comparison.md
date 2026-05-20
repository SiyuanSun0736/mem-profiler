# Token Branch Comparison

Generated: 2026-05-19 17:49:51 UTC

## Inputs

- category: train_set/category_token_transformer/category_vs_baseline_comparison.json
- hotspot: train_set/hotspot_token_transformer/hotspot_vs_baseline_comparison.json
- fixed_work: train_set/fixed_work_token_transformer/fixed_work_vs_baseline_comparison.json

## Overall Test

| Branch | MAE | Delta | dir_acc | Delta | acc_3cls | Delta | tie_rec | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CategoryToken | 0.5733 | 0.0055 | 0.8578 | -0.0442 | 0.7625 | -0.0333 | 0.6667 | 0.0000 |
| HotspotToken | 0.6162 | 0.0484 | 0.8725 | -0.0295 | 0.7625 | -0.0333 | 0.6389 | -0.0278 |
| FixedWorkToken | 0.5977 | 0.0299 | 0.9216 | 0.0196 | 0.7792 | -0.0166 | 0.5278 | -0.1389 |

## Focused Slices

| Branch | near_dir | Delta | O2-O3 dir | Delta | tie_rec | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| CategoryToken | 0.6250 | -0.0750 | 0.5000 | -0.1667 | 0.6667 | 0.0000 |
| HotspotToken | 0.6750 | -0.0250 | 0.7778 | 0.1111 | 0.6389 | -0.0278 |
| FixedWorkToken | 0.7000 | 0.0000 | 0.7778 | 0.1111 | 0.5278 | -0.1389 |
