# Transformer objective ablation

> Generated: 2026-05-11T13:10:03.624462+00:00

## Conclusion

Best objective: `reg_ce` with `aux_class_lambda=0.05` and `direction_lambda=0.0`.

The regression head remains the primary output for continuous log-ratio scoring. The auxiliary CE head is evaluated by `aux_acc_3cls`, `aux_tie_recall`, and hard-pair behavior.

## Trial table

| objective | aux λ | dir λ | test MAE | test R2 | test dir | test 3cls | aux 3cls | aux tie recall | O2-O3 3cls | O2-O3 aux | O2-O3 tie recall | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reg_only | 0.00 | 0.00 | 0.5829 | 0.8009 | 0.8971 | 0.7708 | 0.2667 | 0.4167 | 0.3500 | 0.4000 | 0.4545 | 2.7152 |
| reg_ce | 0.05 | 0.00 | 0.5677 | 0.8031 | 0.9069 | 0.7708 | 0.8417 | 0.7222 | 0.4000 | 0.6000 | 0.5455 | 4.1465 |
| reg_ce | 0.10 | 0.00 | 0.5625 | 0.8067 | 0.8922 | 0.7708 | 0.8375 | 0.6944 | 0.4500 | 0.5500 | 0.5455 | 4.0826 |
| reg_ce | 0.20 | 0.00 | 0.5748 | 0.8040 | 0.8922 | 0.7667 | 0.8375 | 0.6667 | 0.2500 | 0.5500 | 0.5455 | 4.0731 |
| reg_ce | 0.30 | 0.00 | 0.5738 | 0.8035 | 0.8873 | 0.7583 | 0.8292 | 0.5833 | 0.2500 | 0.5500 | 0.5455 | 4.0417 |

## Selection rule

The ranking favors higher auxiliary three-class accuracy, regression-derived three-class accuracy, O2-O3 auxiliary accuracy, O2-O3 tie recall, and direction accuracy, with a small penalty for MAE. This is a model-selection aid; final single-program scoring must still be checked against proxy and time scores.
