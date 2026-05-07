# 当前数据质量审计

> 生成时间：2026-05-07 09:31:00Z  
> 生成脚本：[scripts/audit_train_set_quality.py](../../scripts/audit_train_set_quality.py)  
> 完整 JSON：[data_quality_audit.json](../../train_set/data_quality_audit.json)

## 1. 当前口径

1. 当前 raw/curated manifests 已是严格的 145x4，shared_program_count=145。
2. 当前训练链路使用的 run_features 是过滤后的子集：509 runs、1494 pairs、374 anchors。
3. 当前过滤后仍只有 122 个完整四变体程序，另有 10 个程序缺至少一个 variant。

## 2. 语义过滤完整名单

1. 当前 curated 账本共 580 runs，其中 71 runs 被语义过滤，保留 509 runs。
2. 过滤原因里 `low_active_pid_count=71`，`nonpositive_cycles_per_iter=57`。
3. 完整名单在 [data_quality_audit.json](../../train_set/data_quality_audit.json) 的 `semantic_filter.filtered_runs`。

| Variant | Program | active_pid_count | cycles_per_iter | reasons |
| --- | --- | --- | --- | --- |
| O0 | Bullet | 4 | 72881139839.33 | low_active_pid_count |
| O0 | McCat_03-testtrie | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | McCat_05-eks | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | MiBench_consumer-jpeg | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | MiBench_consumer-typeset | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | PAQ8p | 2 | 114816592508.00 | low_active_pid_count |
| O0 | Prolangs-C++_employ | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | Prolangs-C++_simul | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | Prolangs-C_agrep | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | Prolangs-C_bison | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | SciMark2-C | 1 | 231091550590.00 | low_active_pid_count |
| O0 | mafft | 3 | 111694709705.00 | low_active_pid_count |
| O0 | mediabench_adpcm_rawcaudio | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | mediabench_adpcm_rawdaudio | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |
| O0 | mediabench_jpeg_jpeg-6a | 1 | 0.00 | low_active_pid_count, nonpositive_cycles_per_iter |

## 3. 严格时间过滤与缺失基线

1. 当前 run_features 里有 109 runs 进不了 strict-time 输入，主要原因都是低 `active_window_ratio`。
2. 另有 10 条 run（分布在 4 个程序上）虽通过 strict 输入检查，但缺 strict O0 baseline。
3. 完整名单分别在 [data_quality_audit.json](../../train_set/data_quality_audit.json) 的 `strict_time_filter.filtered_runs`、`strict_time_filter.missing_strict_baseline_rows` 与 `strict_time_filter.missing_strict_baseline_programs`。

| Variant | Program | active_window_ratio | active_windows | windows | reasons |
| --- | --- | --- | --- | --- | --- |
| O0 | BitBench_uuencode | 0.0181 | 36 | 1989 | low_active_window_ratio |
| O0 | FreeBench_analyzer | 0.0613 | 56 | 913 | low_active_window_ratio |
| O0 | FreeBench_distray | 0.0845 | 54 | 639 | low_active_window_ratio |
| O0 | FreeBench_pcompress2 | 0.0991 | 55 | 555 | low_active_window_ratio |
| O0 | MallocBench_gs | 0.0218 | 40 | 1837 | low_active_window_ratio |
| O0 | McCat_01-qbsort | 0.0453 | 49 | 1081 | low_active_window_ratio |
| O0 | McCat_04-bisect | 0.0482 | 41 | 850 | low_active_window_ratio |
| O0 | McCat_08-main | 0.0285 | 43 | 1511 | low_active_window_ratio |
| O0 | McCat_09-vor | 0.0993 | 58 | 584 | low_active_window_ratio |
| O0 | McCat_17-bintr | 0.0801 | 53 | 662 | low_active_window_ratio |
| O0 | MiBench_automotive-susan | 0.0673 | 53 | 788 | low_active_window_ratio |
| O0 | MiBench_network-dijkstra | 0.0385 | 46 | 1195 | low_active_window_ratio |
| O0 | MiBench_network-patricia | 0.0556 | 53 | 954 | low_active_window_ratio |
| O0 | MiBench_security-rijndael | 0.0197 | 35 | 1777 | low_active_window_ratio |
| O0 | MiBench_security-sha | 0.0117 | 24 | 2053 | low_active_window_ratio |

### 3.1 缺失 strict O0 baseline 的程序

| Program | present_variants | strict_input_variants | missing_variants |
| --- | --- | --- | --- |
| Bullet | O1, O2, O3 | O1, O2, O3 | O0 |
| PAQ8p | O2, O3 | O2, O3 | O0, O1 |
| mafft | O1, O2, O3 | O1, O2, O3 | O0 |
| tramp3d-v4 | O1, O2, O3 | O1, O2 | O0 |

### 3.2 缺失 strict O0 baseline 的 run

| Variant | Program | active_window_ratio | wall_time_sec |
| --- | --- | --- | --- |
| O1 | Bullet | 0.7143 | 60.60 |
| O1 | mafft | 0.9231 | 60.59 |
| O1 | tramp3d-v4 | 0.1295 | 60.75 |
| O2 | Bullet | 0.7059 | 60.61 |
| O2 | PAQ8p | 0.8696 | 60.60 |
| O2 | mafft | 0.9231 | 60.58 |
| O2 | tramp3d-v4 | 0.1172 | 60.72 |
| O3 | Bullet | 0.7059 | 60.61 |
| O3 | PAQ8p | 0.8696 | 60.59 |
| O3 | mafft | 0.9231 | 60.56 |

## 4. 过滤后覆盖缺口

1. 当前过滤后的 run_features 覆盖 132 个程序，其中完整四变体程序 122 个。
2. 不完整程序共有 10 个，完整名单在 [data_quality_audit.json](../../train_set/data_quality_audit.json) 的 `coverage_gaps.incomplete_programs`。

| Program | present_variants | missing_variants |
| --- | --- | --- |
| BitBench_uudecode | O0, O1, O2 | O3 |
| BitBench_uuencode | O0 | O1, O2, O3 |
| Bullet | O1, O2, O3 | O0 |
| MiBench_security-sha | O0 | O1, O2, O3 |
| PAQ8p | O2, O3 | O0, O1 |
| Prolangs-C++_city | O0 | O1, O2, O3 |
| mafft | O1, O2, O3 | O0 |
| mediabench_gsm_toast | O0, O1 | O2, O3 |
| mediabench_mpeg2_mpeg2dec | O0, O2 | O1, O3 |
| tramp3d-v4 | O1, O2, O3 | O0 |

## 5. O2/O3 难例与 tie 区间

1. O2/O3 是当前最难的近邻变体：test `acc_3cls=0.4500`，`aux_tie_recall=0.5455`。
2. 当前 O2/O3 全量程序里，repeat-timing 候选 28 个，tie/near-tie 阈值候选 87 个。
3. 完整名单在 [data_quality_audit.json](../../train_set/data_quality_audit.json) 的 `pair_difficulty.o2_o3_programs`。

| Pair | n_programs | tie_rate | median_|log_ratio| | test_acc_3cls | test_aux_tie_recall |
| --- | --- | --- | --- | --- | --- |
| O1-O2 | 126 | 0.1667 | 0.2791 | 0.6500 | 0.6667 |
| O1-O3 | 125 | 0.2080 | 0.2238 | 0.7000 | 1.0000 |
| O2-O3 | 126 | 0.4524 | 0.0587 | 0.4500 | 0.5455 |

### 5.1 O2/O3 样本分流建议

| Program | abs_log_ratio | label | action_bucket | O2_active_ratio | O3_active_ratio |
| --- | --- | --- | --- | --- | --- |
| TSVC_Reductions-flt | 0.0011 | tie | tie_threshold_candidates | 0.8333 | 0.8333 |
| TSVC_ControlLoops-flt | 0.0016 | tie | tie_threshold_candidates | 0.6667 | 0.6556 |
| TSVC_ControlLoops-dbl | 0.0019 | tie | tie_threshold_candidates | 0.6629 | 0.6742 |
| mafft | 0.0020 | tie | tie_threshold_candidates | 0.9231 | 0.9231 |
| PAQ8p | 0.0027 | tie | tie_threshold_candidates | 0.8696 | 0.8696 |
| Fhourstones-3.1 | 0.0032 | tie | tie_threshold_candidates | 0.4255 | 0.4255 |
| NPB-serial_is | 0.0047 | tie | tie_threshold_candidates | 0.8219 | 0.8219 |
| MiBench_network-dijkstra | 0.0050 | tie | repeat_timing_candidates | 0.0125 | 0.0095 |
| TSVC_StatementReordering-dbl | 0.0054 | tie | tie_threshold_candidates | 0.6818 | 0.6818 |
| TSVC_GlobalDataFlow-dbl | 0.0063 | tie | tie_threshold_candidates | 0.6020 | 0.6061 |
| TSVC_LoopRerolling-dbl | 0.0064 | tie | tie_threshold_candidates | 0.7143 | 0.7024 |
| Trimaran_enc-3des | 0.0064 | tie | tie_threshold_candidates | 0.5405 | 0.5405 |
| TSVC_InductionVariable-dbl | 0.0072 | tie | tie_threshold_candidates | 0.6977 | 0.6977 |
| TSVC_Recurrences-dbl | 0.0076 | tie | tie_threshold_candidates | 0.7375 | 0.7407 |
| TSVC_Packing-dbl | 0.0078 | tie | tie_threshold_candidates | 0.6742 | 0.6742 |

## 6. 建议的下一步

1. P1 - 优先补 run_features 过滤口径与文档同步：当前 curated 账本是 145x4，但进入训练链路的只剩 509 runs；语义过滤丢掉了 71 runs，必须以过滤后口径描述模型结果。
1. P2 - 优先补 strict-time 真值而不是继续扩样：严格时间口径先在输入阶段筛掉了 109 runs，另有 10 条 run（分布在 4 个程序上）缺 strict O0 baseline。
1. P3 - 把 O2/O3 难例拆成补采、调阈值、补时序特征三类处理：O2/O3 难例里 repeat-timing 候选 28 个，tie/near-tie 阈值候选 87 个；剩余 11 个更适合先查时序特征。
1. P4 - 把 10 个缺失变体程序从主评估口径里单列：当前只有 122 个程序保留了完整 O0/O1/O2/O3，仍有 10 个程序在过滤后缺变体。

