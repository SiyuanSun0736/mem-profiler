/*
 * bcc_maps.h — BCC eBPF map 声明
 *
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 * 依赖：bcc_types.h 中的 entity_key_t / pid_mem_stats_t。
 */
#pragma once

BPF_PERCPU_HASH(pid_stats, struct entity_key_t, struct pid_mem_stats_t, 8192);
BPF_RINGBUF_OUTPUT(events_rb, 256);   /* 256 pages ≈ 1 MiB */

/* key=0 的配置项 */
BPF_ARRAY(target_pid_map, u32, 1);
BPF_ARRAY(target_tid_map, u32, 1);
BPF_ARRAY(target_comm_map, struct task_comm_filter_t, 1);

/*
 * child_pid_set — 子进程/线程 PID 集合（Python 端由后台线程维护）
 * 用于在 target_pid_map 无法覆盖的子代上允许事件通过。
 * key = child PID，value = 1（仅用作存在标志）。
 */
BPF_HASH(child_pid_set, u32, u8, 4096);
BPF_ARRAY(per_tid_map,    u32, 1);
BPF_ARRAY(emit_events_map, u32, 1);
BPF_ARRAY(fault_classification_map, u32, 1);

BPF_HASH(pending_fault_args, u32, struct pending_fault_t, 8192);
BPF_HASH(pending_mmap_args, u32, struct pending_mmap_args_t, 8192);
BPF_HASH(pending_munmap_args, u32, struct pending_range_args_t, 8192);
BPF_HASH(pending_mprotect_args, u32, struct pending_range_args_t, 8192);
BPF_HASH(pending_brk_args, u32, struct pending_brk_args_t, 8192);
BPF_HASH(last_brk_by_tgid, u32, u64, 8192);
BPF_PERCPU_ARRAY(event_scratch_map, struct mem_event_t, 1);
BPF_PERCPU_ARRAY(lbr_branch_scratch_map, struct lbr_branch_scratch_t, 1);
