/*
 * bcc_types.h — BCC eBPF 程序共享类型定义
 *
 * 包含：内核 helper 前置声明、共享结构体。
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

#include <uapi/linux/ptrace.h>
#include <uapi/linux/perf_event.h>
#include <linux/sched.h>
#include <linux/mm_types.h>
#include <linux/mm.h>

/*
 * BCC 的 helpers.h 只前置声明了 struct bpf_perf_event_data，
 * 但 perf_event handler 里需要读取 regs 和 branch records。这里补一个
 * 与内核公开 UAPI 一致的最小定义，避免依赖 CO-RE 的 vmlinux.h。
 */
struct bpf_perf_event_data {
    struct pt_regs regs;
    u64 sample_period;
    u64 addr;
};

/*
 * bpf_read_branch_records, bpf_perf_event_value, bpf_perf_prog_read_value
 * are already declared by BCC's helpers.h / bpf.h with int (*) return type.
 * Do not re-declare them here to avoid redefinition conflicts.
 */

#define MAX_LBR_ENTRIES 8

/* ------------------------------------------------------------------ */
/* 共享结构体（与用户态保持字段语义一致）                               */
/* ------------------------------------------------------------------ */

struct entity_key_t {
    u32 pid;
    u32 tid;
};

struct task_comm_filter_t {
    char comm[TASK_COMM_LEN];
};

struct pid_mem_stats_t {
    u64 llc_loads;
    u64 llc_load_misses;
    u64 llc_stores;
    u64 llc_store_misses;
    u64 dtlb_loads;
    u64 dtlb_load_misses;
    u64 dtlb_stores;
    u64 dtlb_store_misses;
    u64 dtlb_misses;
    u64 itlb_load_misses;
    u64 cycles;
    u64 instructions;
    u64 minor_faults;
    u64 major_faults;
    u64 lbr_samples;
    u64 lbr_entries;
    u64 samples;
    u64 last_seen_ns;
    char comm[TASK_COMM_LEN];
};

struct lbr_entry_t {
    u64 from_ip;
    u64 to_ip;
    u64 flags;
};

struct mem_event_t {
    u64 ts_ns;
    u32 pid;
    u32 tid;
    char comm[TASK_COMM_LEN];
    u8  event_type; /* 1=llc_load_miss 2=llc_store_miss 3=dtlb_miss 4=minor_fault 5=major_fault 6=lbr */
    u8  lbr_nr;
    u16 _pad0;
    u32 _pad1;
    u64 addr;
    u64 ip;
    struct lbr_entry_t lbr[MAX_LBR_ENTRIES];
};
