/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events.h — eBPF 内核程序与用户态之间共享的数据结构
 *
 * 同时被 bpf/mem_events.bpf.c（内核侧）和 src/ 下的用户态代码引用。
 */
#pragma once

#ifndef __VMLINUX_H__
#include <linux/types.h>
#endif

/* ------------------------------------------------------------------ */
/* Map 键（per-PID 或 per-TID 聚合）                                   */
/* ------------------------------------------------------------------ */
struct entity_key {
    __u32 pid;
    __u32 tid;  /* per_tid=0 时恒为 0，保持 PID 级聚合语义 */
};

/* ------------------------------------------------------------------ */
/* 采集器配置（存放于 config_map[0]）                                  */
/* ------------------------------------------------------------------ */
struct collector_config {
    __u32 target_pid;      /* 目标 PID；0 = 采集所有进程 */
    __u32 target_tid;      /* 目标 TID；0 = 不过滤线程 */
    __u64 window_ns;       /* 时间窗大小（纳秒），用于用户态对齐，内核侧不使用 */
    __u8  emit_events;     /* 1 = 向 ring buffer 输出逐事件记录 */
    __u8  per_tid;         /* 1 = 以 (pid,tid) 为 map 键，做线程级聚合 */
    __u8  enable_lbr;      /* 1 = on_lbr_sample 处理器已挂载，记录 LBR 数据 */
    __u8  track_children;  /* 1 = 同时采集目标进程的子进程/线程（配合 child_pid_set） */
    __u8  _pad1[4];
};

/* ------------------------------------------------------------------ */
/* 每实体（PID 或 TID）累积统计（存放于 pid_stats hash map）            */
/* ------------------------------------------------------------------ */
struct pid_mem_stats {
    /* LLC */
    __u64 llc_loads;         /* LLC load 访问采样计数 */
    __u64 llc_load_misses;   /* LLC load miss 采样计数 */
    __u64 llc_stores;        /* LLC store 访问采样计数 */
    __u64 llc_store_misses;  /* LLC store miss 采样计数 */
    /* dTLB */
    __u64 dtlb_loads;        /* dTLB load 访问采样计数 */
    __u64 dtlb_load_misses;  /* dTLB load miss 采样计数 */
    __u64 dtlb_stores;       /* dTLB store 访问采样计数 */
    __u64 dtlb_store_misses; /* dTLB store miss 采样计数 */
    __u64 dtlb_misses;       /* dTLB 总 miss（load+store，用于向后兼容） */
    /* iTLB */
    __u64 itlb_loads;        /* iTLB load 访问采样计数 */
    __u64 itlb_load_misses;  /* iTLB load miss 采样计数 */
    /* page fault */
    __u64 minor_faults;      /* minor page fault 次数 */
    __u64 major_faults;      /* major page fault 次数 */
    /* LBR */
    __u64 lbr_samples;       /* LBR 采样触发次数 */
    __u64 lbr_entries;       /* 累积读取的 LBR 条目数 */
    /* misc */
    __u64 samples;           /* 所有事件总采样数 */
    __u64 last_seen_ns;      /* 最近一次事件时间戳 (CLOCK_MONOTONIC) */
    char  comm[16];          /* 进程名 */
};

/* ------------------------------------------------------------------ */
/* 逐事件记录（通过 ring buffer 流式输出）                              */
/* ------------------------------------------------------------------ */
#define MEM_EVENT_LLC_LOAD_MISS   1
#define MEM_EVENT_LLC_STORE_MISS  2
#define MEM_EVENT_DTLB_MISS       3
#define MEM_EVENT_MINOR_FAULT     4
#define MEM_EVENT_MAJOR_FAULT     5
#define MEM_EVENT_LBR             6  /* LBR 分支栈采样 */

#define MAX_LBR_ENTRIES           8

/* 单条 LBR（Last Branch Record）分支记录（避免与 vmlinux.h 中的 lbr_entry 冲突） */
struct lbr_sample {
    __u64 from_ip;
    __u64 to_ip;
    __u64 flags;   /* mispred/predicted/cycles 等，直接取 perf_branch_entry 第三字 */
};

struct mem_event {
    __u64 ts_ns;        /* 事件时间戳 (CLOCK_MONOTONIC) */
    __u32 pid;          /* 进程 ID */
    __u32 tid;          /* 线程 ID */
    char  comm[16];     /* 进程名 */
    __u8  event_type;   /* MEM_EVENT_* 常量 */
    __u8  lbr_nr;       /* 有效 LBR 条目数（仅 MEM_EVENT_LBR 有效） */
    __u16 _pad0;
    __u32 _pad1;
    __u64 addr;         /* 出错/采样地址 */
    __u64 ip;           /* 采样时的指令指针 */
    struct lbr_sample lbr[MAX_LBR_ENTRIES]; /* LBR 数据（非 LBR 事件时全零） */
};
