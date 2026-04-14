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
/* 采集器配置（存放于 config_map[0]）                                  */
/* ------------------------------------------------------------------ */
struct collector_config {
    __u32 target_pid;    /* 目标 PID；0 = 采集所有进程 */
    __u32 _pad0;
    __u64 window_ns;     /* 时间窗大小（纳秒），用于用户态对齐，内核侧不使用 */
    __u8  emit_events;   /* 1 = 向 ring buffer 输出逐事件记录 */
    __u8  _pad1[7];
};

/* ------------------------------------------------------------------ */
/* 每 PID 累积统计（存放于 pid_stats hash map）                         */
/* ------------------------------------------------------------------ */
struct pid_mem_stats {
    __u64 llc_load_misses;   /* LLC load miss 采样计数 */
    __u64 llc_store_misses;  /* LLC store miss 采样计数（若硬件支持） */
    __u64 dtlb_misses;       /* dTLB load miss 采样计数 */
    __u64 minor_faults;      /* minor page fault 次数 */
    __u64 major_faults;      /* major page fault 次数 */
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

struct mem_event {
    __u64 ts_ns;        /* 事件时间戳 (CLOCK_MONOTONIC) */
    __u32 pid;          /* 进程 ID */
    __u32 tid;          /* 线程 ID */
    char  comm[16];     /* 进程名 */
    __u8  event_type;   /* MEM_EVENT_* 常量 */
    __u8  _pad[7];
    __u64 addr;         /* 出错/采样地址（perf_event 时为 IP 指向的采样地址） */
    __u64 ip;           /* 采样时的指令指针 */
};
