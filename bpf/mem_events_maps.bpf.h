/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_maps.bpf.h — BPF map 定义
 *
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 * 依赖：mem_events.h 中的 entity_key / pid_mem_stats / collector_config。
 */
#pragma once

/* 每实体（PID 或 TID）累积统计（PERCPU 减少锁竞争，用户态读取后跨 CPU 求和） */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 8192);
    __type(key,   struct entity_key);
    __type(value, struct pid_mem_stats);
} pid_stats SEC(".maps");

/* 逐事件 ring buffer（仅在 emit_events=1 时写入） */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);   /* 16 MiB */
} events_rb SEC(".maps");

/* 采集器配置（key=0 存放 struct collector_config） */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key,   __u32);
    __type(value, struct collector_config);
} config_map SEC(".maps");

/*
 * child_pid_set — 子进程/线程 PID 集合（用户态由后台线程维护）
 * 用于在 target_pid 不匹配时允许其子代的事件通过。
 * key = child PID，value = 1（仅用作存在标志）。
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key,   __u32);
    __type(value, __u8);
} child_pid_set SEC(".maps");
