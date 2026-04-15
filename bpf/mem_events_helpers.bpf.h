/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_helpers.bpf.h — 内联辅助函数与公共入口宏
 *
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 * 依赖：mem_events_maps.bpf.h 中的 map 声明。
 */
#pragma once

/* ================================================================== */
/* 配置与过滤                                                           */
/* ================================================================== */

static __always_inline struct collector_config *cfg_get(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&config_map, &key);
}

static __always_inline bool pid_allowed(__u32 pid, __u32 tid,
                                        const struct collector_config *cfg)
{
    if (cfg->target_pid != 0 && cfg->target_pid != pid)
        return false;
    if (cfg->target_tid != 0 && cfg->target_tid != tid)
        return false;
    return true;
}

static __always_inline struct entity_key make_entity_key(__u32 pid, __u32 tid,
                                                         const struct collector_config *cfg)
{
    struct entity_key key = {
        .pid = pid,
        .tid = cfg->per_tid ? tid : 0,
    };
    return key;
}

/* ================================================================== */
/* Per-entity 统计槽 get-or-init                                        */
/* ================================================================== */

/*
 * CO-RE PERCPU_HASH 上的 get-or-init 模式：
 *   BPF_NOEXIST 保证只有第一次写入会成功；之后的 lookup 必然命中。
 *   PERCPU_HASH 每 CPU 独立存储，同一 CPU 上 BPF 程序不会互相抢占，
 *   因此两步操作在 per-CPU 语义下是安全的（无 TOCTOU 竞争）。
 */
static __always_inline struct pid_mem_stats *
stats_get_or_init(const struct entity_key *key)
{
    struct pid_mem_stats *s = bpf_map_lookup_elem(&pid_stats, key);
    if (!s) {
        struct pid_mem_stats zero = {};
        bpf_get_current_comm(zero.comm, sizeof(zero.comm));
        bpf_map_update_elem(&pid_stats, key, &zero, BPF_NOEXIST);
        s = bpf_map_lookup_elem(&pid_stats, key);
    }
    return s;
}

/* ================================================================== */
/* 统计更新                                                             */
/* ================================================================== */

/* comm 只在 slot 首次使用时刷新（避免每次采样都拷贝 16 字节）。 */
static __always_inline void touch_stats(struct pid_mem_stats *s)
{
    __sync_fetch_and_add(&s->samples, 1);
    s->last_seen_ns = bpf_ktime_get_ns();
    if (s->comm[0] == '\0')
        bpf_get_current_comm(s->comm, sizeof(s->comm));
}

/* ================================================================== */
/* Ring buffer 输出                                                     */
/* ================================================================== */

static __always_inline void emit_event(struct collector_config *cfg,
                                       __u32 pid, __u32 tid,
                                       __u8 type, __u64 addr, __u64 ip)
{
    if (!cfg->emit_events)
        return;
    struct mem_event *ev = bpf_ringbuf_reserve(&events_rb, sizeof(*ev), 0);
    if (!ev)
        return;
    ev->ts_ns      = bpf_ktime_get_ns();
    ev->pid        = pid;
    ev->tid        = tid;
    ev->event_type = type;
    ev->lbr_nr     = 0;
    ev->addr       = addr;
    ev->ip         = ip;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));
    bpf_ringbuf_submit(ev, 0);
}

static __always_inline void emit_lbr(struct collector_config *cfg,
                                     struct bpf_perf_event_data *bpf_ctx,
                                     __u32 pid, __u32 tid, __u64 ip,
                                     __u8 nr,
                                     struct perf_branch_entry *branches)
{
    if (!cfg->emit_events)
        return;
    struct mem_event *ev = bpf_ringbuf_reserve(&events_rb, sizeof(*ev), 0);
    if (!ev)
        return;
    ev->ts_ns      = bpf_ktime_get_ns();
    ev->pid        = pid;
    ev->tid        = tid;
    ev->event_type = MEM_EVENT_LBR;
    ev->lbr_nr     = nr;
    ev->addr       = bpf_ctx->addr;
    ev->ip         = ip;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));

#pragma unroll
    for (int i = 0; i < MAX_LBR_ENTRIES; i++) {
        if (i >= nr)
            break;
        ev->lbr[i].from_ip = branches[i].from;
        ev->lbr[i].to_ip   = branches[i].to;
        /* flags word: mispred/predicted/cycles/type packed in the third u64 */
        ev->lbr[i].flags   = ((__u64 *)&branches[i])[2];
    }
    bpf_ringbuf_submit(ev, 0);
}

/* ================================================================== */
/* HANDLER_PROLOGUE — 各 perf_event handler 公共入口宏                 */
/* ================================================================== */
/*
 * 展开后在当前作用域引入：cfg、pid、tid、key、s。
 * 任一前置条件不满足即 return 0。
 *
 * 用法：
 *   SEC("perf_event")
 *   int on_xxx(struct bpf_perf_event_data *ctx)
 *   {
 *       HANDLER_PROLOGUE();
 *       // ... 只写差异逻辑 ...
 *       return 0;
 *   }
 */
#define HANDLER_PROLOGUE()                                                   \
    struct collector_config *cfg = cfg_get();                                \
    if (!cfg) return 0;                                                      \
    __u64 _pidtid = bpf_get_current_pid_tgid();                             \
    __u32 pid = _pidtid >> 32, tid = (__u32)_pidtid;                        \
    if (!pid_allowed(pid, tid, cfg)) return 0;                               \
    struct entity_key key = make_entity_key(pid, tid, cfg);                  \
    struct pid_mem_stats *s = stats_get_or_init(&key);                       \
    if (!s) return 0

/* LBR handler 需要额外检查 enable_lbr 标志 */
#define HANDLER_PROLOGUE_LBR()                                               \
    struct collector_config *cfg = cfg_get();                                \
    if (!cfg || !cfg->enable_lbr) return 0;                                  \
    __u64 _pidtid = bpf_get_current_pid_tgid();                             \
    __u32 pid = _pidtid >> 32, tid = (__u32)_pidtid;                        \
    if (!pid_allowed(pid, tid, cfg)) return 0;                               \
    struct entity_key key = make_entity_key(pid, tid, cfg);                  \
    struct pid_mem_stats *s = stats_get_or_init(&key);                       \
    if (!s) return 0
