/*
 * bcc_helpers.h — BCC eBPF 内联辅助函数与公共入口宏
 *
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 * 依赖：bcc_maps.h 中的 map 声明。
 */
#pragma once

/* ================================================================== */
/* 配置查询                                                             */
/* ================================================================== */

static __always_inline bool emit_events_enabled(void)
{
    u32 key = 0;
    u32 *enabled = emit_events_map.lookup(&key);
    return enabled && *enabled != 0;
}

static __always_inline bool per_tid_enabled(void)
{
    u32 key = 0;
    u32 *enabled = per_tid_map.lookup(&key);
    return enabled && *enabled != 0;
}

static __always_inline bool task_allowed(u32 pid, u32 tid)
{
    u32 key = 0;
    u32 *tpid = target_pid_map.lookup(&key);
    u32 *ttid = target_tid_map.lookup(&key);

    if (tpid && *tpid != 0 && *tpid != pid)
        return false;
    if (ttid && *ttid != 0 && *ttid != tid)
        return false;
    return true;
}

/* ================================================================== */
/* Per-entity 统计槽                                                    */
/* ================================================================== */

static __always_inline struct entity_key_t make_entity_key(u32 pid, u32 tid)
{
    struct entity_key_t key = {
        .pid = pid,
        .tid = per_tid_enabled() ? tid : 0,
    };
    return key;
}

static __always_inline struct pid_mem_stats_t *get_or_init(struct entity_key_t *key)
{
    struct pid_mem_stats_t zero = {};
    bpf_get_current_comm(zero.comm, sizeof(zero.comm));
    /*
     * lookup_or_try_init is the BCC atomic primitive: atomically returns
     * the existing value (or inserts zero and returns a pointer to it).
     * Avoids the TOCTOU race of the old insert() + lookup() two-step.
     */
    return pid_stats.lookup_or_try_init(key, &zero);
}

/* ================================================================== */
/* 统计更新                                                             */
/* ================================================================== */

static __always_inline void touch_stats(struct pid_mem_stats_t *s)
{
    /*
     * bpf_ktime_get_ns() is cheap (~20 ns).  bpf_get_current_comm() is NOT:
     * it copies 16 bytes from task_struct under RCU and costs ~100-200 ns.
     * We already set comm in get_or_init, so only refresh it here if the
     * slot is still empty (exec() that changed the name is an edge case).
     */
    s->samples++;
    s->last_seen_ns = bpf_ktime_get_ns();
    if (s->comm[0] == '\0')
        bpf_get_current_comm(s->comm, sizeof(s->comm));
}

/* ================================================================== */
/* Ring buffer 输出                                                     */
/* ================================================================== */

static __always_inline void emit(u32 pid, u32 tid, u8 etype, u64 addr, u64 ip)
{
    if (!emit_events_enabled())
        return;

    struct mem_event_t ev = {};
    ev.ts_ns      = bpf_ktime_get_ns();
    ev.pid        = pid;
    ev.tid        = tid;
    ev.event_type = etype;
    ev.addr       = addr;
    ev.ip         = ip;
    bpf_get_current_comm(ev.comm, sizeof(ev.comm));
    events_rb.ringbuf_output(&ev, sizeof(ev), 0);
}

static __always_inline void emit_lbr(struct bpf_perf_event_data *ctx,
                                     u32 pid,
                                     u32 tid,
                                     u64 ip,
                                     u8 nr,
                                     struct perf_branch_entry *branches)
{
    if (!emit_events_enabled())
        return;

    struct mem_event_t ev = {};
    ev.ts_ns      = bpf_ktime_get_ns();
    ev.pid        = pid;
    ev.tid        = tid;
    ev.event_type = 6;
    ev.lbr_nr     = nr;
    ev.addr       = ctx->addr;
    ev.ip         = ip;
    bpf_get_current_comm(ev.comm, sizeof(ev.comm));

#pragma unroll
    for (int i = 0; i < MAX_LBR_ENTRIES; i++) {
        if (i >= nr)
            break;
        ev.lbr[i].from_ip = branches[i].from;
        ev.lbr[i].to_ip   = branches[i].to;
        ev.lbr[i].flags   = ((__u64 *)&branches[i])[2];
    }

    events_rb.ringbuf_output(&ev, sizeof(ev), 0);
}

/* ================================================================== */
/* BCC_PROLOGUE — 各 perf_event handler 公共入口宏                     */
/* ================================================================== */
/*
 * 展开后在当前作用域引入：pid、tid、key、s。
 * 任一前置条件不满足即 return 0。
 *
 * 用法：
 *   int on_xxx(struct bpf_perf_event_data *ctx)
 *   {
 *       BCC_PROLOGUE();
 *       // ... 只写差异逻辑 ...
 *       return 0;
 *   }
 */
#define BCC_PROLOGUE()                                                   \
    u64 _pidtid = bpf_get_current_pid_tgid();                           \
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;                        \
    if (!task_allowed(pid, tid)) return 0;                               \
    struct entity_key_t key = make_entity_key(pid, tid);                 \
    struct pid_mem_stats_t *s = get_or_init(&key);                       \
    if (!s) return 0
