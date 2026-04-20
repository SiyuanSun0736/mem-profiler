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

static __always_inline bool fault_classification_enabled(void)
{
    u32 key = 0;
    u32 *enabled = fault_classification_map.lookup(&key);
    return enabled && *enabled != 0;
}

static __always_inline struct mem_event_t *event_scratch_get(void)
{
    u32 key = 0;
    struct mem_event_t *ev = event_scratch_map.lookup(&key);
    if (!ev)
        return NULL;
    __builtin_memset(ev, 0, sizeof(*ev));
    return ev;
}

static __always_inline bool comm_allowed(void)
{
    u32 key = 0;
    struct task_comm_filter_t *filter = target_comm_map.lookup(&key);
    char comm_buf[TASK_COMM_LEN] = {};

    if (!filter || filter->comm[0] == '\0')
        return true;

    bpf_get_current_comm(comm_buf, sizeof(comm_buf));

#pragma unroll
    for (int i = 0; i < TASK_COMM_LEN; i++) {
        if (filter->comm[i] != comm_buf[i])
            return false;
        if (filter->comm[i] == '\0')
            break;
    }

    return true;
}

static __always_inline bool task_allowed(u32 pid, u32 tid)
{
    u32 key = 0;
    u32 *tpid = target_pid_map.lookup(&key);
    u32 *ttid = target_tid_map.lookup(&key);

    if (tpid && *tpid != 0 && *tpid != pid) {
        /*
         * PID 不匹配根进程 — 检查 child_pid_set。
         * Python 端后台线程会将已知子进程/线程 PID 写入此 map，
         * 从而让继承链上的所有代均能通过过滤。
         */
        u8 *in_set = child_pid_set.lookup(&pid);
        if (!in_set)
            return false;
    }
    if (ttid && *ttid != 0 && *ttid != tid)
        return false;
    if (!comm_allowed())
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

    struct mem_event_t *ev = event_scratch_get();
    if (!ev)
        return;
    ev->ts_ns      = bpf_ktime_get_ns();
    ev->pid        = pid;
    ev->tid        = tid;
    ev->event_type = etype;
    ev->addr       = addr;
    ev->ip         = ip;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));
    events_rb.ringbuf_output(ev, sizeof(*ev), 0);
}

static __always_inline void emit_fault_event(u32 pid,
                                             u32 tid,
                                             u8 etype,
                                             u64 addr,
                                             u64 ip,
                                             u32 fault_flags,
                                             u32 class_flags,
                                             u64 vma_flags)
{
    if (!emit_events_enabled())
        return;

    struct mem_event_t *ev = event_scratch_get();
    if (!ev)
        return;
    ev->ts_ns       = bpf_ktime_get_ns();
    ev->pid         = pid;
    ev->tid         = tid;
    ev->event_type  = etype;
    ev->addr        = addr;
    ev->ip          = ip;
    ev->event_flags = fault_flags;
    ev->class_flags = class_flags;
    ev->vma_flags   = vma_flags;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));
    events_rb.ringbuf_output(ev, sizeof(*ev), 0);
}

static __always_inline void emit_mm_syscall_event(u32 pid,
                                                  u32 tid,
                                                  u8 etype,
                                                  u64 addr,
                                                  u64 ip,
                                                  u64 requested_addr,
                                                  u64 length,
                                                  u32 prot,
                                                  u32 event_flags,
                                                  u32 class_flags,
                                                  s64 delta_bytes)
{
    if (!emit_events_enabled())
        return;

    struct mem_event_t *ev = event_scratch_get();
    if (!ev)
        return;
    ev->ts_ns          = bpf_ktime_get_ns();
    ev->pid            = pid;
    ev->tid            = tid;
    ev->event_type     = etype;
    ev->addr           = addr;
    ev->ip             = ip;
    ev->requested_addr = requested_addr;
    ev->length         = length;
    ev->prot           = prot;
    ev->event_flags    = event_flags;
    ev->class_flags    = class_flags;
    ev->delta_bytes    = delta_bytes;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));
    events_rb.ringbuf_output(ev, sizeof(*ev), 0);
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

    struct mem_event_t *ev = event_scratch_get();
    if (!ev)
        return;
    ev->ts_ns      = bpf_ktime_get_ns();
    ev->pid        = pid;
    ev->tid        = tid;
    ev->event_type = MEM_EVENT_LBR;
    ev->lbr_nr     = nr;
    ev->addr       = ctx->addr;
    ev->ip         = ip;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));

#pragma unroll
    for (int i = 0; i < MAX_LBR_ENTRIES; i++) {
        if (i >= nr)
            break;
        ev->lbr[i].from_ip = branches[i].from;
        ev->lbr[i].to_ip   = branches[i].to;
        ev->lbr[i].flags   = ((__u64 *)&branches[i])[2];
    }

    events_rb.ringbuf_output(ev, sizeof(*ev), 0);
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
