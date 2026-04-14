/*
 * bcc_prog.c — BCC 兼容版 eBPF 程序（P1 原型阶段使用）
 *
 * 与 bpf/mem_events.bpf.c（CO-RE/libbpf 版本）功能等价，
 * 但使用 BCC 宏和标准内核头，可由 src/collector.py 通过 BCC Python 接口
 * 在运行时动态编译加载，无需预先 make。
 *
 * 本版本额外支持：
 *   • 更多 cache/TLB perf counter
 *   • 可选 per-TID 聚合 / TID 过滤
 *   • 可选 LBR 分支栈采样并通过 ring buffer 输出
 */

#include <uapi/linux/ptrace.h>
#include <uapi/linux/perf_event.h>
/* <uapi/linux/bpf.h> is injected by BCC at compile-time; avoid re-including
 * it here to prevent redefinition conflicts with bpf_perf_event_value etc. */
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

static long (*bpf_read_branch_records)(struct bpf_perf_event_data *ctx,
                                       void *buf,
                                       u32 size,
                                       u64 flags) = (void *)119;

/*
 * bpf_perf_prog_read_value(ctx, buf, buf_size)
 * Read the actual PMU counter value (counter, enabled_ns, running_ns) for
 * the perf_event that triggered this BPF program.  Helper index 63.
 * Returns 0 on success, negative errno on failure.
 */
struct bpf_perf_event_value {
    u64 counter;
    u64 enabled;
    u64 running;
};
static long (*bpf_perf_prog_read_value)(struct bpf_perf_event_data *ctx,
                                        struct bpf_perf_event_value *buf,
                                        u32 buf_size) = (void *)63;

#define MAX_LBR_ENTRIES 8

/* ------------------------------------------------------------------ */
/* 共享结构体（与用户态保持字段语义一致）                               */
/* ------------------------------------------------------------------ */

struct entity_key_t {
    u32 pid;
    u32 tid;
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
    u64 itlb_loads;
    u64 itlb_load_misses;
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

/* ------------------------------------------------------------------ */
/* Maps                                                                 */
/* ------------------------------------------------------------------ */

BPF_PERCPU_HASH(pid_stats, struct entity_key_t, struct pid_mem_stats_t, 8192);
BPF_RINGBUF_OUTPUT(events_rb, 256);   /* 256 pages ≈ 1 MiB */

/* key=0 的配置项 */
BPF_ARRAY(target_pid_map, u32, 1);
BPF_ARRAY(target_tid_map, u32, 1);
BPF_ARRAY(per_tid_map, u32, 1);
BPF_ARRAY(emit_events_map, u32, 1);

/* ------------------------------------------------------------------ */
/* 内联辅助                                                             */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* LLC Access / Miss                                                    */
/* ------------------------------------------------------------------ */

int on_llc_load(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->llc_loads++;
    touch_stats(s);
    return 0;
}

int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->llc_load_misses++;
    touch_stats(s);

    u64 ip = PT_REGS_IP(&ctx->regs);
    emit(pid, tid, 1, ip, ip);
    return 0;
}

int on_llc_store(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->llc_stores++;
    touch_stats(s);
    return 0;
}

int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->llc_store_misses++;
    touch_stats(s);
    emit(pid, tid, 2, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

/* ------------------------------------------------------------------ */
/* dTLB / iTLB Access / Miss                                            */
/* ------------------------------------------------------------------ */

int on_dtlb_load(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->dtlb_loads++;
    touch_stats(s);
    return 0;
}

int on_dtlb_load_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->dtlb_load_misses++;
    s->dtlb_misses++;
    touch_stats(s);
    emit(pid, tid, 3, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

int on_dtlb_store(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->dtlb_stores++;
    touch_stats(s);
    return 0;
}

int on_dtlb_store_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->dtlb_store_misses++;
    s->dtlb_misses++;
    touch_stats(s);
    emit(pid, tid, 3, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

int on_itlb_load(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->itlb_loads++;
    touch_stats(s);
    return 0;
}

int on_itlb_load_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    s->itlb_load_misses++;
    touch_stats(s);
    return 0;
}

/* ------------------------------------------------------------------ */
/* LBR 采样                                                             */
/* ------------------------------------------------------------------ */

int on_lbr_sample(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    struct perf_branch_entry branches[MAX_LBR_ENTRIES] = {};
    long bytes = bpf_read_branch_records(ctx, branches, sizeof(branches), 0);
    u8 nr = 0;
    if (bytes > 0) {
        nr = bytes / sizeof(branches[0]);
        if (nr > MAX_LBR_ENTRIES)
            nr = MAX_LBR_ENTRIES;
    }

    s->lbr_samples++;
    s->lbr_entries += nr;
    touch_stats(s);

    emit_lbr(ctx, pid, tid, PT_REGS_IP(&ctx->regs), nr, branches);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Page Fault 追踪（kprobe/handle_mm_fault）                            */
/* ------------------------------------------------------------------ */

int on_page_fault(struct pt_regs *ctx,
                  struct vm_area_struct *vma,
                  unsigned long address,
                  unsigned int flags)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!task_allowed(pid, tid)) return 0;

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) return 0;

    /* FAULT_FLAG_MAJOR = 0x400 */
    bool is_major = (flags & 0x400) != 0;
    if (is_major)
        s->major_faults++;
    else
        s->minor_faults++;
    touch_stats(s);

    emit(pid, tid, is_major ? 5 : 4, address, PT_REGS_IP(ctx));
    return 0;
}
