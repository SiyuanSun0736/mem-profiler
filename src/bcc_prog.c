/*
 * bcc_prog.c — BCC 兼容版 eBPF 程序（P1 原型阶段使用）
 *
 * 与 bpf/mem_events.bpf.c（CO-RE/libbpf 版本）功能等价，
 * 但使用 BCC 宏和标准内核头，可由 src/collector.py 通过 BCC Python 接口
 * 在运行时动态编译加载，无需预先 make。
 *
 * 生产环境建议迁移到 CO-RE 版本（bpf/mem_events.bpf.c + bpf/mem_events.skel.h）。
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/mm_types.h>
#include <linux/mm.h>

/*
 * BCC 的 helpers.h 只前置声明了 struct bpf_perf_event_data，
 * 但 perf_event handler 里需要读取 regs。这里补一个与内核公开 UAPI
 * 一致的最小定义，避免依赖 CO-RE 的 vmlinux.h。
 */
struct bpf_perf_event_data {
    struct pt_regs regs;
    u64 sample_period;
    u64 addr;
};

/* ------------------------------------------------------------------ */
/* 共享结构体（与 bpf/mem_events.h 保持字段语义一致）                   */
/* ------------------------------------------------------------------ */

struct pid_mem_stats_t {
    u64 llc_load_misses;
    u64 llc_store_misses;
    u64 dtlb_misses;
    u64 minor_faults;
    u64 major_faults;
    u64 samples;
    u64 last_seen_ns;
    char comm[TASK_COMM_LEN];
};

struct mem_event_t {
    u64 ts_ns;
    u32 pid;
    u32 tid;
    char comm[TASK_COMM_LEN];
    u8  event_type; /* 1=llc_load 2=llc_store 3=dtlb 4=minor_fault 5=major_fault */
    u64 addr;
    u64 ip;
};

/* ------------------------------------------------------------------ */
/* Maps                                                                 */
/* ------------------------------------------------------------------ */

BPF_PERCPU_HASH(pid_stats, u32, struct pid_mem_stats_t, 8192);
BPF_RINGBUF_OUTPUT(events_rb, 256);   /* 256 pages ≈ 1 MiB */

/* target_pid_map[0] = 目标 PID；0 表示采集所有进程 */
BPF_ARRAY(target_pid_map, u32, 1);

/* ------------------------------------------------------------------ */
/* 内联辅助                                                             */
/* ------------------------------------------------------------------ */

static __always_inline bool pid_allowed(u32 pid)
{
    u32 key = 0;
    u32 *tpid = target_pid_map.lookup(&key);
    return (!tpid || *tpid == 0 || *tpid == pid);
}

static __always_inline struct pid_mem_stats_t *get_or_init(u32 pid)
{
    struct pid_mem_stats_t zero = {};
    bpf_get_current_comm(zero.comm, sizeof(zero.comm));
    pid_stats.insert(&pid, &zero);   /* 已存在则忽略 */
    return pid_stats.lookup(&pid);
}

static __always_inline void emit(u32 pid, u32 tid, u8 etype, u64 addr, u64 ip)
{
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

/* ------------------------------------------------------------------ */
/* LLC Load Miss 采样                                                   */
/* ------------------------------------------------------------------ */

int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!pid_allowed(pid)) return 0;

    struct pid_mem_stats_t *s = get_or_init(pid);
    if (!s) return 0;
    s->llc_load_misses++;
    s->samples++;
    s->last_seen_ns = bpf_ktime_get_ns();
    bpf_get_current_comm(s->comm, sizeof(s->comm));

    u64 ip = PT_REGS_IP(&ctx->regs);
    emit(pid, tid, 1, ip, ip);
    return 0;
}

/* ------------------------------------------------------------------ */
/* LLC Store Miss 采样                                                  */
/* ------------------------------------------------------------------ */

int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!pid_allowed(pid)) return 0;

    struct pid_mem_stats_t *s = get_or_init(pid);
    if (!s) return 0;
    s->llc_store_misses++;
    s->samples++;
    emit(pid, tid, 2, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

/* ------------------------------------------------------------------ */
/* dTLB Miss 采样                                                       */
/* ------------------------------------------------------------------ */

int on_dtlb_miss(struct bpf_perf_event_data *ctx)
{
    u64 pidtid = bpf_get_current_pid_tgid();
    u32 pid = pidtid >> 32, tid = (u32)pidtid;
    if (!pid_allowed(pid)) return 0;

    struct pid_mem_stats_t *s = get_or_init(pid);
    if (!s) return 0;
    s->dtlb_misses++;
    emit(pid, tid, 3, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
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
    if (!pid_allowed(pid)) return 0;

    struct pid_mem_stats_t *s = get_or_init(pid);
    if (!s) return 0;

    /* FAULT_FLAG_MAJOR = 0x400 */
    bool is_major = (flags & 0x400) != 0;
    if (is_major)
        s->major_faults++;
    else
        s->minor_faults++;

    emit(pid, tid, is_major ? 5 : 4, address, PT_REGS_IP(ctx));
    return 0;
}
