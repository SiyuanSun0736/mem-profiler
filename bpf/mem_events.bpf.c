// SPDX-License-Identifier: GPL-2.0
/*
 * mem_events.bpf.c — 细粒度进程访存事件 eBPF 内核程序（CO-RE 版本）
 *
 * 追踪目标：
 *   • LLC load / store miss（通过 perf_event 硬件采样）
 *   • dTLB load miss（通过 perf_event 硬件采样）
 *   • minor / major page fault（通过 kprobe/handle_mm_fault）
 *
 * 编译方式：
 *   make          （需要 clang >= 12、libbpf、bpftool、linux-headers）
 *
 * 用户态联动：
 *   bcc 原型  → src/collector.py（加载 src/bcc_prog.c）
 *   libbpf    → 使用 make 生成的 bpf/mem_events.skel.h
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "mem_events.h"

/* ================================================================== */
/* Maps                                                                 */
/* ================================================================== */

/* 每 PID 累积统计（PERCPU 减少锁竞争，用户态读取后跨 CPU 求和） */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 8192);
    __type(key,   __u32);                  /* pid */
    __type(value, struct pid_mem_stats);
} pid_stats SEC(".maps");

/* 逐事件 ring buffer（仅在 emit_events=1 时写入） */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);          /* 16 MiB */
} events_rb SEC(".maps");

/* 采集器配置（key=0 存放 struct collector_config） */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key,   __u32);
    __type(value, struct collector_config);
} config_map SEC(".maps");

/* ================================================================== */
/* 内联辅助函数                                                         */
/* ================================================================== */

static __always_inline struct collector_config *cfg_get(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&config_map, &key);
}

static __always_inline bool pid_allowed(__u32 pid, const struct collector_config *cfg)
{
    return cfg->target_pid == 0 || cfg->target_pid == pid;
}

static __always_inline struct pid_mem_stats *stats_get_or_init(__u32 pid)
{
    struct pid_mem_stats *s = bpf_map_lookup_elem(&pid_stats, &pid);
    if (!s) {
        struct pid_mem_stats zero = {};
        bpf_get_current_comm(zero.comm, sizeof(zero.comm));
        bpf_map_update_elem(&pid_stats, &pid, &zero, BPF_NOEXIST);
        s = bpf_map_lookup_elem(&pid_stats, &pid);
    }
    return s;
}

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
    ev->addr       = addr;
    ev->ip         = ip;
    bpf_get_current_comm(ev->comm, sizeof(ev->comm));
    bpf_ringbuf_submit(ev, 0);
}

/* ================================================================== */
/* LLC Load Miss 采样（perf_event HARDWARE CACHE_MISSES）               */
/* ================================================================== */

SEC("perf_event")
int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg)
        return 0;

    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid    = pidtid >> 32;
    __u32 tid    = (__u32)pidtid;

    if (!pid_allowed(pid, cfg))
        return 0;

    struct pid_mem_stats *s = stats_get_or_init(pid);
    if (!s)
        return 0;

    __sync_fetch_and_add(&s->llc_load_misses, 1);
    __sync_fetch_and_add(&s->samples, 1);
    s->last_seen_ns = bpf_ktime_get_ns();

    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_LOAD_MISS, ip, ip);
    return 0;
}

/* ================================================================== */
/* LLC Store Miss 采样                                                  */
/* ================================================================== */

SEC("perf_event")
int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg)
        return 0;

    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid    = pidtid >> 32;
    __u32 tid    = (__u32)pidtid;

    if (!pid_allowed(pid, cfg))
        return 0;

    struct pid_mem_stats *s = stats_get_or_init(pid);
    if (!s)
        return 0;

    __sync_fetch_and_add(&s->llc_store_misses, 1);
    __sync_fetch_and_add(&s->samples, 1);

    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_STORE_MISS, ip, ip);
    return 0;
}

/* ================================================================== */
/* dTLB Load Miss 采样                                                  */
/* ================================================================== */

SEC("perf_event")
int on_dtlb_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg)
        return 0;

    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid    = pidtid >> 32;
    __u32 tid    = (__u32)pidtid;

    if (!pid_allowed(pid, cfg))
        return 0;

    struct pid_mem_stats *s = stats_get_or_init(pid);
    if (!s)
        return 0;

    __sync_fetch_and_add(&s->dtlb_misses, 1);

    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_DTLB_MISS, ip, ip);
    return 0;
}

/* ================================================================== */
/* Page Fault 追踪（kprobe/handle_mm_fault）                            */
/* ================================================================== */

SEC("kprobe/handle_mm_fault")
int BPF_KPROBE(on_page_fault,
               struct vm_area_struct *vma,
               unsigned long address,
               unsigned int flags)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg)
        return 0;

    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid    = pidtid >> 32;
    __u32 tid    = (__u32)pidtid;

    if (!pid_allowed(pid, cfg))
        return 0;

    struct pid_mem_stats *s = stats_get_or_init(pid);
    if (!s)
        return 0;

    /* FAULT_FLAG_MAJOR = 0x400（见 include/linux/mm_types.h） */
    bool is_major = (flags & 0x400) != 0;
    if (is_major)
        __sync_fetch_and_add(&s->major_faults, 1);
    else
        __sync_fetch_and_add(&s->minor_faults, 1);

    __u64 ip = PT_REGS_IP(ctx);
    __u8 etype = is_major ? MEM_EVENT_MAJOR_FAULT : MEM_EVENT_MINOR_FAULT;
    emit_event(cfg, pid, tid, etype, address, ip);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
