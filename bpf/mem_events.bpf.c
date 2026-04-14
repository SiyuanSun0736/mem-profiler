// SPDX-License-Identifier: GPL-2.0
/*
 * mem_events.bpf.c — 细粒度进程访存事件 eBPF 内核程序（CO-RE 版本）
 *
 * 追踪目标：
 *   • LLC load/store 访问与 miss（perf_event 硬件采样）
 *   • dTLB load/store 访问与 miss（perf_event 硬件采样）
 *   • iTLB load 访问与 miss（perf_event 硬件采样）
 *   • LBR 分支栈采样（perf_event, bpf_read_branch_records, 需内核 >= 5.8）
 *   • minor / major page fault（kprobe/handle_mm_fault）
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

/*
 * 更新公共字段。comm 只在 slot 为空时刷新（避免每次采样都拷贝 16 字节）。
 */
static __always_inline void touch_stats(struct pid_mem_stats *s)
{
    __sync_fetch_and_add(&s->samples, 1);
    s->last_seen_ns = bpf_ktime_get_ns();
    if (s->comm[0] == '\0')
        bpf_get_current_comm(s->comm, sizeof(s->comm));
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
/* LLC Access / Miss                                                    */
/* ================================================================== */

SEC("perf_event")
int on_llc_load(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->llc_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->llc_load_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_LOAD_MISS, ip, ip);
    return 0;
}

SEC("perf_event")
int on_llc_store(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->llc_stores, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->llc_store_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_STORE_MISS, ip, ip);
    return 0;
}

/* ================================================================== */
/* dTLB / iTLB Access / Miss                                            */
/* ================================================================== */

SEC("perf_event")
int on_dtlb_load(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->dtlb_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_dtlb_load_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->dtlb_load_misses, 1);
    __sync_fetch_and_add(&s->dtlb_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_DTLB_MISS, ip, ip);
    return 0;
}

SEC("perf_event")
int on_dtlb_store(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->dtlb_stores, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_dtlb_store_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->dtlb_store_misses, 1);
    __sync_fetch_and_add(&s->dtlb_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_DTLB_MISS, ip, ip);
    return 0;
}

SEC("perf_event")
int on_itlb_load(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->itlb_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_itlb_load_miss(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg) return 0;
    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;
    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;
    __sync_fetch_and_add(&s->itlb_load_misses, 1);
    touch_stats(s);
    return 0;
}

/* ================================================================== */
/* LBR 分支栈采样（需内核 >= 5.8，perf_event branch_sample_type）       */
/* ================================================================== */

SEC("perf_event")
int on_lbr_sample(struct bpf_perf_event_data *ctx)
{
    struct collector_config *cfg = cfg_get();
    if (!cfg || !cfg->enable_lbr)
        return 0;

    __u64 pidtid = bpf_get_current_pid_tgid();
    __u32 pid = pidtid >> 32, tid = (__u32)pidtid;
    if (!pid_allowed(pid, tid, cfg)) return 0;

    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s) return 0;

    struct perf_branch_entry branches[MAX_LBR_ENTRIES] = {};
    long bytes = bpf_read_branch_records(ctx, branches, sizeof(branches), 0);
    __u8 nr = 0;
    if (bytes > 0) {
        nr = (__u8)((__u64)bytes / sizeof(branches[0]));
        if (nr > MAX_LBR_ENTRIES)
            nr = MAX_LBR_ENTRIES;
    }

    __sync_fetch_and_add(&s->lbr_samples, 1);
    __sync_fetch_and_add(&s->lbr_entries, nr);
    touch_stats(s);

    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_lbr(cfg, ctx, pid, tid, ip, nr, branches);
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

    if (!pid_allowed(pid, tid, cfg))
        return 0;

    struct entity_key key = make_entity_key(pid, tid, cfg);
    struct pid_mem_stats *s = stats_get_or_init(&key);
    if (!s)
        return 0;

    /* FAULT_FLAG_MAJOR = 0x400（见 include/linux/mm_types.h） */
    bool is_major = (flags & 0x400) != 0;
    if (is_major)
        __sync_fetch_and_add(&s->major_faults, 1);
    else
        __sync_fetch_and_add(&s->minor_faults, 1);
    touch_stats(s);

    __u64 ip = PT_REGS_IP(ctx);
    __u8 etype = is_major ? MEM_EVENT_MAJOR_FAULT : MEM_EVENT_MINOR_FAULT;
    emit_event(cfg, pid, tid, etype, address, ip);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
