/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_tlb.bpf.h — dTLB/iTLB 访问与 miss perf_event handler
 *
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 */
#pragma once

/* ------------------------------------------------------------------ */
/* dTLB                                                                 */
/* ------------------------------------------------------------------ */

SEC("perf_event")
int on_dtlb_load(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->dtlb_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_dtlb_load_miss(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
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
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->dtlb_stores, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_dtlb_store_miss(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->dtlb_store_misses, 1);
    __sync_fetch_and_add(&s->dtlb_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_DTLB_MISS, ip, ip);
    return 0;
}

/* ------------------------------------------------------------------ */
/* iTLB                                                                 */
/* ------------------------------------------------------------------ */

SEC("perf_event")
int on_itlb_load(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->itlb_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_itlb_load_miss(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->itlb_load_misses, 1);
    touch_stats(s);
    return 0;
}
