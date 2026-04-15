/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_llc.bpf.h — LLC load/store 访问与 miss perf_event handler
 *
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 */
#pragma once

SEC("perf_event")
int on_llc_load(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->llc_loads, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->llc_load_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_LOAD_MISS, ip, ip);
    return 0;
}

SEC("perf_event")
int on_llc_store(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->llc_stores, 1);
    touch_stats(s);
    return 0;
}

SEC("perf_event")
int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE();
    __sync_fetch_and_add(&s->llc_store_misses, 1);
    touch_stats(s);
    __u64 ip = PT_REGS_IP(&ctx->regs);
    emit_event(cfg, pid, tid, MEM_EVENT_LLC_STORE_MISS, ip, ip);
    return 0;
}
