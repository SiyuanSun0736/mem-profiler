/*
 * bcc_llc.h — LLC load/store 访问与 miss perf_event handler
 *
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

int on_llc_load(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->llc_loads++;
    touch_stats(s);
    return 0;
}

int on_llc_load_miss(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->llc_load_misses++;
    touch_stats(s);
    u64 ip = PT_REGS_IP(&ctx->regs);
    emit(pid, tid, 1, ip, ip);
    return 0;
}

int on_llc_store(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->llc_stores++;
    touch_stats(s);
    return 0;
}

int on_llc_store_miss(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->llc_store_misses++;
    touch_stats(s);
    emit(pid, tid, 2, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}
