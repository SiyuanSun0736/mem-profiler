/*
 * bcc_tlb.h — dTLB/iTLB 访问与 miss perf_event handler
 *
 * 每次 handler 触发代表 sample_period 个底层硬件事件已发生，
 * 通过累加 ctx->sample_period 输出窗口内的近似事件计数。
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

/* ------------------------------------------------------------------ */
/* dTLB                                                                 */
/* ------------------------------------------------------------------ */

int on_dtlb_load(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->dtlb_loads += ctx->sample_period;
    touch_stats(s);
    return 0;
}

int on_dtlb_load_miss(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->dtlb_load_misses += ctx->sample_period;
    s->dtlb_misses += ctx->sample_period;
    touch_stats(s);
    emit(pid, tid, 3, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

int on_dtlb_store(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->dtlb_stores += ctx->sample_period;
    touch_stats(s);
    return 0;
}

int on_dtlb_store_miss(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->dtlb_store_misses += ctx->sample_period;
    s->dtlb_misses += ctx->sample_period;
    touch_stats(s);
    emit(pid, tid, 3, PT_REGS_IP(&ctx->regs), PT_REGS_IP(&ctx->regs));
    return 0;
}

/* ------------------------------------------------------------------ */
/* iTLB                                                                 */
/* ------------------------------------------------------------------ */

int on_itlb_load_miss(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->itlb_load_misses += ctx->sample_period;
    touch_stats(s);
    return 0;
}
