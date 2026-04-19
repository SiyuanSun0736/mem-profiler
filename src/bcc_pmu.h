/*
 * bcc_pmu.h — CPU cycles / instructions perf_event handler
 *
 * 每次 handler 触发代表 sample_period 个硬件事件已发生，
 * 通过累加 ctx->sample_period 得到窗口内的近似事件计数。
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

int on_cycles(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->cycles += ctx->sample_period;
    touch_stats(s);
    return 0;
}

int on_instructions(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();
    s->instructions += ctx->sample_period;
    touch_stats(s);
    return 0;
}
