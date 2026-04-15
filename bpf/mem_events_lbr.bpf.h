/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_lbr.bpf.h — LBR 分支栈采样 perf_event handler
 *
 * 需要内核 >= 5.8（bpf_read_branch_records helper）。
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 */
#pragma once

SEC("perf_event")
int on_lbr_sample(struct bpf_perf_event_data *ctx)
{
    HANDLER_PROLOGUE_LBR();

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
