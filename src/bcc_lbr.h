/*
 * bcc_lbr.h — LBR 分支栈采样 perf_event handler
 *
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

int on_lbr_sample(struct bpf_perf_event_data *ctx)
{
    BCC_PROLOGUE();

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
