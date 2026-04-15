/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mem_events_fault.bpf.h — minor/major page fault kprobe handler
 *
 * 挂载点：kprobe/handle_mm_fault
 * 被 mem_events.bpf.c 通过 #include 引入；不可单独编译。
 */
#pragma once

SEC("kprobe/handle_mm_fault")
int BPF_KPROBE(on_page_fault,
               struct vm_area_struct *vma,
               unsigned long address,
               unsigned int flags)
{
    HANDLER_PROLOGUE();

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
