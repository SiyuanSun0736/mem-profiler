/*
 * bcc_fault.h — minor/major page fault kprobe handler
 *
 * 挂载点：kprobe/handle_mm_fault
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

int on_page_fault(struct pt_regs *ctx,
                  struct vm_area_struct *vma,
                  unsigned long address,
                  unsigned int flags)
{
    BCC_PROLOGUE();

    /* FAULT_FLAG_MAJOR = 0x400 */
    bool is_major = (flags & 0x400) != 0;
    if (is_major)
        s->major_faults++;
    else
        s->minor_faults++;
    touch_stats(s);

    emit(pid, tid, is_major ? 5 : 4, address, PT_REGS_IP(ctx));
    return 0;
}
