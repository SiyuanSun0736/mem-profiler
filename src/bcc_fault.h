/*
 * bcc_fault.h — enhanced page fault kprobe + kretprobe handler
 *
 * 挂载点：kprobe/handle_mm_fault + kretprobe/handle_mm_fault
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

static __always_inline u32 classify_fault_flags(u64 vma_flags,
                                                u32 fault_flags,
                                                bool file_backed)
{
    u32 class_flags = file_backed ? MEM_CLASS_FILE : MEM_CLASS_ANON;

    if (vma_flags & VM_SHARED)
        class_flags |= MEM_CLASS_SHARED;
    else
        class_flags |= MEM_CLASS_PRIVATE;

    if (fault_flags & FAULT_FLAG_WRITE)
        class_flags |= MEM_CLASS_WRITE;
    if (fault_flags & FAULT_FLAG_INSTRUCTION)
        class_flags |= MEM_CLASS_EXEC;

    return class_flags;
}

int on_page_fault(struct pt_regs *ctx,
                  struct vm_area_struct *vma,
                  unsigned long address,
                  unsigned int flags)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_fault_t pending = {
        .address = address,
        .ip = PT_REGS_IP(ctx),
        .vma_flags = 0,
        .fault_flags = flags,
        .class_flags = 0,
    };

    if (!task_allowed(pid, tid))
        return 0;

    if (fault_classification_enabled() && vma) {
        u64 vma_flags = 0;
        struct file *vm_file = NULL;

        bpf_probe_read_kernel(&vma_flags, sizeof(vma_flags), &vma->vm_flags);
        bpf_probe_read_kernel(&vm_file, sizeof(vm_file), &vma->vm_file);

        pending.vma_flags = vma_flags;
        pending.class_flags = classify_fault_flags(vma_flags, flags, vm_file != NULL);
    }

    pending_fault_args.update(&tid, &pending);
    return 0;
}

int on_page_fault_return(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_fault_t *pending = pending_fault_args.lookup(&tid);

    if (!pending)
        return 0;

    vm_fault_t ret = (vm_fault_t)PT_REGS_RC(ctx);
    if (ret & VM_FAULT_RETRY) {
        pending_fault_args.delete(&tid);
        return 0;
    }

    if (!task_allowed(pid, tid)) {
        pending_fault_args.delete(&tid);
        return 0;
    }

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) {
        pending_fault_args.delete(&tid);
        return 0;
    }

    bool is_major = (ret & VM_FAULT_MAJOR) != 0;
    if (is_major)
        s->major_faults++;
    else
        s->minor_faults++;

    if (fault_classification_enabled()) {
        if (pending->class_flags & MEM_CLASS_ANON)
            s->anon_faults++;
        if (pending->class_flags & MEM_CLASS_FILE)
            s->file_faults++;
        if (pending->class_flags & MEM_CLASS_SHARED)
            s->shared_faults++;
        if (pending->class_flags & MEM_CLASS_PRIVATE)
            s->private_faults++;
        if (pending->class_flags & MEM_CLASS_WRITE)
            s->write_faults++;
        if (pending->class_flags & MEM_CLASS_EXEC)
            s->instruction_faults++;
    }

    touch_stats(s);

    emit_fault_event(pid,
                     tid,
                     is_major ? MEM_EVENT_MAJOR_FAULT : MEM_EVENT_MINOR_FAULT,
                     pending->address,
                     pending->ip,
                     pending->fault_flags,
                     pending->class_flags,
                     pending->vma_flags);

    pending_fault_args.delete(&tid);
    return 0;
}
