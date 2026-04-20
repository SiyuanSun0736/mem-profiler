/*
 * bcc_mm_syscalls.h — mmap/munmap/mprotect/brk kprobe + kretprobe handlers
 *
 * 挂载点：对应 syscall wrapper 的 entry/return。
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

#ifndef MM_SYSCALL_WRAPPER
#define MM_SYSCALL_WRAPPER 0
#endif

static __always_inline u32 classify_mmap_flags(u32 prot, u32 map_flags)
{
    u32 class_flags = 0;

    if (map_flags & MAP_ANONYMOUS)
        class_flags |= MEM_CLASS_ANON;
    else
        class_flags |= MEM_CLASS_FILE;

    if (map_flags & MAP_SHARED)
        class_flags |= MEM_CLASS_SHARED;
    else
        class_flags |= MEM_CLASS_PRIVATE;

    if (prot & PROT_WRITE)
        class_flags |= MEM_CLASS_WRITE;
    if (prot & PROT_EXEC)
        class_flags |= MEM_CLASS_EXEC;

    return class_flags;
}

static __always_inline u32 classify_prot_flags(u32 prot)
{
    u32 class_flags = 0;

    if (prot & PROT_WRITE)
        class_flags |= MEM_CLASS_WRITE;
    if (prot & PROT_EXEC)
        class_flags |= MEM_CLASS_EXEC;

    return class_flags;
}

static __always_inline u64 mm_syscall_arg1(struct pt_regs *ctx)
{
#if MM_SYSCALL_WRAPPER
    struct pt_regs *regs = (struct pt_regs *)PT_REGS_PARM1(ctx);
    u64 value = 0;
    bpf_probe_read_kernel(&value, sizeof(value), &regs->di);
    return value;
#else
    return PT_REGS_PARM1(ctx);
#endif
}

static __always_inline u64 mm_syscall_arg2(struct pt_regs *ctx)
{
#if MM_SYSCALL_WRAPPER
    struct pt_regs *regs = (struct pt_regs *)PT_REGS_PARM1(ctx);
    u64 value = 0;
    bpf_probe_read_kernel(&value, sizeof(value), &regs->si);
    return value;
#else
    return PT_REGS_PARM2(ctx);
#endif
}

static __always_inline u64 mm_syscall_arg3(struct pt_regs *ctx)
{
#if MM_SYSCALL_WRAPPER
    struct pt_regs *regs = (struct pt_regs *)PT_REGS_PARM1(ctx);
    u64 value = 0;
    bpf_probe_read_kernel(&value, sizeof(value), &regs->dx);
    return value;
#else
    return PT_REGS_PARM3(ctx);
#endif
}

static __always_inline u64 mm_syscall_arg4(struct pt_regs *ctx)
{
#if MM_SYSCALL_WRAPPER
    struct pt_regs *regs = (struct pt_regs *)PT_REGS_PARM1(ctx);
    u64 value = 0;
    bpf_probe_read_kernel(&value, sizeof(value), &regs->r10);
    return value;
#else
    return PT_REGS_PARM4(ctx);
#endif
}

int on_mmap_enter(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_mmap_args_t args = {
        .requested_addr = mm_syscall_arg1(ctx),
        .length = mm_syscall_arg2(ctx),
        .prot = (u32)mm_syscall_arg3(ctx),
        .flags = (u32)mm_syscall_arg4(ctx),
    };

    if (!task_allowed(pid, tid))
        return 0;

    pending_mmap_args.update(&tid, &args);
    return 0;
}

int on_mmap_return(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_mmap_args_t *args = pending_mmap_args.lookup(&tid);

    if (!args)
        return 0;

    long ret = PT_REGS_RC(ctx);
    if (ret < 0 || !task_allowed(pid, tid)) {
        pending_mmap_args.delete(&tid);
        return 0;
    }

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) {
        pending_mmap_args.delete(&tid);
        return 0;
    }

    s->mmap_calls++;
    s->mmap_bytes += args->length;
    touch_stats(s);

    emit_mm_syscall_event(pid,
                          tid,
                          MEM_EVENT_MMAP,
                          (u64)ret,
                          PT_REGS_IP(ctx),
                          args->requested_addr,
                          args->length,
                          args->prot,
                          args->flags,
                          classify_mmap_flags(args->prot, args->flags),
                          0);

    pending_mmap_args.delete(&tid);
    return 0;
}

int on_munmap_enter(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_range_args_t args = {
        .addr = mm_syscall_arg1(ctx),
        .length = mm_syscall_arg2(ctx),
        .prot = 0,
    };

    if (!task_allowed(pid, tid))
        return 0;

    pending_munmap_args.update(&tid, &args);
    return 0;
}

int on_munmap_return(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_range_args_t *args = pending_munmap_args.lookup(&tid);

    if (!args)
        return 0;

    long ret = PT_REGS_RC(ctx);
    if (ret != 0 || !task_allowed(pid, tid)) {
        pending_munmap_args.delete(&tid);
        return 0;
    }

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) {
        pending_munmap_args.delete(&tid);
        return 0;
    }

    s->munmap_calls++;
    s->munmap_bytes += args->length;
    touch_stats(s);

    emit_mm_syscall_event(pid,
                          tid,
                          MEM_EVENT_MUNMAP,
                          args->addr,
                          PT_REGS_IP(ctx),
                          0,
                          args->length,
                          0,
                          0,
                          0,
                          0);

    pending_munmap_args.delete(&tid);
    return 0;
}

int on_mprotect_enter(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_range_args_t args = {
        .addr = mm_syscall_arg1(ctx),
        .length = mm_syscall_arg2(ctx),
        .prot = (u32)mm_syscall_arg3(ctx),
    };

    if (!task_allowed(pid, tid))
        return 0;

    pending_mprotect_args.update(&tid, &args);
    return 0;
}

int on_mprotect_return(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_range_args_t *args = pending_mprotect_args.lookup(&tid);

    if (!args)
        return 0;

    long ret = PT_REGS_RC(ctx);
    if (ret != 0 || !task_allowed(pid, tid)) {
        pending_mprotect_args.delete(&tid);
        return 0;
    }

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) {
        pending_mprotect_args.delete(&tid);
        return 0;
    }

    s->mprotect_calls++;
    s->mprotect_bytes += args->length;
    touch_stats(s);

    emit_mm_syscall_event(pid,
                          tid,
                          MEM_EVENT_MPROTECT,
                          args->addr,
                          PT_REGS_IP(ctx),
                          0,
                          args->length,
                          args->prot,
                          0,
                          classify_prot_flags(args->prot),
                          0);

    pending_mprotect_args.delete(&tid);
    return 0;
}

int on_brk_enter(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_brk_args_t args = {
        .requested_addr = mm_syscall_arg1(ctx),
    };

    if (!task_allowed(pid, tid))
        return 0;

    pending_brk_args.update(&tid, &args);
    return 0;
}

int on_brk_return(struct pt_regs *ctx)
{
    u64 _pidtid = bpf_get_current_pid_tgid();
    u32 pid = _pidtid >> 32, tid = (u32)_pidtid;
    struct pending_brk_args_t *args = pending_brk_args.lookup(&tid);

    if (!args)
        return 0;

    u64 current_brk = (u64)PT_REGS_RC(ctx);
    if (!task_allowed(pid, tid) || current_brk == 0) {
        pending_brk_args.delete(&tid);
        return 0;
    }

    struct entity_key_t key = make_entity_key(pid, tid);
    struct pid_mem_stats_t *s = get_or_init(&key);
    if (!s) {
        pending_brk_args.delete(&tid);
        return 0;
    }

    s64 delta = 0;
    u64 *last_brk = last_brk_by_tgid.lookup(&pid);
    if (last_brk) {
        delta = (s64)current_brk - (s64)(*last_brk);
        if (delta > 0)
            s->brk_growth_bytes += (u64)delta;
        else if (delta < 0)
            s->brk_shrink_bytes += (u64)(-delta);
    }

    last_brk_by_tgid.update(&pid, &current_brk);
    s->brk_calls++;
    touch_stats(s);

    emit_mm_syscall_event(pid,
                          tid,
                          MEM_EVENT_BRK,
                          current_brk,
                          PT_REGS_IP(ctx),
                          args->requested_addr,
                          0,
                          0,
                          0,
                          0,
                          delta);

    pending_brk_args.delete(&tid);
    return 0;
}
