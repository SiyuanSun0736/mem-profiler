/*
 * bcc_types.h — BCC eBPF 程序共享类型定义
 *
 * 包含：内核 helper 前置声明、共享结构体。
 * 被 bcc_prog.c 通过 #include 引入；不可单独编译。
 */
#pragma once

#include <uapi/linux/ptrace.h>
#include <uapi/linux/perf_event.h>
#include <linux/sched.h>
#include <linux/mm_types.h>
#include <linux/mm.h>

#ifndef VM_FAULT_MAJOR
#define VM_FAULT_MAJOR 0x00000004
#endif

#ifndef VM_FAULT_RETRY
#define VM_FAULT_RETRY 0x00000400
#endif

#ifndef FAULT_FLAG_WRITE
#define FAULT_FLAG_WRITE 0x00000001
#endif

#ifndef FAULT_FLAG_INSTRUCTION
#define FAULT_FLAG_INSTRUCTION 0x00000100
#endif

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x20
#endif

#ifndef MAP_SHARED
#define MAP_SHARED 0x01
#endif

#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x02
#endif

#ifndef PROT_WRITE
#define PROT_WRITE 0x2
#endif

#ifndef PROT_EXEC
#define PROT_EXEC 0x4
#endif

#ifndef PT_REGS_SYSCALL_REGS
#define PT_REGS_SYSCALL_REGS(ctx) ((struct pt_regs *)PT_REGS_PARM1(ctx))
#endif

#ifndef PT_REGS_PARM1_SYSCALL
#define PT_REGS_PARM1_SYSCALL(x) PT_REGS_PARM1(x)
#endif

#ifndef PT_REGS_PARM2_SYSCALL
#define PT_REGS_PARM2_SYSCALL(x) PT_REGS_PARM2(x)
#endif

#ifndef PT_REGS_PARM3_SYSCALL
#define PT_REGS_PARM3_SYSCALL(x) PT_REGS_PARM3(x)
#endif

#ifndef PT_REGS_PARM4_SYSCALL
#if defined(__x86_64__)
#define PT_REGS_PARM4_SYSCALL(x) ((x)->r10)
#else
#define PT_REGS_PARM4_SYSCALL(x) PT_REGS_PARM4(x)
#endif
#endif

/*
 * BCC 的 helpers.h 只前置声明了 struct bpf_perf_event_data，
 * 但 perf_event handler 里需要读取 regs 和 branch records。这里补一个
 * 与内核公开 UAPI 一致的最小定义，避免依赖 CO-RE 的 vmlinux.h。
 */
struct bpf_perf_event_data {
    struct pt_regs regs;
    u64 sample_period;
    u64 addr;
};

/*
 * bpf_read_branch_records, bpf_perf_event_value, bpf_perf_prog_read_value
 * are already declared by BCC's helpers.h / bpf.h with int (*) return type.
 * Do not re-declare them here to avoid redefinition conflicts.
 */

#define MAX_LBR_ENTRIES 8

#define MEM_EVENT_LLC_LOAD_MISS   1
#define MEM_EVENT_LLC_STORE_MISS  2
#define MEM_EVENT_DTLB_MISS       3
#define MEM_EVENT_MINOR_FAULT     4
#define MEM_EVENT_MAJOR_FAULT     5
#define MEM_EVENT_LBR             6
#define MEM_EVENT_MMAP            7
#define MEM_EVENT_MUNMAP          8
#define MEM_EVENT_MPROTECT        9
#define MEM_EVENT_BRK             10

#define MEM_CLASS_ANON      (1U << 0)
#define MEM_CLASS_FILE      (1U << 1)
#define MEM_CLASS_SHARED    (1U << 2)
#define MEM_CLASS_PRIVATE   (1U << 3)
#define MEM_CLASS_WRITE     (1U << 4)
#define MEM_CLASS_EXEC      (1U << 5)

/* ------------------------------------------------------------------ */
/* 共享结构体（与用户态保持字段语义一致）                               */
/* ------------------------------------------------------------------ */

struct entity_key_t {
    u32 pid;
    u32 tid;
};

struct task_comm_filter_t {
    char comm[TASK_COMM_LEN];
};

struct pending_fault_t {
    u64 address;
    u64 ip;
    u64 vma_flags;
    u32 fault_flags;
    u32 class_flags;
};

struct pending_mmap_args_t {
    u64 requested_addr;
    u64 length;
    u32 prot;
    u32 flags;
};

struct pending_range_args_t {
    u64 addr;
    u64 length;
    u32 prot;
    u32 _pad;
};

struct pending_brk_args_t {
    u64 requested_addr;
};

struct pid_mem_stats_t {
    u64 llc_loads;
    u64 llc_load_misses;
    u64 llc_stores;
    u64 llc_store_misses;
    u64 dtlb_loads;
    u64 dtlb_load_misses;
    u64 dtlb_stores;
    u64 dtlb_store_misses;
    u64 dtlb_misses;
    u64 itlb_load_misses;
    u64 cycles;
    u64 instructions;
    u64 minor_faults;
    u64 major_faults;
    u64 anon_faults;
    u64 file_faults;
    u64 shared_faults;
    u64 private_faults;
    u64 write_faults;
    u64 instruction_faults;
    u64 mmap_calls;
    u64 munmap_calls;
    u64 mprotect_calls;
    u64 brk_calls;
    u64 mmap_bytes;
    u64 munmap_bytes;
    u64 mprotect_bytes;
    u64 brk_growth_bytes;
    u64 brk_shrink_bytes;
    u64 lbr_samples;
    u64 lbr_entries;
    u64 samples;
    u64 last_seen_ns;
    char comm[TASK_COMM_LEN];
};

struct lbr_entry_t {
    u64 from_ip;
    u64 to_ip;
    u64 flags;
};

struct lbr_branch_scratch_t {
    struct perf_branch_entry entries[MAX_LBR_ENTRIES];
};

struct mem_event_t {
    u64 ts_ns;
    u32 pid;
    u32 tid;
    char comm[TASK_COMM_LEN];
    u8  event_type;
    u8  lbr_nr;
    u16 _pad0;
    u32 prot;
    u32 event_flags;
    u32 class_flags;
    u32 _pad1;
    u64 addr;
    u64 ip;
    u64 length;
    u64 requested_addr;
    u64 vma_flags;
    s64 delta_bytes;
    struct lbr_entry_t lbr[MAX_LBR_ENTRIES];
};
