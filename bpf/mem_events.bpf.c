// SPDX-License-Identifier: GPL-2.0
/*
 * mem_events.bpf.c — 细粒度进程访存事件 eBPF 内核程序（CO-RE 版本）
 *
 * 追踪目标：
 *   • LLC load/store 访问与 miss（perf_event 硬件采样）
 *   • dTLB load/store 访问与 miss（perf_event 硬件采样）
 *   • iTLB load 访问与 miss（perf_event 硬件采样）
 *   • LBR 分支栈采样（perf_event, bpf_read_branch_records, 需内核 >= 5.8）
 *   • minor / major page fault（kprobe/handle_mm_fault）
 *
 * 编译方式：
 *   make          （需要 clang >= 12、libbpf、bpftool、linux-headers）
 *
 * 用户态联动：
 *   bcc 原型  → src/collector.py（加载 src/bcc_prog.c）
 *   libbpf    → 使用 make 生成的 bpf/mem_events.skel.h
 *
 * 文件组织：
 *   mem_events_maps.bpf.h     — BPF map 定义
 *   mem_events_helpers.bpf.h  — 内联辅助函数 + HANDLER_PROLOGUE 宏
 *   mem_events_llc.bpf.h      — LLC load/store handler
 *   mem_events_tlb.bpf.h      — dTLB/iTLB handler
 *   mem_events_lbr.bpf.h      — LBR 分支栈采样 handler
 *   mem_events_fault.bpf.h    — page fault kprobe handler
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "mem_events.h"

#include "mem_events_maps.bpf.h"
#include "mem_events_helpers.bpf.h"
#include "mem_events_llc.bpf.h"
#include "mem_events_tlb.bpf.h"
#include "mem_events_lbr.bpf.h"
#include "mem_events_fault.bpf.h"

char LICENSE[] SEC("license") = "GPL";
