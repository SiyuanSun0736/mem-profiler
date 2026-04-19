/*
 * bcc_prog.c — BCC 兼容版 eBPF 程序（P1 原型阶段使用）
 *
 * 与 bpf/mem_events.bpf.c（CO-RE/libbpf 版本）功能等价，
 * 但使用 BCC 宏和标准内核头，可由 src/collector.py 通过 BCC Python 接口
 * 在运行时动态编译加载，无需预先 make。
 *
 * 本版本额外支持：
 *   • 更多 cache/TLB perf counter
 *   • 可选 per-TID 聚合 / TID 过滤
 *   • 可选 LBR 分支栈采样并通过 ring buffer 输出
 *
 * 文件组织（均位于 src/ 目录下）：
 *   bcc_types.h    — 内核 helper 前置声明 + 共享结构体
 *   bcc_maps.h     — BPF map 声明
 *   bcc_helpers.h  — 内联辅助函数 + BCC_PROLOGUE 宏
 *   bcc_llc.h      — LLC load/store handler
 *   bcc_tlb.h      — dTLB/iTLB handler
 *   bcc_lbr.h      — LBR 分支栈采样 handler
 *   bcc_fault.h    — page fault kprobe handler
 */

#include "bcc_types.h"
#include "bcc_maps.h"
#include "bcc_helpers.h"
#include "bcc_pmu.h"
#include "bcc_llc.h"
#include "bcc_tlb.h"
#include "bcc_lbr.h"
#include "bcc_fault.h"
