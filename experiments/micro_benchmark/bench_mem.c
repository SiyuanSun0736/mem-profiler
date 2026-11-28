#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#define LLC_BUF_SIZE (64 * 1024 * 1024)   // 64 MB (exceeds 16MB/32MB L3)
#define TLB_BUF_SIZE (32 * 1024 * 1024)   // 32 MB
#define PAGE_SIZE 4096

// volatile to prevent compiler optimization
volatile uint64_t g_sink = 0;

__attribute__((noinline))
void workload_random_llc(uint8_t *buf, size_t size, int iterations) {
    uint64_t sum = 0;
    uint32_t lcg = 123456789;
    for (int i = 0; i < iterations; i++) {
        lcg = lcg * 1664525 + 1013904223;
        size_t idx = (lcg % (size / 64)) * 64;
        sum += buf[idx];
    }
    g_sink += sum;
}

__attribute__((noinline))
void workload_stride_tlb(uint8_t *buf, size_t size, int iterations) {
    uint64_t sum = 0;
    size_t pages = size / PAGE_SIZE;
    for (int i = 0; i < iterations; i++) {
        size_t idx = (i % pages) * PAGE_SIZE;
        buf[idx] = (uint8_t)(buf[idx] + 1);
        sum += buf[idx];
    }
    g_sink += sum;
}

__attribute__((noinline))
void workload_page_fault(size_t alloc_size, int count) {
    uint64_t sum = 0;
    for (int c = 0; c < count; c++) {
        uint8_t *p = (uint8_t *)mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) continue;
        for (size_t off = 0; off < alloc_size; off += PAGE_SIZE) {
            p[off] = 0xAA;
            sum += p[off];
        }
        munmap(p, alloc_size);
    }
    g_sink += sum;
}

__attribute__((noinline))
void workload_cpu_compute(int iterations) {
    uint64_t val = 1;
    for (int i = 0; i < iterations; i++) {
        val = (val ^ (val << 13)) + i;
    }
    g_sink += val;
}

int main(int argc, char **argv) {
    int duration_sec = 0;
    int fixed_rounds = 0;
    int wait_sync = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            fixed_rounds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            duration_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--wait-sync") == 0) {
            wait_sync = 1;
        } else if (atoi(argv[i]) > 0 && fixed_rounds == 0 && duration_sec == 0) {
            duration_sec = atoi(argv[i]);
        }
    }
    if (fixed_rounds <= 0 && duration_sec <= 0) {
        duration_sec = 6;
    }

    if (wait_sync) {
        raise(SIGSTOP);
    }

    uint8_t *llc_buf = (uint8_t *)malloc(LLC_BUF_SIZE);
    uint8_t *tlb_buf = (uint8_t *)malloc(TLB_BUF_SIZE);
    if (!llc_buf || !tlb_buf) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    memset(llc_buf, 0x55, LLC_BUF_SIZE);
    memset(tlb_buf, 0x33, TLB_BUF_SIZE);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int round = 0;
    while (1) {
        if (fixed_rounds > 0) {
            if (round >= fixed_rounds) break;
        } else {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
            if (elapsed >= duration_sec) break;
        }

        workload_random_llc(llc_buf, LLC_BUF_SIZE, 3000000);
        workload_stride_tlb(tlb_buf, TLB_BUF_SIZE, 1000000);
        workload_page_fault(4 * 1024 * 1024, 8);
        workload_cpu_compute(5000000);
        round++;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double total_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

    printf("[bench_mem] Completed %d rounds in %.2f ms (sink=%lu)\n",
           round, total_ms, (unsigned long)g_sink);

    free(llc_buf);
    free(tlb_buf);
    return 0;
}
