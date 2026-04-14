# eBPF CO-RE 程序构建
#
# 依赖: clang >= 12, llvm, libbpf-dev, linux-headers-$(uname -r), bpftool
#   Ubuntu/Debian: sudo apt install clang llvm libbpf-dev linux-headers-$(uname -r) bpftool

CLANG      ?= clang
BPFTOOL    ?= bpftool
ARCH       ?= $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')
KERNEL_BTF ?= /sys/kernel/btf/vmlinux

# 包含路径：优先系统 libbpf，其次 bpf/
BPF_CFLAGS  = -g -O2 -target bpf \
              -D__TARGET_ARCH_$(ARCH) \
			  -Wno-missing-declarations \
              -I/usr/include/$(shell uname -m)-linux-gnu \
              -Ibpf

BPF_SRC     = bpf/mem_events.bpf.c
BPF_OBJ     = bpf/mem_events.bpf.o
SKEL_HEADER = bpf/mem_events.skel.h
VMLINUX_H   = bpf/vmlinux.h

.PHONY: all clean vmlinux help

all: vmlinux $(BPF_OBJ) $(SKEL_HEADER)
	@echo "[OK]   构建完成: $(BPF_OBJ)  $(SKEL_HEADER)"

vmlinux: $(VMLINUX_H)

$(VMLINUX_H):
	@echo "[GEN]  $@"
	$(BPFTOOL) btf dump file $(KERNEL_BTF) format c > $@

$(BPF_OBJ): $(BPF_SRC) bpf/mem_events.h $(VMLINUX_H)
	@echo "[BPF]  $@"
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

$(SKEL_HEADER): $(BPF_OBJ)
	@echo "[SKEL] $@"
	$(BPFTOOL) gen skeleton $< > $@

clean:
	rm -f $(BPF_OBJ) $(SKEL_HEADER) $(VMLINUX_H)
	@echo "[OK]   清理完成"

help:
	@echo "targets: all  clean  vmlinux  help"
	@echo ""
	@echo "  all     - 生成 vmlinux.h，编译 eBPF 对象，生成 skeleton header"
	@echo "  vmlinux - 仅生成 bpf/vmlinux.h（需要 bpftool 和 /sys/kernel/btf/vmlinux）"
	@echo "  clean   - 删除所有生成文件"
	@echo ""
	@echo "可覆盖的变量:"
	@echo "  CLANG=$(CLANG)"
	@echo "  BPFTOOL=$(BPFTOOL)"
	@echo "  ARCH=$(ARCH)"
	@echo "  KERNEL_BTF=$(KERNEL_BTF)"
