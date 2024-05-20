# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022 University of California, Riverside

ifneq ($(shell pkg-config --exists libconfig && echo 0), 0)
$(error "libconfig is not installed")
endif

CFLAGS = $(shell pkg-config --cflags libconfig)
LDFLAGS = $(shell pkg-config --libs-only-L libconfig)
LDLIBS = $(shell pkg-config --libs-only-l libconfig)

CFLAGS += -MMD -MP -O3 -Wall -Werror
LDLIBS += -lbpf -lm -pthread

CLANG = clang
CLANGFLAGS = -g -O2
BPF_FLAGS = -target bpf

.PHONY: all clean

all: bin sockmap_manager ebpf/sk_msg_kern.o

ebpf/sk_msg_kern.o: ebpf/sk_msg_kern.c
	@ $(CLANG) $(CLANGFLAGS) $(BPF_FLAGS) -c -o $@ $<

sockmap_manager: bin/sockmap_manager

bin/sockmap_manager: src/sockmap_manager.o
	@ echo "CC $@"
	@ $(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

-include $(patsubst %.o, %.d, $(wildcard src/*.o))

%.o: %.c
	@ echo "CC $@"
	@ $(CC) -c $(CFLAGS) -o $@ $<

bin:
	@ mkdir -p $@

clean:
	@ echo "RM -r src/*.d src/*.o bin"
	@ $(RM) -r src/*.d src/*.o bin ebpf/*.o