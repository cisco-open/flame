# LIFL Instructions

This document provides instructions on how to use LIFL in flame.

## Prerequisites
The target runtime environment of LIFL is Linux **only**. LIFL requires Linux kernel version >= 5.15. We have tested LIFL on Ubuntu 20.

## Environment Setup

### 1. Upgrade kernel
*Note: if you have kernel version >=5.15, please skip this step*

```bash
# Execute the kernel upgrade script
cd third_party/spright_utility/scripts
./upgrade_kernel.sh
```

### 2. Install libbpf

```bash
# Install deps for libbpf
sudo apt update && sudo apt install -y flex bison build-essential dwarves libssl-dev \
                    libelf-dev pkg-config libconfig-dev clang gcc-multilib

# Execute the libbpf installation script
cd third_party/spright_utility/scripts
./libbpf.sh
```

## Shared Memory Backend in LIFL

The [shared memory backend](../../lib/python/flame/backend/shm.py) in LIFL uses eBPF's sockmap and SK_MSG to pass buffer references between aggregators. We introduce a "[sockmap_manager](../../third_party/spright_utility/src/sockmap_manager.c)" on each node to manage the registration of aggregator's socket to the in-kernel sockmap. You must run the `sockmap_manager` first.

```bash
# Execute the sockmap_manager
cd third_party/spright_utility/

sudo ./bin/sockmap_manager
```

To enable Shared Memory Backend in the channel, you need to add `shm` to the `brokers` field in the config:

```yaml
    "brokers": [
        {
            "host": "localhost",
            "sort": "mqtt"
        },
        {
            "host": "localhost:10104",
            "sort": "p2p"
        },
        {
            "host": "localhost:10105",
            "sort": "shm"
        }
    ],
```

You also need to specify the backend type of the channel to `shm` so that the channel will choose to use shared memory backend during its initialization.

```yaml
    "channels": [
        {
            "name": "top-agg-coord-channel",
            ...
        },
        {
            "name": "global-channel",
            ...
            "backend": "shm",
            ...
        }
    ],
```

We offer sample configs in the [coord_3_hier_syncfl_mnist](../../lib/python/examples/coord_3_hier_syncfl_mnist/) and [coord_hier_syncfl_mnist](../../lib/python/examples/coord_hier_syncfl_mnist/) examples.

## Hierarchical Aggregation in LIFL

Flame initially supports hierarchical aggregation with two levels: top level and leaf level. The example of two-level hierarchical aggregation is at [coord_hier_syncfl_mnist](../../lib/python/examples/coord_hier_syncfl_mnist/). LIFL extends hierarchical aggregation in Flame with three levels: top level, middle level, and leaf level. The example of three-level hierarchical aggregation is at [coord_3_hier_syncfl_mnist](../../lib/python/examples/coord_3_hier_syncfl_mnist/).

## Eager Aggregation in LIFL

Flame initially supports lazy aggregation only. LIFL adds additional support for having eager aggregation in Flame, which gives us more flexible timing on the aggregation process. The example to run eager aggregation is availble at [eager_hier_mnist](../../lib/python/examples/eager_hier_mnist/). The implementation of eager aggregation is available at [eager_syncfl](../../lib/python/flame/mode/horizontal/eager_syncfl/).

## Problems when running LIFL
1. When you run `sudo ./bin/sockmap_manager`, you receive 
```
./bin/sockmap_manager: error while loading shared libraries: libbpf.so.0: cannot open shared object file: No such file or directory
```

Solutions: This may happen when you use Ubuntu 22, which has the libbpf 0.5.0 pre-installed. You need to re-link the `/lib/x86_64-linux-gnu/libbpf.so.0` to `libbpf.so.0.6.0`
```bash
# Assume you have executed the libbpf installation script
cd third_party/spright_utility/scripts/libbpf/src

# Copy libbpf.so.0.6.0 to /lib/x86_64-linux-gnu/
sudo cp libbpf.so.0.6.0 /lib/x86_64-linux-gnu/

# Re-link libbpf.so.0
sudo ln -sf /lib/x86_64-linux-gnu/libbpf.so.0.6.0 /lib/x86_64-linux-gnu/libbpf.so.0
```