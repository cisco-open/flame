# Flame Examples

This directory contains examples demonstrating various federated learning scenarios, algorithms, and topologies using the Flame framework.

> **New here?** For environment setup and copy-paste commands to run
> async_cifar10 / async_google_speech / fwdllm, start with
> [`lib/python/README.md`](../README.md).

## Running experiments via the YAML launcher

Most examples can be driven by `flame.launch`, which reads a single experiment
YAML and spawns one aggregator plus N trainers:

```bash
python -m flame.launch.run_experiment \
    lib/python/examples/feddance_cifar10/experiments/configs/smoke_10trainer.yaml
```

The launcher resolves the example dir from the YAML path (or `--example-dir`),
loads its `configs/trainer_base.yaml`, injects per-trainer data from the
shared `_metadata/` bundle, and spawns processes with `--config-json` payloads.

Experiment-YAML shape:

```yaml
experiments:
  - name: my_experiment
    metadata:
      dir: examples/_metadata             # default: <example>/metadata
    trainer:
      num_trainers: 10
      start_id: 1
      dataset: {name: cifar10, dirichlet_alpha: 0.1}
      availability: {mode: syn_0}
    aggregator:
      config_template: aggregator/config.json
      selector: feddance
      agg_goal: 5
    execution:
      num_gpus: 1
      sleep_between_spawns: 0.5
      monitoring: {enabled: false}
```

## Shared metadata (`_metadata/`)

Per-trainer data lives in one shared hierarchy so multiple examples can reuse
the same device population, traces, and dataset splits:

```
_metadata/
├── trainer_registry.yaml                  # n=300 device population
├── availability_traces/
│   ├── mobiperf_traces.yaml               # per-device 2-state / 3-state
│   └── synthetic_traces.yaml              # syn_0 (uniform) + syn_20 / syn_50 (per-trainer)
└── dataset_splits/
    └── <dataset>_alpha<a>_n<N>.yaml       # per-trainer index lists
```

The hierarchy axes:

| Layer | File | Varies with |
| :--- | :--- | :--- |
| Device population | `trainer_registry.yaml` | N (future n=1000 → `trainer_registry_n1000.yaml`) |
| Availability | `availability_traces/*.yaml` | population, per-device or per-trainer realizations |
| Data partition | `dataset_splits/<ds>_alpha<a>_n<N>.yaml` | dataset × N × Dirichlet α |

Adding a new example: create `aggregator/pytorch/main.py` and `trainer/pytorch/main.py`
that call `from flame.launch.cli import load_config_from_argv`, then add
`configs/trainer_base.yaml` and `experiments/configs/<name>.yaml`. The launcher
handles the rest.

Migrating an **existing** example off per-trainer JSON / shell scripts onto this
launcher: follow [`MIGRATING_TO_LAUNCHER.md`](MIGRATING_TO_LAUNCHER.md) (trainer +
aggregator changes, adding dataset data to `_metadata/`, baseline catalog, and
legacy decommission). `async_cifar10/` is the reference implementation.

For provenance on the n=300 metadata, see
[`_metadata/migration_plan_async_cifar10.yaml`](_metadata/migration_plan_async_cifar10.yaml).
To re-run / extend that migration, see [`lib/python/scripts/migrate_async_cifar10.py`](../scripts/migrate_async_cifar10.py).

## Basic Federated Learning

### [mnist/](mnist/)
Basic horizontal federated learning with MNIST dataset. Good starting point for understanding FL fundamentals.

### [cifar10/](cifar10/)
Standard horizontal FL on CIFAR-10 with image classification using CNNs.

### [medmnist/](medmnist/)
Medical image classification with PathMNIST dataset. Demonstrates adaptive aggregation with data heterogeneity.

### [adult/](adult/)
Federated learning on the Adult Census Income dataset for tabular data classification.

## Asynchronous Federated Learning

### [async_cifar10/](async_cifar10/) ⭐
Large-scale asynchronous FL on CIFAR-10 with 300 trainers. Demonstrates:
- Client selection strategies (Oort, FedBuff)
- Availability tracking (oracular, unaware modes)
- Non-IID data distribution (Dirichlet sampling)
- Real-world availability traces (MobiPerf, synthetic)

### [async_mnist/](async_mnist/)
Asynchronous federated learning on MNIST dataset.

### [async_google_speech/](async_google_speech/)
Asynchronous FL for speech recognition using Google Speech Commands dataset.

### [async_hier_cifar10/](async_hier_cifar10/)
Asynchronous hierarchical federated learning with CIFAR-10.

### [async_hier_mnist/](async_hier_mnist/)
Asynchronous hierarchical FL topology with MNIST dataset.

## Hierarchical Federated Learning

### [hier_mnist/](hier_mnist/)
Basic hierarchical FL with two-tier topology (top aggregator, middle aggregators, trainers).

### [hier_cifar10/](hier_cifar10/)
Hierarchical federated learning on CIFAR-10 dataset.

### [coord_hier_syncfl_mnist/](coord_hier_syncfl_mnist/)
Synchronous hierarchical FL with coordinator for MNIST.

### [coord_hier_asyncfl_mnist/](coord_hier_asyncfl_mnist/)
Asynchronous hierarchical FL with coordinator for MNIST.

### [coord_3_hier_syncfl_mnist/](coord_3_hier_syncfl_mnist/)
Three-tier synchronous hierarchical FL with coordinator.

### [eager_hier_mnist/](eager_hier_mnist/)
Eager aggregation variant of hierarchical FL on MNIST.

## Advanced Algorithms

### [cifar10_scaffold/](cifar10_scaffold/)
**SCAFFOLD** algorithm on CIFAR-10. Uses control variates to correct for client drift in non-IID settings.  
Paper: [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)

### [medmnist_oort/](medmnist_oort/)
**Oort** client selection on MedMNIST. Guided participant selection based on statistical utility and system utility.  
Paper: [Oort: Efficient Federated Learning via Guided Participant Selection (OSDI'21)](https://www.usenix.org/conference/osdi21/presentation/lai)

### [medmnist_fedprox/](medmnist_fedprox/)
**FedProx** on MedMNIST. Addresses system heterogeneity with proximal term (mu parameter).  
Paper: [Federated Optimization in Heterogeneous Networks (MLSys'20)](https://arxiv.org/abs/1812.06127)

### [medmnist_feddyn/](medmnist_feddyn/)
**FedDyn** on MedMNIST. Dynamic regularization approach for improved convergence.  
Paper: [Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263)

### [compas_fedgft/](compas_fedgft/)
**FedGFT** on COMPAS dataset. Group fairness-aware federated learning for bias mitigation.  
Paper: [Mitigating Group Bias in Federated Learning (MobiSys'22)](https://arxiv.org/abs/2305.09931)

### [fedbalancer_mnist/](fedbalancer_mnist/)
**FedBalancer** on MNIST. Data and pace control for efficient FL on heterogeneous clients.  
Paper: [FedBalancer (MobiSys'22)](https://arxiv.org/abs/2201.01601)

## Specialized Examples

### [fwdllm/](fwdllm/)
Federated learning for Large Language Models (LLMs) with forward-only training for text classification tasks.

### [hybrid/](hybrid/)
Hybrid federated learning combining vertical and horizontal FL approaches.

### [dp_mnist/](dp_mnist/)
Differential privacy (DP) with federated learning on MNIST for privacy-preserving training.

### [dist_mnist/](dist_mnist/)
Distributed training on MNIST (non-federated, centralized data distribution).

### [fedscale/](fedscale/)
Integration examples with FedScale framework:
- `femnist/` - Federated EMNIST 
- `coord_hier_syncfl_femnist/` - Hierarchical coordinated FL on FEMNIST

## Utilities

### [device_traces/](device_traces/)
Device availability traces for simulating realistic client participation patterns.

### [expt_result_plots/](expt_result_plots/)
Plotting utilities and analysis notebooks for experiment results.

### Notebooks
- `analyze_motiv_select.ipynb` - Client selection analysis
- `analyze_run_agg_logs.ipynb` - Aggregator log analysis
- `create_cifar10_google_speech_trainer_configs.ipynb` - Config generation

### Scripts
- `run.py` - Generic runner for multiple examples

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch (version varies by example)
- MQTT broker (mosquitto) for most examples
- Conda/virtualenv recommended

### Installation

1. **Install Flame library**:
   ```bash
   cd /path/to/flame/lib/python
   pip install -e .
   ```

2. **Setup MQTT broker** (required for most examples):
   ```bash
   sudo apt-get install mosquitto mosquitto-clients
   sudo systemctl start mosquitto
   ```

3. **Navigate to example**:
   ```bash
   cd examples/<example_name>
   ```

4. **Follow example-specific README** for detailed setup and execution.

### Quick Start Pattern

Most examples follow this pattern:
```bash
# Activate environment
conda activate flame

# Run using the provided script
python run.py  # or bash example.sh
```

## Example Categories Summary

| Category | Examples | Key Features |
|----------|----------|--------------|
| **Basic FL** | mnist, cifar10, medmnist, adult | Standard horizontal FL |
| **Async FL** | async_cifar10, async_mnist, async_google_speech | Non-blocking updates, client selection |
| **Hierarchical** | hier_mnist, hier_cifar10, coord_* | Multi-tier aggregation |
| **Algorithms** | scaffold, fedprox, feddyn, oort, fedbalancer, fedgft | Advanced optimization & selection |
| **Specialized** | fwdllm, hybrid, dp_mnist | LLMs, privacy, hybrid approaches |

## Documentation

- [Flame Documentation](https://github.com/cisco-open/flame/tree/main/docs)
- [Flame Basics](../../docs/flame-basics.md)
- [SDK Documentation](../../docs/sdk/)

## Support

For issues: https://github.com/cisco-open/flame/issues
