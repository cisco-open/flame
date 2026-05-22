# Async CIFAR-10 Federated Learning

Asynchronous federated learning on CIFAR-10 with 300 trainers, demonstrating client selection strategies (Oort, FedBuff), availability tracking, and non-IID data distributions.

## Quick Start

```bash
# 1. Set up the conda env (Python 3.11 + flame + examples extras).
#    Run from the repo root:
bash scripts/setup_env.sh my_flame_env

# 2. Check MQTT broker is running.
systemctl is-active mosquitto || pgrep mosquitto

# 3. Run an experiment via the YAML launcher.
conda activate my_flame_env
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn20_smoke.yaml
```

Logs land in `experiments/run_<timestamp>_<name>/` (aggregator + trainers,
plus the merged `aggregator_config.json` for reproducibility).

## What This Example Does

Simulates a federated learning environment with:
- **1 Aggregator**: Central server coordinating training
- **300 Trainers**: Clients with local CIFAR-10 data partitions
- **MQTT Communication**: Message broker for parameter exchange
- **Smart Selection**: Oort strategy picks best trainers based on utility
- **Availability Traces**: Simulates real-world trainer unavailability (syn0/20/50, MobiPerf)
- **Non-IID Data**: Dirichlet sampling (alpha=0.1) creates heterogeneous data distributions

The experiment runs until reaching 70% test accuracy, testing 4 availability scenarios automatically.

## Directory Structure

```
async_cifar10/
├── aggregator/               # Central server entrypoints (one per stack)
│   ├── pytorch/main_asyncfl_agg.py    # asyncfl stack (felix, fedbuff)
│   ├── pytorch/main_oort_sync_agg.py  # sync oort stack (oort, refl)
│   └── pytorch/main_fedavg_agg.py     # base syncfl stack (fedavg)
├── trainer/                  # Client trainers
│   └── pytorch/main.py
├── configs/
│   └── trainer_base.yaml    # Per-example trainer template
├── expt_scripts_2026/        # Experiment YAMLs (launcher inputs)
│   ├── <baseline>_n10_*.yaml      # 10-trainer smoke tests
│   └── felix_n300_*.yaml          # full-scale runs
└── data/                    # CIFAR-10 (auto-downloaded by trainers)

# Shared across examples (sibling at examples/_metadata):
examples/_metadata/
├── trainer_registry.yaml             # n=300 device population
├── availability_traces/              # mobiperf + synthetic
├── dataset_splits/                   # per-(dataset, alpha, N) splits
├── baselines.yaml                    # felix / fedbuff / fedavg / oort / refl
└── aggregator_base.json              # generic aggregator boilerplate
```

## Manual Setup

If `scripts/setup_env.sh` doesn't work for your system:

```bash
conda create -n my_flame_env python=3.11 -y
conda activate my_flame_env
pip install -e lib/python[examples,dev]

# Check MQTT broker
systemctl is-active mosquitto 2>/dev/null || pgrep mosquitto
```

Required deps (installed automatically by the extras above):
- core (flame): paho-mqtt, pydantic, mlflow, grpcio, protobuf, PyYAML, ...
- examples: torch, torchvision, sortedcontainers, wandb
- wandb, sortedcontainers
- All Flame library dependencies

## Baselines & smoke tests

A *baseline* (selector + optimizer + aggregator stack + tracking) is defined
once in `examples/_metadata/baselines.yaml`. An experiment YAML names a baseline
and overrides only what it needs. The launcher resolves the aggregator stack
from the baseline and **refuses mismatched selector/stack combinations** (e.g.
an async selector on the sync stack), so you can't accidentally run the wrong
pairing.

Ready-to-run 10-trainer smoke tests live in `expt_scripts_2026/`. Run any one
with (from the repo root):

```bash
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/<smoke>.yaml
```

| Baseline | Selector / Optimizer | Stack       | Smoke YAML |
|----------|----------------------|-------------|------------|
| felix    | async_oort / fedbuff | asyncfl     | `felix_n10_alpha100_syn20_smoke.yaml` |
| fedbuff  | fedbuff / fedbuff    | asyncfl     | `fedbuff_n10_alpha100_smoke.yaml` |
| fedavg   | random / fedavg      | base syncfl | `fedavg_n10_alpha100_smoke.yaml` |
| oort     | oort / fedavg        | sync oort   | `oort_n10_alpha100_syn0_smoke.yaml` |
| refl     | refl_oort / refl     | sync oort   | `refl_n10_alpha100_syn0_smoke.yaml` |

Scale up by copying a smoke YAML and raising `num_trainers` (see
`felix_n300_alpha100_syn20.yaml`). Logs land in `experiments/run_<ts>_<name>/`.

> **FedDance** is not yet runnable here — see [`FEDDANCE_TODO.md`](FEDDANCE_TODO.md).
>
> The shell scripts under `*_expts/scripts/` are **deprecated** (see the
> `DEPRECATED.md` in each directory); use the launcher above.

## Key Configuration Parameters

Override per experiment under `aggregator.config_overrides` / `trainer.*` in the
experiment YAML; baseline-wide values live in `baselines.yaml`.

- `hyperparameters.aggGoal` — updates aggregated before a global step
- `hyperparameters.trackTrainerAvail.{type,trace}` — `ORACULAR`+`syn_0/20/50`
  (or `mobiperf_*`) vs disabled
- `selector.kwargs.aggr_num` — trainers selected per round
- `trainer.dataset.dirichlet_alpha` — `0.1` (highly non-IID) … `100` (near IID)

## Monitoring

```bash
# Watch aggregator
tail -f eurosys26_expts/agg_logs/agg_*.log | grep "test accuracy"

# Watch trainers
tail -f eurosys26_expts/trainer_logs/log_trainer_*.log

# Weights & Biases (if configured)
# Visit: https://wandb.ai/your-username/ft-distr-ml
```

**Expected progress**: ~30% (round 50) → ~60% (round 200) → 70% (stop)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MQTT connection failed | Check if mosquitto is running: `systemctl is-active mosquitto` or `pgrep mosquitto`. Contact admin if not running. |
| CUDA out of memory | Edit `exec_*_trainers.sh`, increase `NUM_AVAIL_GPUS` or reduce `batchSize` |
| Import error: `flame` | `cd ../../ && pip install -e .` |
| Processes hang | `pkill -f flame.launch.run_experiment; pkill -f aggregator/pytorch; pkill -f trainer/pytorch` |

## Next steps

- First task for a new contributor: [`ava_first_task.md`](ava_first_task.md)
  (split the trainer delay into compute + RTT, then make RTT time-varying).

## References

- [Oort Paper (OSDI'21)](https://www.usenix.org/conference/osdi21/presentation/lai)
- [FedBuff Paper](https://arxiv.org/abs/2106.06639)
- [Flame Documentation](https://github.com/cisco-open/flame/tree/main/docs)
