#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Emit the n=50 Felix streaming-misprioritization experiment YAMLs.

PHASE 1 (now): UNIFORM streaming, parallelized across 4 nodes. Each node runs ONE
baseline B and its per-baseline online oracle B_oracle (same selector as B, plus
aggregator-side true-utility injection -- see aggregator/pytorch/oracle_utility.py),
sequentially. B_oracle is "B fed true utilities at each selection step", so its
time-to-acc / trajectory is the per-baseline oracular counterfactual of B.
Staggered streaming is PHASE 2 (set STAGGER_CONDITIONS = [False, True]).

Outputs (expt_scripts_2026/):
  n50_alpha0.1_syn0_stream_unif_sim.yaml          all 8 uniform arms (single node / smoke)
  n50_alpha0.1_syn0_stream_unif_node{1..4}.yaml   per-node {baseline + baseline_oracle}

Node map (HW only affects wall speed in sim):
  node1 (128c/500GB): felix    + felix_oracle
  node2 (128c/500GB): feddance + feddance_oracle
  node3 (96c/250GB):  oort     + oort_oracle
  node4 (96c/250GB):  refl     + refl_oracle

    NUM_GPUS=<per-node gpu count> python gen_n50_experiment.py
"""
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "expt_scripts_2026"))

N = 50
ALPHA = 0.1
HORIZON_S = 10800          # uniform streaming: data fully unlocked by this sim-sec
ONSET_MAX_S = 5400         # staggered (phase 2): max per-client start delay
RATE_JITTER = 0.5          # staggered (phase 2): +/- span jitter
ROUNDS_CAP = 20000         # hard safety cap
MAX_RUNTIME_S = 12600      # virtual-time safety cap
EVAL_EVERY = 10
TARGET_ACC = 0.60
STABLE_EVALS = 20
SAMPLE_SIZE = 256
NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))   # set per node; 0/CPU still runs

STAGGER_CONDITIONS = [False]   # PHASE 1: uniform only. Phase 2: [False, True].

# baseline -> (selector_sort, tracking_mode, selector_kwargs)
ARMS = {
    "felix":    ("async_oort", "client_notify", {"c": 10, "aggGoal": 10}),
    "oort":     ("oort",       "oracular",      {"aggr_num": 10}),
    "refl":     ("refl_oort",  "oracular",      {"aggr_num": 10}),
    "feddance": ("feddance",   "client_notify", {"aggr_num": 10}),
}

NODE_BASELINE = {1: "felix", 2: "feddance", 3: "oort", 4: "refl"}

EXEC = {
    "num_gpus": NUM_GPUS,
    "sleep_between_spawns": 1.0,
    "aggregator_warmup_time": 60,
    "monitoring": {
        "enabled": True, "check_interval_seconds": 30,
        "ram_warning_percent": 80.0, "ram_critical_percent": 90.0,
        "gpu_warning_percent": 80.0, "gpu_critical_percent": 90.0,
    },
}


def data_streaming(staggered):
    ds = {"enabled": "True", "full_data_available_after_s": HORIZON_S}
    if staggered:
        ds["stagger"] = {
            "enabled": "True", "onset_max_s": ONSET_MAX_S,
            "rate_jitter": RATE_JITTER, "min_visible": 1,
        }
    return ds


def make_arm(baseline, staggered, oracle=False, suffix=""):
    sort, tracking, sel_kwargs = ARMS[baseline]
    cond = "stag" if staggered else "unif"
    ocl = "_oracle" if oracle else ""
    name = f"{baseline}{ocl}_n50_alpha0.1_syn0_stream_{cond}_sim{suffix}"
    agg_hp = {
        "batchSize": 10, "learningRate": 0.01,
        "rounds": ROUNDS_CAP, "max_runtime_s": MAX_RUNTIME_S,
        "aggGoal": 10, "evalEveryNRounds": EVAL_EVERY,
        "targetAccuracy": TARGET_ACC, "stableEvalsAboveTarget": STABLE_EVALS,
        "min_trainers_to_start": N - 2, "min_trainers_join_timeout_s": 600,
        "checkpoint": {"enabled": "True", "every_n_rounds": EVAL_EVERY},
    }
    if oracle:
        # aggregator-side injector needs the streaming horizon + split selectors;
        # inject true local accuracy too so FedDance's A_m is oracular.
        agg_hp["data_streaming"] = data_streaming(staggered)
        agg_hp["oracle_utility_injection"] = {
            "enabled": "True", "alpha": ALPHA, "num_trainers": N,
            "sample_size": SAMPLE_SIZE,
            "inject_accuracy": "True" if baseline == "feddance" else "False",
        }
    return {
        "name": name,
        "description": f"{baseline}{' (online oracle)' if oracle else ''} | "
                       f"{'staggered' if staggered else 'uniform'} streaming | "
                       f"n=50 alpha=0.1 syn_0 (sim){suffix}.",
        "baseline": baseline,
        "trainer": {
            "num_trainers": N, "start_id": 1,
            "dataset": {"name": "cifar10", "dirichlet_alpha": ALPHA},
            "availability": {"mode": "syn_0"},
            "enable_training_delays": True,
            "time_mode": "simulated",
            "hyperparameters": {"batchSize": 10, "learningRate": 0.01},
            "config_overrides": {"hyperparameters": {
                "data_streaming": data_streaming(staggered),
                "util_counterfactual": {
                    "enabled": "True", "every_n_rounds": EVAL_EVERY,
                    "sample_size": SAMPLE_SIZE},
            }},
        },
        "aggregator": {
            "config_template": "../_metadata/aggregator_base.json",
            "selector": sort,
            "tracking_mode": tracking,
            "agg_goal": 10,
            "log_to_wandb": False,
            "config_overrides": {
                "job": {"id": name},
                "hyperparameters": agg_hp,
                "selector": {"kwargs": sel_kwargs},
            },
            "execution": EXEC,
        },
        "execution": EXEC,
    }


def _dump(experiments, path, note):
    header = (
        "# n=50 Felix streaming-misprioritization experiment (sim, UNIFORM).\n"
        f"# GENERATED by scripts/gen_n50_experiment.py -- {note}\n"
        "# Each baseline B is paired with B_oracle (same selector + aggregator-side\n"
        "# true-utility injection). Edit the generator, not this file.\n"
        "# Run: python -m flame.launch.run_experiment <this file>\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump({"experiments": experiments}, fh,
                       sort_keys=False, default_flow_style=False, width=100)
    print(f"wrote {os.path.relpath(path)} ({len(experiments)} arms)")


def main():
    # 1) combined: all 8 uniform arms (4 baseline + 4 oracle), single node / smoke.
    combined = []
    for st in STAGGER_CONDITIONS:
        for b in ARMS:
            combined.append(make_arm(b, st, oracle=False))
            combined.append(make_arm(b, st, oracle=True))
    _dump(combined, os.path.join(_OUTDIR, "n50_alpha0.1_syn0_stream_unif_sim.yaml"),
          "all uniform arms (single node / smoke)")

    # 2) per-node files: {baseline_i + baseline_i_oracle}.
    for node, baseline in NODE_BASELINE.items():
        exps = [make_arm(baseline, False, oracle=False),
                make_arm(baseline, False, oracle=True, suffix=f"_node{node}")]
        _dump(exps, os.path.join(
            _OUTDIR, f"n50_alpha0.1_syn0_stream_unif_node{node}.yaml"),
            f"node{node}: {baseline} + {baseline}_oracle")


if __name__ == "__main__":
    main()
