# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared experiment launcher.

Example-agnostic tooling for running federated experiments described as
YAML. Each example (cifar10, async_cifar10, feddance_cifar10, ...) hands
this launcher its trainer/aggregator entry points + per-example trainer
base template; metadata (trainer registry, traces, dataset splits) is
shared from examples/_metadata/ by default.
"""

from flame.launch.baselines import (
    deep_merge,
    format_provenance,
    load_baselines,
    merge_with_provenance,
)
from flame.launch.experiment_config import (
    ExperimentBatch,
    ExperimentConfig,
    load_experiment_config,
)
from flame.launch.spawner import ConfigGenerator, MetadataLoader, TrainerSpawner
from flame.launch.aggregator_spawner import AggregatorSpawner
from flame.launch.snapshot import ExperimentSnapshot
from flame.launch.execution_config_generator import (
    create_execution_config,
    save_execution_config,
)

# resource_monitor needs psutil; lazy-import to keep flame.launch usable without it.
try:
    from flame.launch.resource_monitor import create_monitor_from_config
except ImportError:
    create_monitor_from_config = None  # type: ignore

__all__ = [
    "ExperimentBatch",
    "ExperimentConfig",
    "load_experiment_config",
    "ConfigGenerator",
    "MetadataLoader",
    "TrainerSpawner",
    "AggregatorSpawner",
    "ExperimentSnapshot",
    "create_execution_config",
    "save_execution_config",
    "create_monitor_from_config",
    "load_baselines",
    "deep_merge",
    "merge_with_provenance",
    "format_provenance",
]
