# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""CIFAR-10 horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import argparse
import ast
import calendar
import gc
import hashlib
import logging
import os
import sys
import threading
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from flame.config import Config, TrainerAvailState
from flame.mode.horizontal.trainer import Trainer
from flame import telemetry
from flame.telemetry.events import (
    build_avail_change,
    build_trainer_round,
    build_util_disparity,
)
from torchvision.datasets import CIFAR10
from memory_profiler import MemoryProfiler

logger = logging.getLogger(__name__)


def _stagger_params(trainer_id, onset_max_s, base_span_s, rate_jitter):
    """Per-client streaming schedule (onset, span), deterministic in trainer_id.

    Used for staggered data streaming so different clients' data arrives in
    different sim-time windows. Mirrored EXACTLY in
    scripts/analysis/oracle_misselection.py:stagger_params -- if you change the
    derivation here, change it there too or the offline oracle will reconstruct
    the wrong visible prefixes.

        onset_s = onset_max_s * u1
        span_s  = base_span_s * (1 + rate_jitter * (2*u2 - 1))   (>= base_span/4)

    where u1, u2 in [0,1) come from disjoint 32-bit slices of
    sha256(f"{trainer_id}:stagger").
    """
    h = hashlib.sha256(f"{trainer_id}:stagger".encode()).hexdigest()
    u1 = int(h[0:8], 16) / 0xFFFFFFFF
    u2 = int(h[8:16], 16) / 0xFFFFFFFF
    onset_s = onset_max_s * u1
    span_s = base_span_s * (1.0 + rate_jitter * (2.0 * u2 - 1.0))
    return onset_s, max(base_span_s / 4.0, span_s)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """Forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PyTorchCifar10Trainer(Trainer):
    """PyTorch CIFAR-10 Trainer."""

    def __init__(self, config: Config, battery_threshold, time_mode="simulated") -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None
        # Oort requires its loss function to have 'reduction'
        # parameter
        self.loss_fn = torch.nn.CrossEntropyLoss

        self.device = None
        self.train_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.trainer_id = self.config.task_id

        self.lr_decay_enabled = getattr(self.config.hyperparameters, 'lr_decay_enabled', False)
        self.lr_decay_factor = getattr(self.config.hyperparameters, 'lr_decay_factor', 0.98)
        self.lr_decay_epoch = getattr(self.config.hyperparameters, 'lr_decay_epoch', 10)
        self.min_learning_rate = getattr(self.config.hyperparameters, 'min_learning_rate', 1e-4)

        self.criterion = None

        self.task_to_perform = "train"

        self.use_oort_loss_fn = self.config.hyperparameters.use_oort_loss_fn
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list
        self.trainer_start_ts = time.time()

        if "enabled" in self.config.hyperparameters.heartbeats:
            self.heartbeats_enabled = self.config.hyperparameters.heartbeats["enabled"]
        else:
            self.heartbeats_enabled = False

        if "frequency_s" in self.config.hyperparameters.heartbeats:
            self.heartbeats_second_freq = self.config.hyperparameters.heartbeats[
                "frequency_s"
            ]
        else:
            self.heartbeats_second_freq = 99999

        if self.heartbeats_enabled is True:
            self.timestamp_next_heartbeat_s = (
                self.trainer_start_ts + self.heartbeats_second_freq
            )
        else:
            self.timestamp_next_heartbeat_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
            )

        self.client_notify = self.config.hyperparameters.client_notify

        # Normalize to bool: may arrive as Python bool or string "True"/"False".
        _tde = self.config.hyperparameters.training_delay_enabled
        self.training_delay_enabled = (
            _tde if isinstance(_tde, bool) else str(_tde).strip().lower() == "true"
        )
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)

        # Sim-only post-compute completion leg (§3i): the real per-trainer cycle has
        # ~1.6s after compute (buffer-residence queue_wait + re-dispatch latency)
        # that the sim sct omitted -> sim cycle short -> advance under-charges.
        # Added to sim_round_duration so sct = send_ts + max(gpu, D) + leg. Staleness
        # (= cycle/advance) is invariant to it; only advance/throughput are corrected.
        _leg = getattr(self.config.hyperparameters, "sim_completion_leg_s", 0.0)
        self.sim_completion_leg_s = float(_leg) if _leg is not None else 0.0

        self.time_mode = str(time_mode)
        self.simulated = self.time_mode == "simulated"
        self._sim_send_ts = None  # set by aggregator stamp on each task (sim mode)

        # Use the battery_threshold to determine the
        # avl_events_3_state config. Default to 50 if not provided
        self.event_battery_threshold = battery_threshold
        logger.info(
            f"Trainer id {self.trainer_id} has battery threshold set to {self.event_battery_threshold}"
        )

        def parse_trace(value):
            if isinstance(value, list):
                return value  # Already parsed (from JSON config)
            else:
                # String format (from file config) - validate it's a safe list literal
                try:
                    parsed = ast.literal_eval(value)
                    if not isinstance(parsed, list):
                        raise ValueError(f"Expected list, got {type(parsed)}")
                    return parsed
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid trace format: {e}")

        if self.event_battery_threshold == 50:
            self.avl_events_3_state = parse_trace(
                self.config.hyperparameters.avl_events_mobiperf_3st_50
            )
        elif self.event_battery_threshold == 75:
            self.avl_events_3_state = parse_trace(
                self.config.hyperparameters.avl_events_mobiperf_3st_75
            )

        self.avl_events_mobiperf_2st = parse_trace(
            self.config.hyperparameters.avl_events_mobiperf_2st
        )

        # Storing synthetic avail traces
        self.avl_events_syn_0 = parse_trace(
            self.config.hyperparameters.avl_events_syn_0
        )

        self.avl_events_syn_20 = parse_trace(
            self.config.hyperparameters.avl_events_syn_20
        )

        self.avl_events_syn_50 = parse_trace(
            self.config.hyperparameters.avl_events_syn_50
        )

        if self.client_notify["trace"] == "mobiperf_3st":
            self.state_avl_event_ts = self.avl_events_3_state
            logger.info(
                f"Set avl_events_3_state for trainer id {self.trainer_id} using battery threshold {self.event_battery_threshold}"
            )
        elif self.client_notify["trace"] == "mobiperf_2st":
            self.state_avl_event_ts = self.avl_events_mobiperf_2st
            logger.info(
                f"Set avl_events_mobiperf_2st for trainer id {self.trainer_id}."
            )
        elif self.client_notify["trace"] == "syn_0":
            self.state_avl_event_ts = self.avl_events_syn_0
            logger.info(f"Set avl_events_syn_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_20":
            self.state_avl_event_ts = self.avl_events_syn_20
            logger.info(f"Set avl_events_syn_20 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_50":
            self.state_avl_event_ts = self.avl_events_syn_50
            logger.info(f"Set avl_events_syn_50 for trainer id {self.trainer_id}.")
        else:
            logger.info(
                f"No avl_events set for trainer id {self.trainer_id} since state not specified."
            )

        self.avl_state = TrainerAvailState.AVL_TRAIN

        # flag to decide whether the trainer upon unavailability will wait or exit
        self.wait_until_next_avl = self.config.hyperparameters.wait_until_next_avl

        ds_cfg = getattr(self.config.hyperparameters, "data_streaming", None) or {}
        self.data_streaming_enabled = str(ds_cfg.get("enabled", "False")) == "True"
        self.data_streaming_full_after_s = float(
            ds_cfg.get("full_data_available_after_s", 0)
        )
        # Optional staggered streaming: each client gets its OWN onset (start
        # delay) and span (time to fill), derived deterministically from its
        # trainer_id so the offline oracle can reconstruct the exact schedule.
        # Uniform streaming is the special case onset=0, span=full_after_s.
        stg = ds_cfg.get("stagger", {}) or {}
        self.stream_stagger_enabled = str(stg.get("enabled", "False")) == "True"
        self.stream_onset_max_s = float(stg.get("onset_max_s", 0.0))
        self.stream_rate_jitter = float(stg.get("rate_jitter", 0.0))
        self.stream_min_visible = int(stg.get("min_visible", 1) or 1)
        # Defaults (overwritten per-client in load_data once trainer_id-seeded):
        self._stream_onset_s = 0.0
        self._stream_span_s = self.data_streaming_full_after_s
        logger.info(
            f"Trainer {self.trainer_id}: data streaming "
            f"{'ENABLED' if self.data_streaming_enabled else 'DISABLED'} "
            f"(full_data_available_after_s={self.data_streaming_full_after_s}, "
            f"stagger={'ON' if self.stream_stagger_enabled else 'off'})"
        )

        uc_cfg = getattr(self.config.hyperparameters, "util_counterfactual", None) or {}
        self.util_cf_enabled = str(uc_cfg.get("enabled", "False")) == "True"
        self.util_cf_every_n = int(uc_cfg.get("every_n_rounds", 1) or 1)
        _ss = uc_cfg.get("sample_size", 256)
        self.util_cf_sample_size = int(_ss) if _ss not in (None, "None", "") else None
        self._pool_tensor_cache = None  # lazily materialized full-pool tensors
        logger.info(
            f"Trainer {self.trainer_id}: util counterfactual "
            f"{'ENABLED' if self.util_cf_enabled else 'DISABLED'} "
            f"(every_n_rounds={self.util_cf_every_n}, sample_size={self.util_cf_sample_size})"
        )

        # Initialize memory profiler. Off by default: its per-round heap walks
        # (gc.collect + 3x gc.get_objects() with a per-object torch.is_tensor
        # check) dominate per-round wall time at high trainer-per-host
        # concurrency. Enable only when chasing a leak via the
        # `memory_profiling_enabled: "True"` hyperparameter.
        _mp = getattr(self.config.hyperparameters, "memory_profiling_enabled", False)
        self.memory_profiling_enabled = (
            _mp if isinstance(_mp, bool) else str(_mp).strip().lower() == "true"
        )
        self.memory_profiler = MemoryProfiler(
            trainer_id=str(self.trainer_id),
            log_interval_rounds=5,  # Detailed logs every 5 rounds
            enabled=self.memory_profiling_enabled,
        )
        logger.info(
            f"Trainer {self.trainer_id}: Memory profiler "
            f"{'ENABLED' if self.memory_profiling_enabled else 'DISABLED (default)'}"
        )

    def check_and_sleep(self):
        """Induce transient unavailability"""
        pass

    def _sim_now(self) -> float:
        """Wall-elapsed (real) or last-task sim_send_ts (simulated)."""
        if self.simulated:
            return float(self._sim_send_ts) if self._sim_send_ts is not None else 0.0
        return time.time() - self.trainer_start_ts

    def _refresh_avl_for_sim(self) -> None:
        """Advance availability state to current sim-time (sim mode only)."""
        if not self.simulated:
            return
        guard = 0
        while (
            self.state_avl_event_ts
            and self._sim_now() >= self.state_avl_event_ts[0][0]
            and guard < 100000
        ):
            self.check_and_update_state_avl()
            guard += 1

    def check_and_update_state_avl(self):
        if hasattr(self, "cm") and self.cm is not None:
            if len(self.state_avl_event_ts) > 0:
                # event timestamps are in sim-seconds since start; compare to
                # the current sim-time (no speedup_factor in either mode).
                sim_elapsed = self._sim_now()
                if sim_elapsed >= self.state_avl_event_ts[0][0]:
                    state_to_set = self.state_avl_event_ts.pop(0)[1]
                    old_status = self.avl_state.value
                    try:
                        self.avl_state = TrainerAvailState(state_to_set)
                    except ValueError:
                        logger.error(
                            f"Invalid status encountered: {state_to_set}. Retaining old status {old_status}."
                        )
                        return
                    new_status = self.avl_state.value
                    logger.info(
                        f"Changed the availability status of trainer {self.trainer_id} from {old_status} to {new_status}"
                    )
                    if telemetry.is_enabled():
                        ev, fields = build_avail_change(
                            round_num=int(getattr(self, "_round", 0)),
                            old_state=str(old_status),
                            new_state=str(new_status),
                        )
                        telemetry.emit(ev, **fields)
                    if self.client_notify["enabled"] == "True":
                        self._perform_channel_state_update(
                            tag="upload",
                            state=self.avl_state,
                            timestamp=str(time.time()),
                        )
            else:
                logger.debug(
                    f"No availability events pending for trainer {self.trainer_id}"
                )
        else:
            logger.info(
                f"Channel manager not set yet for trainer {self.trainer_id}. "
                f"Skipping avail status update. "
                f"Sleep for 20s before checking again."
            )
            time.sleep(20)

    def initialize(self) -> None:
        """Initialize role."""
        self.memory_profiler.log_component_memory("initialize", "BEFORE")

        # Honour single-thread pinning set by the spawner via OMP_NUM_THREADS=1.
        if os.environ.get("OMP_NUM_THREADS") == "1":
            torch.set_num_threads(1)
            logger.info(f"Trainer {self.trainer_id}: torch.set_num_threads(1) (cpu_pinning active)")

        # Report actual post-fork placement so pinning can be verified from logs.
        try:
            _cpu_cores = sorted(os.sched_getaffinity(0))
        except AttributeError:
            _cpu_cores = []
        _gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        logger.info(
            f"[PLACEMENT] trainer={self.trainer_id} "
            f"gpu={_gpu_env} cpu_cores={_cpu_cores}"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        
        # Log model memory usage
        model_info = self.memory_profiler.analyze_model_memory(self.model)
        logger.info(
            f"Task_id: {self.trainer_id} Model initialized: "
            f"{model_info['total_params']} params, "
            f"{model_info['param_memory_mb']:.1f} MB"
        )
        
        self.memory_profiler.log_component_memory("initialize", "AFTER")
        
        logger.debug(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}"
        )

    def load_data(self) -> None:
        """Load data."""
        self.memory_profiler.log_component_memory("load_data", "BEFORE")
        
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = CIFAR10(
            "/home/dgarg39/flame/lib/python/examples/async_cifar10/data",
            train=True,
            download=True,
            transform=transform_train,
        )

        # create indices into a list and convert to tensor
        indices = torch.tensor(self.trainer_indices_list)

        dataset = data_utils.Subset(dataset, indices)

        # GPU pre-load for small datasets (cuts CPU RAM). Full pool is retained;
        # the loader is (re)built from a prefix in _rebuild_stream_loader.
        dataset_size = len(indices)
        gpu_preload_threshold = 2000  # Adjust based on GPU memory availability

        if dataset_size <= gpu_preload_threshold and self.device is not None:
            logger.info(
                f"Trainer {self.trainer_id}: Pre-loading {dataset_size} samples to GPU "
                f"to reduce CPU RAM usage"
            )

            # Load all data to GPU at once
            temp_loader = torch.utils.data.DataLoader(
                dataset, batch_size=dataset_size, shuffle=False
            )

            all_data = []
            all_targets = []
            for data, target in temp_loader:
                all_data.append(data.to(self.device))
                all_targets.append(target.to(self.device))

            # Retain the full GPU pool; loader built below from a prefix
            self._stream_gpu = True
            self._stream_all_data = torch.cat(all_data)
            self._stream_all_targets = torch.cat(all_targets)
            self._stream_train_kwargs = {
                "batch_size": self.batch_size,
                "drop_last": False,  # keep incomplete batches in FL
                "shuffle": True,
                "num_workers": 0,  # data already on GPU
            }

            del temp_loader, all_data, all_targets, dataset
            logger.info(
                f"Trainer {self.trainer_id}: Successfully pre-loaded data to GPU"
            )
        else:
            # Standard loading for larger datasets; retain full Subset
            self._stream_gpu = False
            self._stream_full_dataset = dataset
            self._stream_train_kwargs = {
                "batch_size": self.batch_size,
                "drop_last": False,  # keep incomplete batches in FL
                "shuffle": True,
                "num_workers": 0,  # reduces CPU RAM from worker processes
                "pin_memory": True,  # faster CPU->GPU transfers
            }
            logger.info(
                f"Trainer {self.trainer_id}: Using standard loading "
                f"({dataset_size} samples exceeds GPU pre-load threshold)"
            )

        # Fixed shuffle of the pool (seeded by trainer_id) = data arrival
        # order; streaming reveals a growing prefix of it.
        self._stream_total = dataset_size
        seed = int(hashlib.sha256(str(self.trainer_id).encode()).hexdigest(), 16) % (2**31)
        self._stream_order = torch.randperm(
            self._stream_total, generator=torch.Generator().manual_seed(seed)
        )

        # Per-client streaming schedule (staggered onset + span). Mirrored EXACTLY
        # in scripts/analysis/oracle_misselection.py:stagger_params -- keep in sync.
        if self.stream_stagger_enabled and self.data_streaming_full_after_s > 0:
            self._stream_onset_s, self._stream_span_s = _stagger_params(
                self.trainer_id,
                onset_max_s=self.stream_onset_max_s,
                base_span_s=self.data_streaming_full_after_s,
                rate_jitter=self.stream_rate_jitter,
            )
            logger.info(
                f"Trainer {self.trainer_id}: staggered stream "
                f"onset={self._stream_onset_s:.0f}s span={self._stream_span_s:.0f}s"
            )

        # Build initial loader (full pool unless streaming is enabled)
        self._rebuild_stream_loader()
        gc.collect()

        # Log DataLoader memory info
        dataloader_info = self.memory_profiler.get_dataloader_memory(self.train_loader)
        logger.info(
            f"Task_id: {self.trainer_id} DataLoader created: "
            f"dataset_size={dataloader_info['dataset_size']}, "
            f"batch_size={dataloader_info['batch_size']}, "
            f"num_workers={dataloader_info['num_workers']}"
        )
        
        self.memory_profiler.log_component_memory("load_data", "AFTER")

        logger.debug(
            f"Task_id: {self.trainer_id} load_data completed at timestamp: "
            f"{time.time()}"
        )

    def _visible_sample_count(self) -> int:
        """Samples unlocked so far: linear in sim-time, full after the client's span.

        Uniform streaming: onset=0, span=full_after_s (one global horizon).
        Staggered streaming: per-client onset/span (set in load_data) so different
        clients' data arrives in different sim-time windows.
        """
        if not self.data_streaming_enabled or self.data_streaming_full_after_s <= 0:
            return self._stream_total
        # *_after_s and onset/span are in sim-seconds; _sim_now() is sim-time
        # (wall-clock in real mode, stamped task time in simulated mode).
        span = self._stream_span_s if self._stream_span_s > 0 else self.data_streaming_full_after_s
        frac = min(1.0, max(0.0, (self._sim_now() - self._stream_onset_s) / span))
        n = math.floor(frac * self._stream_total)
        # >= stream_min_visible so the loader is non-empty even before onset.
        floor_n = self.stream_min_visible if self.stream_stagger_enabled else 1
        return min(self._stream_total, max(floor_n, n))

    def _rebuild_stream_loader(self) -> None:
        """Rebuild train_loader over the currently-visible prefix of the pool."""
        n = self._visible_sample_count()
        full = n >= self._stream_total
        if self._stream_gpu:
            if full:
                subset = data_utils.TensorDataset(
                    self._stream_all_data, self._stream_all_targets
                )
            else:
                # align index with data device (order is built on CPU)
                pos = self._stream_order[:n].to(self._stream_all_data.device)
                subset = data_utils.TensorDataset(
                    self._stream_all_data[pos], self._stream_all_targets[pos]
                )
        else:
            pos = self._stream_order if full else self._stream_order[:n]
            subset = data_utils.Subset(self._stream_full_dataset, pos.tolist())
        self.train_loader = torch.utils.data.DataLoader(
            subset, **self._stream_train_kwargs
        )

    def _pool_tensors(self):
        """Return (data, targets) tensors for the full sample pool on device.

        Reuses the GPU-preloaded pool when available; otherwise materializes
        the CPU Subset once and caches it. Only used for opt-in counterfactual
        telemetry, so the one-time cost is acceptable.
        """
        if getattr(self, "_stream_gpu", False):
            return self._stream_all_data, self._stream_all_targets
        if self._pool_tensor_cache is not None:
            return self._pool_tensor_cache
        loader = torch.utils.data.DataLoader(
            self._stream_full_dataset, batch_size=512, shuffle=False
        )
        datas, targets = [], []
        for d, t in loader:
            datas.append(d)
            targets.append(t)
        data = torch.cat(datas)
        target = torch.cat(targets)
        self._pool_tensor_cache = (data, target)
        return self._pool_tensor_cache

    def _oort_utility(self, data, targets, norm_n, sample_size=None):
        """Oort statistical utility over a (sampled) set: N * sqrt(mean(loss^2)).

        Mirrors the trainer's Oort utility but computed with no_grad over an
        arbitrary index set, so the streamed-prefix and full-pool values are
        directly comparable. Returns (utility, n_used).
        """
        n = data.shape[0]
        if n == 0:
            return 0.0, 0
        if sample_size is not None and n > sample_size:
            sel = torch.randperm(n)[:sample_size]
            data = data[sel.to(data.device)]
            targets = targets[sel.to(targets.device)]
        criterion = self.loss_fn(reduction="none")
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            targets = targets.to(self.device)
            output = self.model(data)
            per_sample = criterion(output, targets)
            sumsq = torch.square(per_sample).sum().item()
        n_used = data.shape[0]
        utility = norm_n * math.sqrt(sumsq / n_used) if n_used > 0 else 0.0
        return utility, n_used

    def _emit_util_disparity(self, round_num, elapsed_s):
        """Compute and emit streamed-prefix vs full-pool utility (opt-in)."""
        if not (self.util_cf_enabled and telemetry.is_enabled()):
            return
        if self.util_cf_every_n > 1 and (round_num % self.util_cf_every_n != 0):
            return
        try:
            data, targets = self._pool_tensors()
            total = self._stream_total
            visible_n = self._visible_sample_count()
            order = self._stream_order.to(data.device)
            ss = self.util_cf_sample_size
            util_streamed, _ = self._oort_utility(
                data[order[:visible_n]], targets[order[:visible_n]],
                norm_n=visible_n, sample_size=ss,
            )
            util_full, n_used = self._oort_utility(
                data[order], targets[order], norm_n=total, sample_size=ss,
            )
            ev, fields = build_util_disparity(
                round_num=int(round_num),
                elapsed_s=elapsed_s,
                visible_samples=int(visible_n),
                total_samples=int(total),
                utility_streamed=util_streamed,
                utility_full=util_full,
                sample_size_used=n_used,
            )
            telemetry.emit(ev, **fields)
        except Exception as e:  # telemetry must never break training
            logger.debug(f"util disparity emit failed: {e}")

    def train(self) -> None:
        logger.info(f"Entered train method for {self.trainer_id}")
        # Per-phase timing: time from train() entry to the start of the GPU
        # compute loop (setup/avail/loader-rebuild overhead). Reported in
        # [TRAIN_CYCLE] + telemetry so the breakdown is first-class.
        _phase_train_entry = time.time()

        # Log memory before training round (no-op unless profiling enabled)
        self.memory_profiler.log_memory_before_round()

        # NOTE: we deliberately do NOT call torch.cuda.empty_cache()/gc.collect()
        # per round here. With many trainers co-located on one GPU, empty_cache
        # forces a CUDA sync and frees the caching allocator's blocks, so the
        # next round re-allocates from the driver (serialized across processes)
        # — it inflates per-round time instead of helping. The allocator reuses
        # freed blocks within a process on its own.

        if self.task_to_perform != "train":
            logger.info(f"Trainer {self.trainer_id} is not required to train")
            return
        # telemetry: measure time spent waiting on availability (vs. computing)
        _wait_time_s = 0.0
        # simulated mode: refresh availability from the trace at this task's
        # sim-time before deciding (sim-time advances only with new tasks).
        self._refresh_avl_for_sim()
        # don't enter the if condition if the three_state_avl switch is off
        # if we are checking for three_state_avl - check if the mechanism is to wait or exit
        if self.avl_state != TrainerAvailState.AVL_TRAIN:
            if self.simulated:
                # sim-time can't advance while we block, so a real-time wait
                # would hang. The aggregator selects available trainers; being
                # unavailable here means skip this task (it will re-select).
                logger.info(
                    f"Trainer id {self.trainer_id} not available to train "
                    f"(simulated, sim_t={self._sim_now()}); skipping task."
                )
                return
            if self.wait_until_next_avl == "True":
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Waiting for it to be available"
                )
                _wait_start = time.time()
                while self.avl_state != TrainerAvailState.AVL_TRAIN:
                    time.sleep(1)
                _wait_time_s = time.time() - _wait_start
            else:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Exiting training."
                )
                return

        logger.info(f"Trainer {self.trainer_id} available to train")

        # Refresh visible data for this selection (no-op if streaming off)
        if self.data_streaming_enabled:
            self._rebuild_stream_loader()
            logger.info(
                f"Trainer {self.trainer_id} streaming: "
                f"{self._visible_sample_count()}/{self._stream_total} samples visible"
            )

        """Train a model."""
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Apply learning rate decay if enabled (REFL uses this, Oort doesn't)
        current_lr = self.learning_rate
        if self.lr_decay_enabled and hasattr(self, '_round') and self._round > 1:
            num_decays = (self._round - 1) // self.lr_decay_epoch
            current_lr = max(
                self.learning_rate * (self.lr_decay_factor ** num_decays),
                self.min_learning_rate
            )
            logger.info(
                f"Trainer {self.trainer_id} Round {self._round}: LR decayed to {current_lr:.6f} "
                f"(base_lr={self.learning_rate}, num_decays={num_decays})"
            )
        else:
            logger.debug(f"Trainer {self.trainer_id}: Using base LR {current_lr}")
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr)

        # reset stat utility for OORT
        self.reset_stat_utility()

        num_batches = len(self.train_loader)
        dataset_size = len(self.train_loader.dataset)
        _D = self.training_delay_s if self.training_delay_enabled else 0.0
        if self.simulated:
            _expected_wallclock_hint = f"~GPU wall-clock only; virtual_advance=max(gpu,D={_D:.1f}s)"
        else:
            _expected_wallclock_hint = f"~max(gpu,D={_D:.1f}s) wall-clock; sleep=max(0,D-gpu)"
        logger.info(
            f"[TRAIN_START] Trainer {self.trainer_id} starting training with "
            f"model_version={self._round}, dataset_size={dataset_size}, "
            f"num_batches={num_batches}, batch_size={self.batch_size}, epochs={self.epochs}, "
            f"time_mode={self.time_mode}, expected_cycle_time={_expected_wallclock_hint}"
        )
        _cycle_start = time.time()

        total_batches_processed = 0
        final_loss = None
        self._grad_norm_epoch1 = None
        # Reset per-round local training accuracy (FedDance's A_m reads this via
        # MessageType.LOCAL_ACCURACY; harmless for other selectors). Also
        # initializes the accumulators, so no init_oort_variables dependency.
        self.reset_local_accuracy()
        _gpu_start = time.time()
        # Setup/avail/loader-rebuild overhead before the compute loop.
        _pre_train_s = _gpu_start - _phase_train_entry
        for epoch in range(1, self.epochs + 1):
            epoch_batches, epoch_loss = self._train_epoch(epoch)
            total_batches_processed += epoch_batches
            if epoch_loss is not None:
                final_loss = epoch_loss
        # real GPU/compute time for this round, excluding any simulated delay
        _real_gpu_time_s = time.time() - _gpu_start
        # Post-compute overhead (cleanup, delta-l2, telemetry) starts here.
        _phase_post_start = time.time()

        # Log training completion summary
        loss_str = f"{final_loss:.6f}" if final_loss is not None else "N/A"
        logger.info(
            f"[TRAIN_COMPLETE] Trainer {self.trainer_id} completed training with "
            f"model_version={self._round}, dataset_size={dataset_size}, "
            f"total_batches_processed={total_batches_processed}, final_loss={loss_str}"
        )

        # save dataset size so that the info can be shared with
        # aggregator
        self.dataset_size = len(self.train_loader.dataset)
        
        # Drop grads (cheap, frees their memory for reuse within this process).
        # We intentionally skip empty_cache()/gc.collect() here — see the note
        # at the top of train(): they hurt under co-located concurrency.
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Log memory after training round (no-op unless profiling enabled)
        self.memory_profiler.log_memory_after_round()

        _modeled_delay_s = self.training_delay_s if self.training_delay_enabled else 0.0
        _remaining_time = max(0.0, _modeled_delay_s - _real_gpu_time_s)
        _overran = self.training_delay_enabled and _real_gpu_time_s > _modeled_delay_s
        self._training_budget_s = _modeled_delay_s

        if _overran:
            logger.warning(
                f"[TIMING_OVERRUN] Trainer {self.trainer_id} round={self._round} "
                f"exceeded budget: gpu={_real_gpu_time_s:.2f}s > budget={_modeled_delay_s:.2f}s "
                f"(excess={_real_gpu_time_s - _modeled_delay_s:.2f}s). "
                f"Reduce trainers-per-GPU or add GPUs."
            )

        # max(gpu, D): no contention → D; overrun → gpu > D (OORT sees trainer as slow).
        sim_round_duration = _real_gpu_time_s + _remaining_time  # = max(gpu, D)

        self._sim_round_duration = sim_round_duration

        # §3i: the completion timestamp (sct = when the update COMMITS) = send_ts +
        # compute + post-compute leg. The leg (buffer-residence + re-dispatch latency)
        # is added ONLY here, NOT to _sim_round_duration — so trainer_speed_s, OORT
        # utility, the gate predictor and the P3/T2 controls all keep pure compute,
        # and only the virtual clock (which advances to sct) sees the real cycle time.
        _leg = self.sim_completion_leg_s if self.simulated else 0.0
        self._sim_completion_ts = (
            (self._sim_send_ts if self._sim_send_ts is not None else self._sim_now())
            + sim_round_duration
            + _leg
        )

        # ||trained - received global||: update magnitude this round. At this
        # point self.weights still holds the received global (the later
        # _send_weights tasklet runs _update_weights); the model holds the
        # trained weights. Float params only (skip int buffers). Non-fatal.
        # Telemetry-only: skip entirely when telemetry is off, and accumulate
        # the squared-diff on-device so we sync once (not once per parameter).
        delta_weight_l2 = None
        if telemetry.is_enabled():
            try:
                ref = getattr(self, "weights", None)
                if ref is not None:
                    _sq = None
                    for k, v in self.model.state_dict().items():
                        if k in ref and torch.is_floating_point(v):
                            d = v.detach().float() - ref[k].detach().float().to(v.device)
                            s = torch.sum(d * d)
                            _sq = s if _sq is None else _sq + s
                    if _sq is not None:
                        delta_weight_l2 = math.sqrt(float(_sq.item()))
            except Exception as e:
                logger.debug(f"delta_weight_l2 compute failed: {e}")

        # Post-compute overhead so far (cleanup + delta-l2), before the modeled
        # sleep. Together with _pre_train_s and _real_gpu_time_s this is the
        # full trainer-side breakdown of where a round's wall time goes.
        _post_train_s = time.time() - _phase_post_start

        if telemetry.is_enabled():
            visible = (
                self._visible_sample_count()
                if self.data_streaming_enabled
                else self._stream_total
            )
            ev, fields = build_trainer_round(
                round_num=int(getattr(self, "_round", 0)),
                real_gpu_time_s=_real_gpu_time_s,
                sim_round_duration_s=sim_round_duration,
                wait_time_s=_wait_time_s,
                avail_state=self.avl_state.value,
                visible_samples=int(visible),
                total_samples=int(self._stream_total),
                dataset_size=int(dataset_size),
                stat_utility=float(self._stat_utility)
                if isinstance(self._stat_utility, (int, float))
                else float(getattr(self._stat_utility, "item", lambda: 0.0)()),
                final_loss=final_loss,
                delta_weight_l2=delta_weight_l2,
                extra={
                    "sim_completion_ts": self._sim_completion_ts,
                    "sim_send_ts": float(self._sim_send_ts) if self._sim_send_ts is not None else None,
                    "time_mode": self.time_mode,
                    "training_budget_s": _modeled_delay_s,
                    "remaining_time_s": _remaining_time,
                    "overran": _overran,
                    "grad_norm_epoch1": self._grad_norm_epoch1,
                    "task_to_perform": getattr(self, "task_to_perform", None),
                    "lr": current_lr,
                    "pre_train_s": _pre_train_s,
                    "gpu_compute_s": _real_gpu_time_s,
                    "sleep_s": _remaining_time,
                    "post_train_s": _post_train_s,
                    **getattr(self, "_phase_times", {}),
                },
            )
            telemetry.emit(ev, **fields)
            self._emit_util_disparity(
                int(getattr(self, "_round", 0)), self._sim_now()
            )

        if not self.simulated and _remaining_time > 0:
            time.sleep(_remaining_time)

        _cycle_elapsed = time.time() - _cycle_start
        if self.simulated:
            logger.info(
                f"[TRAIN_CYCLE] Trainer {self.trainer_id} round={self._round} "
                f"time_mode=simulated: wall={_cycle_elapsed:.2f}s "
                f"GPU={_real_gpu_time_s:.2f}s budget={_modeled_delay_s:.1f}s "
                f"pre={_pre_train_s:.2f}s post={_post_train_s:.2f}s "
                f"virtual_advance={sim_round_duration:.2f}s "
                f"{'OVERRUN' if _overran else 'OK'} "
                f"sct={self._sim_completion_ts:.2f}"
            )
        else:
            logger.info(
                f"[TRAIN_CYCLE] Trainer {self.trainer_id} round={self._round} "
                f"time_mode=real: wall={_cycle_elapsed:.2f}s "
                f"GPU={_real_gpu_time_s:.2f}s budget={_modeled_delay_s:.1f}s "
                f"pre={_pre_train_s:.2f}s post={_post_train_s:.2f}s "
                f"sleep={_remaining_time:.2f}s total={sim_round_duration:.1f}s "
                f"{'OVERRUN' if _overran else 'OK'}"
            )

    def _train_epoch(self, epoch):
        self.model.train()
        
        # Log memory for first epoch to track per-batch memory
        if epoch == 1:
            self.memory_profiler.log_component_memory(f"epoch_{epoch}", "START")

        batches_processed = 0
        last_loss = None
        # Accumulate per-step gradient L2 on epoch 1 (telemetry: relate update
        # magnitude to amount of unlocked data under streaming).
        _grad_norm_accum = 0.0
        _grad_norm_batches = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True for better memory
            output = self.model(data)

            if self.use_oort_loss_fn == "False":
                # Loss function to use with Fedbuff
                loss = F.nll_loss(output, target)
            elif self.use_oort_loss_fn == "True":
                # Calculate statistical utility of a trainer while
                # calculating loss
                loss = self.oort_loss(output, target, epoch, batch_idx)

            # accumulate per-round local training accuracy (FedDance A_m signal)
            self.update_local_accuracy(output, target)

            loss.backward()

            # Epoch-1 gradient L2 (telemetry only): one fused GPU reduction and
            # a single .item() sync per batch, instead of a .item() per param
            # (which forced ~12 GPU->CPU syncs/batch on the shared-GPU queue).
            if epoch == 1 and telemetry.is_enabled():
                _gsq = None
                for p in self.model.parameters():
                    if p.grad is not None:
                        s = p.grad.detach().pow(2).sum()
                        _gsq = s if _gsq is None else _gsq + s
                if _gsq is not None:
                    _grad_norm_accum += float(_gsq.sqrt().item())
                    _grad_norm_batches += 1

            self.optimizer.step()
            batches_processed += 1

            # Detach tensors to break computation graph and free memory
            # Log every batch for small trainers, every 100 for large trainers
            num_batches = len(self.train_loader)
            should_log = (num_batches <= 10) or (batch_idx % 100 == 0)
            if should_log:
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100.0 * batch_idx / len(self.train_loader)
                # Use .item() and detach to avoid keeping computation graph
                loss_val = loss.detach().item()
                last_loss = loss_val
                logger.info(
                    f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]" 
                    f"\tloss: {loss_val:.6f}"
                )
            
            # Drop references so the graph/activations can be freed; the
            # caching allocator reuses the blocks for the next batch without
            # an explicit (and, under co-location, costly) empty_cache().
            del output, data, target, loss

        # normalize statistical utility of a trainer based on the size
        # of the dataset
        self.normalize_stat_utility(epoch)

        if epoch == 1:
            self._grad_norm_epoch1 = (
                _grad_norm_accum / _grad_norm_batches
                if _grad_norm_batches
                else None
            )

        # Log memory after first epoch (no-op unless profiling enabled)
        if epoch == 1:
            self.memory_profiler.log_component_memory(f"epoch_{epoch}", "END")

        return batches_processed, last_loss

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement only forward pass evaluate if the trainer is available to train or to evaluate
        # Evaluate after train is written in the train_epoch method itself

        # Evaluate will be skipped if one of these three is satisfied:
        # 1. task_to_perform is train
        # 2. switch to check for three_state_avl is off
        # 3. Trainer is unavailable and we don't want it to wait for availability
        if (
            self.task_to_perform != "eval"
            or self.client_notify["trace"] == "two_state"
            or (
                self.avl_state == TrainerAvailState.UN_AVL
                and self.wait_until_next_avl == "False"
            )
        ):
            logger.warning(
                f"Evaluate (forward pass) will not be run for trainer id {self.trainer_id}. task_to_perform = {self.task_to_perform} and trainer avl_state = {self.avl_state.value} and wait_until_next_avl = {self.wait_until_next_avl}"
            )
            return

        # simulated mode: refresh availability at this task's sim-time.
        self._refresh_avl_for_sim()
        if self.avl_state == TrainerAvailState.UN_AVL:
            if self.simulated:
                logger.info(
                    f"Trainer id {self.trainer_id} unavailable for eval "
                    f"(simulated, sim_t={self._sim_now()}); skipping."
                )
                return
            logger.warning(
                f"Trainer id {self.trainer_id} is not available to perform forward pass evaluate. Waiting for it to be available"
            )
            while self.avl_state == TrainerAvailState.UN_AVL:
                time.sleep(1)

        # Use the same currently-visible data as training (no-op if off)
        if self.data_streaming_enabled:
            self._rebuild_stream_loader()

        logger.info(f"Starting eval (forward pass) for trainer id {self.trainer_id}")
        for epoch in range(1, self.epochs + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if self.use_oort_loss_fn == "False":
                    # Loss function to use with Fedbuff
                    loss = F.nll_loss(output, target)
                elif self.use_oort_loss_fn == "True":
                    # Calculate statistical utility of a trainer while
                    # calculating loss
                    loss = self.oort_loss(output, target, epoch, batch_idx)
                if batch_idx % 100 == 0:
                    done = batch_idx * len(data)
                    total = len(self.train_loader.dataset)
                    percent = 100.0 * batch_idx / len(self.train_loader)
                    logger.info(
                        f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                        f"\tloss: {loss.item():.6f}"
                    )

            # normalize statistical utility of a trainer based on the size
            # of the dataset
            self.normalize_stat_utility(epoch)
        # Eval is ~20x faster than training (NPUs don't support training), so
        # the modeled eval delay is training_delay_s/20. real mode sleeps it;
        # simulated mode skips it (folded into the reported sim duration).
        if self.training_delay_enabled and not self.simulated:
            eval_delay = math.floor(self.training_delay_s / 20.0)
            time.sleep(eval_delay)
            logger.debug(
                f"Delayed eval time for trainer " f"{self.trainer_id} by {eval_delay}s"
            )

    def initiate_heartbeat(self) -> None:
        while True:
            # dup_check_and_sleep operates on a copy to avoid mutating state on the heartbeat thread
            time.sleep(self.heartbeats_second_freq)
            self.dup_check_and_sleep()
            logger.debug("Initiating send heartbeat to aggregator")
            self.send_heartbeat_to_agg()

    def notify_trainer_avail(self) -> None:
        while True:
            time.sleep(1)  # Will check every 1 second
            self.check_and_update_state_avl()


def main():
    import argparse
    import json
    import signal
    import atexit

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.json",
        help="Path to config JSON file",
        required=False,
    )
    parser.add_argument(
        "--config-json",
        type=str,
        help="Config as JSON string (alternative to --config file)",
        required=False,
    )

    # Add a parser argument to get battery threshold (either 50 or 75)
    parser.add_argument(
        "--battery_threshold",
        type=int,
        choices=[50, 75],
        default=50,
        help="Battery threshold for the trainer 3-state events (either 50 or 75)",
        required=False,
    )

    # Simulation time mode (replaces the removed speedup_factor).
    parser.add_argument(
        "--time_mode",
        type=str,
        choices=["real", "simulated"],
        default="simulated",
        help="'real': sleep modeled delays at true pace. 'simulated': skip "
        "sleeps; aggregator orders updates by a virtual clock.",
        required=False,
    )

    args = parser.parse_args()
    
    # Early startup logging - print to ensure it appears even if logger not configured yet
    print(f"[TRAINER STARTUP] Process started, PID: {os.getpid()}")

    # Handle config loading: either from file or JSON string
    if args.config_json:
        # Load config from JSON string (new programmatic spawning mode)
        config_dict = json.loads(args.config_json)
        print(f"[TRAINER STARTUP] Loaded config from JSON string")
        # Create a temporary config file or pass dict directly
        # For now, write to temp file for compatibility with Config class
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_config_path = f.name

        try:
            config = Config(temp_config_path)
        finally:
            # Clean up temp file even if Config() fails
            os.unlink(temp_config_path)
    elif args.config:
        # Load config from file (legacy mode)
        config = Config(args.config)
        print(f"[TRAINER STARTUP] Loaded config from file: {args.config}")
    else:
        raise ValueError("Must provide either --config or --config-json")

    print(f"[TRAINER STARTUP] Creating trainer object...")
    t = PyTorchCifar10Trainer(config, args.battery_threshold, args.time_mode)

    # Structured telemetry (no-op unless $FLAME_TELEMETRY_DIR is set by the
    # launcher). One JSONL file per trainer process.
    telemetry.configure(role="trainer", end_id=str(t.trainer_id))
    
    print(f"[TRAINER STARTUP] Trainer created - ID: {t.trainer_id}, Job: {t.config.job.job_id}")
    logger.info(f"========== TRAINER STARTED: ID={t.trainer_id}, PID={os.getpid()} ==========")
    
    print(
        f"# Trainer id: {t.trainer_id}, time_mode: {t.time_mode}, "
        f"has heartbeats_enabled: {t.heartbeats_enabled}, "
        f"has client_notify: {t.client_notify['enabled']}, "
        f"training_delay_enabled: {t.training_delay_enabled}, "
        f"training_delay_s: {t.training_delay_s}"
    )

    # Register exit handler to generate memory report
    def cleanup_and_report():
        """Generate memory profiling report on exit."""
        try:
            report = t.memory_profiler.generate_report()
            logger.info(f"\n{report}")
            print(f"\n{report}")
        except Exception as e:
            logger.error(f"Error generating memory report: {e}")
    
    atexit.register(cleanup_and_report)
    
    # Handle SIGTERM gracefully
    def signal_handler(signum, frame):
        logger.info(f"Trainer {t.trainer_id} received signal {signum}, generating report...")
        cleanup_and_report()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if t.heartbeats_enabled == "True":
        logger.info(
            f"Will initiate thread to send heartbeats for " f"trainer {t.trainer_id}"
        )
        heartbeat_thread = threading.Thread(target=t.initiate_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    elif t.client_notify["trace"] is not None:
        logger.info(
            f"Will initiate thread to update state of " f"trainer {t.trainer_id}"
        )
        if t.client_notify["enabled"] == "True":
            logger.info(f"Will send avail notifications for trainer {t.trainer_id}")
        # Note that even though trainer sends notifications, only
        # async_oort will use it. Other selectors will not use it so
        # it can remain enabled.
        avail_notify_thread = threading.Thread(target=t.notify_trainer_avail)
        avail_notify_thread.daemon = True
        avail_notify_thread.start()

    print(f"[TRAINER STARTUP] Starting compose and run for trainer {t.trainer_id}...")
    logger.info(f"Trainer {t.trainer_id} initiating compose() and run() - will now connect to aggregator")
    t.compose()
    t.run()


if __name__ == "__main__":
    main()
