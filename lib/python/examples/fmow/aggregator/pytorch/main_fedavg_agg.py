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
"""FMoW horizontal FL, FedAvg  SyncFL aggregator for PyTorch."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator

from pathlib import Path
import torch.utils.data as data_utils

# FMoW Imports
from model import build_model
from fmow_dataset import FMoWDataset
from config import load_config
from satellite_availability import SatelliteAvailability

def initialize_wandb():
    wandb.init(
        project="ft-distr-ml",
        config={
            "client_learning_rate": 0.25,
            "architecture": "CNN",
            "dataset": "FMoW",
            "fl-type": "sync, fedavg",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "Fedavg SyncFL no failure run",
        },
    )


logger = logging.getLogger(__name__)


class PyTorchFMoWAggregator(TopAggregator):
    """PyTorch FMoW Aggregator."""

    def __init__(self, config: Config, log_to_wandb: bool = False) -> None:
        """Initialize a class instance."""
        self.config = config
        self.fmow_cfg = load_config(self.config.hyperparameters.fmow_config_path)
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.loss_list = []

        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            initialize_wandb()

    def read_trainer_unavailability(self, trace=None, base_dir=None) -> dict | None:
        """Load satellite events locally; delegate other traces to Flame."""
        if trace != "satellite":
            return super().read_trainer_unavailability(trace, base_dir=base_dir)

        availability = SatelliteAvailability(self.fmow_cfg.availability.trace_path)
        registry_path = (
            Path(__file__).resolve().parents[2] / "metadata" / "trainer_registry.yaml"
        )
        events = availability.events_by_endpoint(registry_path)
        logger.info(f"[AVAIL] loaded {len(events)} satellite trainer traces")
        return events

    def initialize(self):
        """Initialize role."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = build_model(self.fmow_cfg.dataset.num_classes).to(self.device)


    def load_data(self) -> None:
        """Load a test dataset."""
        fmow_dataset = FMoWDataset(self.fmow_cfg.dataset.root_dir)
        fmow_dataset.set_transform(self.fmow_cfg.dataset.image_size)
        test_indices = fmow_dataset.get_split_indices("test").tolist()
        dataset = data_utils.Subset(fmow_dataset, test_indices)

        test_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": 0,  # Changed from 2 to 0 - reduces CPU RAM usage
            "pin_memory": True,  # Use pinned memory for faster CPU->GPU transfers
        }

        self.test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

        # store data into dataset for analysis (e.g., bias)
        self.dataset = Dataset(dataloader=self.test_loader)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Gate eval cadence (evalEveryNRounds) instead of evaluating every round;
        # the full test pass dominates per-round cost at n300. Used by the
        # FedDance arm (this is its aggregator stack). Always eval round 1.
        eval_every = (
            getattr(self.config.hyperparameters, "eval_every_n_rounds", 10) or 10
        )
        is_last_round = self._round >= self._rounds
        if self._round != 1 and (self._round % eval_every != 0) and not is_last_round:
            return
        # Off the critical path: snapshot weights now, run the test-set forward
        # pass in a daemon thread so the aggregator keeps progressing.
        eval_model = self._eval_snapshot_model()
        if eval_model is None:
            return
        round_num = self._round
        test_loader, device = self.test_loader, self.device

        max_batches = None if is_last_round else 5

        def _job():
            try:
                eval_model.eval()
                total_batches = len(test_loader) if max_batches is None else max_batches
                logger.info(f"[EVAL_START] round={round_num} total_batches={total_batches} device={device}")
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for i, (data, target) in enumerate(test_loader):
                        if max_batches is not None and i >= max_batches:
                            break
                        data, target = data.to(device), target.to(device)
                        output = eval_model(data)
                        test_loss += F.cross_entropy(output, target, reduction="sum").item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += data.size(0)
                        if i % 100 == 0:
                            logger.info(f"[EVAL_PROGRESS] round={round_num} batch={i}/{total_batches} samples={total}")
                        
                self._eval_emit(round_num, test_loss / total, correct / total)
                self._eval_inflight = False
            except Exception as e:  # eval must never break training
                logger.warning(f"[ASYNC_EVAL] failed (non-fatal): {e}")
                self._eval_inflight = False

        if is_last_round:
            _job()
        else:
            import threading
            threading.Thread(target=_job, daemon=True).start()

        # print to save to file
        logger.debug(f"loss list at fmow agg: {self.loss_list}")

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in aggregator
        pass


if __name__ == "__main__":
    import argparse

    from flame.launch.cli import load_config_from_argv

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_run_name", type=str)
    args, _ = parser.parse_known_args()

    config = load_config_from_argv()

    a = PyTorchFMoWAggregator(config, args.log_to_wandb)

    from flame import telemetry

    telemetry.configure(role="aggregator", end_id=config.job.job_id)

    a.compose()
    a.run()
