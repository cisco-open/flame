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
"""CIFAR-10 horizontal FL, OORT aggregator for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml

# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.oort.top_aggregator import TopAggregator
from torchvision.datasets import CIFAR10
from sortedcontainers import SortedDict


_METADATA_DIR = (Path(__file__).resolve().parents[3] / "_metadata")
_TRACE_KEY_TO_MOBIPERF_SUB = {
    "mobiperf_2st": "states_2st",
    "mobiperf_3st_50": "states_3st_50",
    "mobiperf_3st_75": "states_3st_75",
}


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # REFL/Oort CIFAR-10 uses client LR = 0.01 (from REFL config)
            # REFL uses LR decay: decay_factor=0.98 every decay_epoch=10 rounds
            # Oort baselines don't use LR decay
            "client_learning_rate": 0.01,
            "lr_decay_enabled": True,  # For REFL experiments
            "lr_decay_factor": 0.98,
            "lr_decay_epoch": 10,
            "min_learning_rate": 0.0001,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "sync, oort",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            # fedbuff "client-concurrency C": 20,
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "First oort no failure run",
        },
    )


logger = logging.getLogger(__name__)


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


class PyTorchCifar10Aggregator(TopAggregator):
    """PyTorch CIFAR-10 Aggregator."""

    def __init__(
        self, config: Config, log_to_wandb: bool, wandb_run_name: str = None
    ) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.track_trainer_avail = (
            self.config.hyperparameters.track_trainer_avail or None
        )
        self.trainer_event_dict = None
        if (
            self.track_trainer_avail["enabled"]
            and self.track_trainer_avail["type"] == "ORACULAR"
        ):
            self.trainer_event_dict = self.read_trainer_unavailability(
                self.track_trainer_avail["trace"]
            )
        else:
            print(
                "Did not read oracular trainer jsons. "
                f"enabled={self.track_trainer_avail.get('enabled')}, "
                f"type={self.track_trainer_avail.get('type')}, "
                f"trace={self.track_trainer_avail.get('trace', '<unset>')}"
            )
        print("self.trainer_event_dict: ", self.trainer_event_dict)

        self.loss_list = []

        # Use wandb logging if enabled
        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            initialize_wandb(run_name=wandb_run_name)

    def initialize(self):
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        
        # Initialize aggregator start time for oracular availability tracking
        self.agg_start_time_ts = time.time()
        logger.info(f"Aggregator initialized at timestamp: {self.agg_start_time_ts}")

    def read_trainer_unavailability(self, trace=None) -> dict:
        """Build trainer_id -> SortedDict(timestamp -> state) for `trace`.

        Reads from the shared examples/_metadata/ bundle (registry + traces),
        not from legacy per-trainer JSON files.
        """
        logger.info(f"Reading trainer unavailability for trace: {trace}")

        registry_path = _METADATA_DIR / "trainer_registry.yaml"
        with open(registry_path) as f:
            registry = yaml.safe_load(f)["trainers"]

        if trace in _TRACE_KEY_TO_MOBIPERF_SUB:
            sub = _TRACE_KEY_TO_MOBIPERF_SUB[trace]
            with open(_METADATA_DIR / "availability_traces/mobiperf_traces.yaml") as f:
                traces = yaml.safe_load(f)["traces"]

            def lookup(tk: str, trainer_id: int) -> list:
                return traces[f"device_{trainer_id:03d}"][sub]

        elif trace and trace.startswith("syn_"):
            with open(_METADATA_DIR / "availability_traces/synthetic_traces.yaml") as f:
                syn = yaml.safe_load(f)["traces"]
            if trace not in syn:
                logger.warning(f"trace {trace!r} not found in synthetic_traces.yaml")
                return None
            entry = syn[trace]
            per_trainer = entry.get("per_trainer", {}).get("n300", {})
            pattern = entry.get("pattern", [])

            def lookup(tk: str, trainer_id: int) -> list:
                return per_trainer.get(tk) or pattern

        else:
            logger.warning(f"unsupported trace name: {trace!r}")
            return None

        trainer_events_dict = {}
        for tk, meta in registry.items():
            trainer_id = meta["trainer_id"]
            task_id = meta["task_id"]
            events = lookup(tk, trainer_id)
            state_dict = SortedDict()
            for timestamp, state in events:
                state_dict[timestamp] = state
            trainer_events_dict[task_id] = state_dict

        logger.info(
            f"Loaded availability traces for {len(trainer_events_dict)} trainers "
            f"(trace={trace})"
        )
        return trainer_events_dict

    def load_data(self) -> None:
        """Load a test dataset."""
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = CIFAR10(
            "/home/dgarg39/flame/lib/python/examples/async_cifar10/data",
            train=False,
            download=True,
            transform=transform_test,
        )

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
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        test_loss /= total
        test_accuracy = correct / total

        logger.info(
            f"Test loss: {test_loss}, test accuracy: "
            f"{correct}/{total} ({test_accuracy})"
        )

        # update metrics after each evaluation so that the metrics can
        # be logged in a model registry.
        self.update_metrics({"test-loss": test_loss, "test-accuracy": test_accuracy})

        # add metrics to wandb log
        if self.log_to_wandb:
            wandb.log({"test_acc": test_accuracy, "test_loss": test_loss})
        self.loss_list.append(test_loss)

        # print to save to file
        logger.debug(f"loss list at cifar agg: {self.loss_list}")

    def get_curr_unavail_trainers(self) -> list:
        """
        Get list of currently unavailable trainers based on oracular traces.
        
        Uses binary search to find the most recent event for each trainer
        at the current time, and returns trainers that are in UN_AVL state.
        
        Returns:
            list: List of trainer IDs that are currently unavailable
        """
        curr_unavail_trainer_list = []
        
        if self.trainer_event_dict is None:
            return curr_unavail_trainer_list
        
        # Get aggregator time since start
        agg_time_since_start_s = time.time() - self.agg_start_time_ts
        
        for trainer_id, event_dict in list(self.trainer_event_dict.items()):
            logger.debug(
                f"Checking trainer {trainer_id}'s availability at time {agg_time_since_start_s}s"
            )
            
            if not event_dict:
                continue  # Skip if no events for trainer
            
            # Binary search for closest past event
            # bisect_right returns insertion point, subtract 1 for last event <= time
            idx = event_dict.bisect_right(agg_time_since_start_s) - 1
            logger.debug(f"Trainer {trainer_id}: event index = {idx}")
            
            if idx >= 0:
                # Get the most recent event
                most_recent_event = event_dict.peekitem(idx)
                most_recent_event_ts = most_recent_event[0]
                most_recent_event_state = most_recent_event[1]
                
                logger.debug(
                    f"Trainer {trainer_id}: most recent event at {most_recent_event_ts}s -> {most_recent_event_state}"
                )
                
                if most_recent_event_state == "UN_AVL":
                    logger.debug(f"Marking trainer {trainer_id} as unavailable")
                    curr_unavail_trainer_list.append(trainer_id)
                elif most_recent_event_state == "AVL_TRAIN":
                    logger.debug(f"Trainer {trainer_id} is available")
                else:
                    logger.warning(
                        f"Trainer {trainer_id} has unknown state: {most_recent_event_state}"
                    )
        
        logger.info(
            f"[ORACULAR] Current unavailable trainers: {len(curr_unavail_trainer_list)} "
            f"out of {len(self.trainer_event_dict)} total @ time={agg_time_since_start_s:.1f}s"
        )
        
        return curr_unavail_trainer_list

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

    a = PyTorchCifar10Aggregator(config, args.log_to_wandb, args.wandb_run_name)
    a.compose()
    a.run()
