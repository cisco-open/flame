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
"""CIFAR-10 horizontal FL aggregator for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import argparse
import ast
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from torchvision.datasets import CIFAR10
from sortedcontainers import SortedDict


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # fedbuff
            "server_learning_rate": 40.9,
            "client_learning_rate": 0.000195,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "async, fedbuff",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            # fedbuff
            "client-concurrency C": 20,
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
        self.reject_stale_updates = (
            self.config.hyperparameters.reject_stale_updates or False
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
                f"Did not read oracular trainer jsons. Enabled value: {self.track_trainer_avail['enabled']}, type: {self.track_trainer_avail['type']}, trace: {self.track_trainer_avail.get('trace', '<unset>')}"
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

    def read_trainer_unavailability(self, trace=None) -> None:
        """
        Read availability trace pattern from central trace file.

        Currently returns None to disable oracular pre-loading.
        The selector can still access traces via availability_trace_file config.

        This allows the aggregator to work with dynamic trainer spawning
        without hardcoding the number of expected trainers.
        """
        return None

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
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
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
        self.dataset = Dataset(dataloader=self.test_loader)

    def train(self) -> None:
        """Train a model."""
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model every evalEveryNRounds rounds (default 10)."""
        eval_every = self.config.hyperparameters.eval_every_n_rounds or 10
        if self._round % eval_every != 0:
            return
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

        self.update_metrics({"test-loss": test_loss, "test-accuracy": test_accuracy})

        if self.log_to_wandb:
            wandb.log({"test_acc": test_accuracy, "test_loss": test_loss})
        self.loss_list.append(test_loss)

        logger.debug(f"loss list at cifar agg: {self.loss_list}")

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        pass


if __name__ == "__main__":
    from flame.launch.cli import load_config_from_argv

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Flag to log to Weights and Biases"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, help="Name of the Weights and Biases run"
    )
    args, _ = parser.parse_known_args()

    config = load_config_from_argv()

    a = PyTorchCifar10Aggregator(config, args.log_to_wandb, args.wandb_run_name)

    # Structured telemetry (no-op unless $FLAME_TELEMETRY_DIR is set by the
    # launcher). One JSONL file for the aggregator process.
    from flame import telemetry

    telemetry.configure(role="aggregator", end_id=config.job.job_id)

    a.compose()
    a.run()
