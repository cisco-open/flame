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

import ast
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.oort.top_aggregator import TopAggregator
from torchvision.datasets import CIFAR10
from sortedcontainers import SortedDict


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # fedbuff "server_learning_rate": 40.9,
            # "client_learning_rate": 0.000195,
            # oort
            "client_learning_rate": 0.04,
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
                f"Did not read oracular trainer jsons. Enabled value: {self.track_trainer_avail['enabled']}, type: {self.track_trainer_avail['type']}, trace: {self.track_trainer_avail['trace']}"
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

    def read_trainer_unavailability(self, trace=None) -> dict:
        print(f"Came to read_trainer_unavailability, trace: {trace}")
        trainer_events_dict = {}

        # Set path to read JSON files from (TODO: Remove hardcoding later)
        files_path = "../../trainer/config_dir0.1_num300_traceFail_6d_3state_oort"

        # Set range of trainer IDs to read from
        trainer_start_num = 1
        trainer_end_num = 300

        for i in range(trainer_start_num, trainer_end_num + 1):
            dirname = os.path.dirname(__file__)
            file_path = os.path.join(dirname, files_path, f"trainer_{i}.json")

            with open(file_path) as f:
                trainer_json = json.load(f)
                curr_trainer_id = trainer_json["taskid"]
                event_list = ast.literal_eval(trainer_json["hyperparameters"][trace])

                # SortedDict for efficient timestamp lookup
                state_dict = SortedDict()

                # Process the events
                for timestamp, event_name in event_list:
                    state_dict[timestamp] = event_name

                trainer_events_dict[curr_trainer_id] = state_dict
                print(f"Completed file read for {file_path}")

        print("Completed reading all trainer unavailability from files")
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
            "num_workers": 2,
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

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in aggregator
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")
    # Add the --log_to_wandb argument
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Flag to log to Weights and Biases"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, help="Name of the Weights and Biases run"
    )

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchCifar10Aggregator(config, args.log_to_wandb, args.wandb_run_name)
    a.compose()
    a.run()
