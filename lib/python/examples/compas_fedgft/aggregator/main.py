# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""COMPAS FedGFT aggregator for PyTorch."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flame.common.util import get_dataset_filename
from flame.config import Config
from flame.mode.horizontal.fedgft.top_aggregator import TopAggregator

logger = logging.getLogger(__name__)


class LogisticRegression(nn.Module):
    # require 32*32 pixels
    def __init__(self, in_features=120, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_classes),
        )

    def forward(self, x):
        logits = self.classifier(x)
        # Since NLL or KL divergence uses log-probability as input, we need to use log_softmax
        probs = F.log_softmax(logits, dim=1)
        return probs


class CompasDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        npz_file = np.load(filename)
        self.dataset = npz_file["dataset"]
        self.targets = npz_file["target"]
        self.group = npz_file["group"]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index, :], self.targets[index], self.group[index]


class PyTorchCompasAggregator(TopAggregator):
    """PyTorch COMPAS FedGFT Aggregator"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset = None

        self.batch_size = self.config.hyperparameters.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        """Initialize."""
        self.model = LogisticRegression(in_features=10).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_data(self) -> None:
        """Load a test dataset."""
        filename = get_dataset_filename(self.config.dataset)
        dataset = CompasDataset(filename=filename)

        self.test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self.dataset_size = len(dataset)

    def pre_process(self) -> None:
        """Log and report Bias."""
        curr_bias = abs(self.optimizer.get_bias())
        logger.info(f"Bias: {curr_bias}")
        self.update_metrics({"bias": curr_bias})

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, group in self.test_loader:
                data, target, group = (
                    data.to(self.device),
                    target.to(self.device),
                    group.to(self.device),
                )
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

        logger.info(f"Loss: {test_loss}")
        logger.info(f"Accuracy: {test_accuracy}")

        self.update_metrics({"test-loss": test_loss, "test-accuracy": test_accuracy})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchCompasAggregator(config)
    a.compose()
    a.run()
