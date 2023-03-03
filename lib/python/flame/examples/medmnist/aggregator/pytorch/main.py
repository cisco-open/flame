# Copyright 2022 Cisco Systems, Inc. and its affiliates
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

"""MedMNIST aggregator for PyTorch."""

import logging

from flame.config import Config
from flame.dataset import Dataset # Not sure why we need this.
from flame.mode.horizontal.top_aggregator import TopAggregator
import torch

logger = logging.getLogger(__name__)

class CNN(torch.nn.Module):
    """CNN Class"""

    def __init__(self, num_classes):
        """Initialize."""
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PyTorchMedMNistAggregator(TopAggregator):
    """PyTorch MedMNist Aggregator"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset: Dataset = None # Not sure why we need this.

    def initialize(self):
        """Initialize."""
        self.model = CNN(num_classes=9)

    def load_data(self) -> None:
        """Load a test dataset."""
        # Implement this if loading data is needed in aggregator
        pass

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Implement this if testing is needed in aggregator
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchMedMNistAggregator(config)
    a.compose()
    a.run()
