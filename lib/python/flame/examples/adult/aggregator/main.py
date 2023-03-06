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
"""Adult income data horizontal FL aggregator."""

import logging

import torch
import torch.nn as nn
from flame.config import Config
from flame.mode.horizontal.top_aggregator import TopAggregator

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self, input_size=30, scale=4):
        """Initialize."""
        super().__init__()
        self.ff = nn.Sequential(*[
            nn.Linear(input_size, 10 * scale),
            nn.ReLU(),
            # nn.Linear(10*scale, 10*scale),
            # nn.ReLU(),
            nn.Linear(10 * scale, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        """Forward."""
        return self.ff(x)


class PyTorchAdultAggregator(TopAggregator):
    """PyTorch Adult/Census Income Top Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset = None
        self.metrics = None
        self.model = None

        self.device = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 104  # use a value computed from trainer script
        self.model = Net(self.input_dim).to(self.device)

    def load_data(self) -> None:
        """Load data."""
        pass

    def train(self) -> None:
        """Train a model."""
        pass

    def evaluate(self) -> None:
        """Evaluate a model."""
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchAdultAggregator(config)
    t.compose()
    t.run()
