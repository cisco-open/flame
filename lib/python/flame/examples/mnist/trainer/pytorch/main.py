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
"""MNIST horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from pytorch:
https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class PyTorchMnistTrainer(Trainer):
    """PyTorch Mnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters['epochs']
        self.batch_size = 16
        if 'batchSize' in self.config.hyperparameters:
            self.batch_size = self.config.hyperparameters['batchSize']

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)

    def load_data(self) -> None:
        """Load data."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        dataset = datasets.MNIST('./data',
                                 train=True,
                                 download=True,
                                 transform=transform)

        indices = torch.arange(2000)
        dataset = data_utils.Subset(dataset, indices)
        train_kwargs = {'batch_size': self.batch_size}

        self.train_loader = torch.utils.data.DataLoader(
            dataset, **train_kwargs)

    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100. * batch_idx / len(self.train_loader)
                logger.info(f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                            f"\tloss: {loss.item():.6f}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchMnistTrainer(config)
    t.compose()
    t.run()
