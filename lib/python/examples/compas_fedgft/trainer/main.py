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
"""COMPAS FedGFT trainer for PyTorch."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flame.common.util import get_dataset_filename
from flame.config import Config
from flame.mode.horizontal.fedgft.trainer import Trainer

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


class PyTorchCompasTrainer(Trainer):
    """PyTorch COMPAS FedGFT Trainer"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_size = 0

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = None
        self.val_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size
        self.learning_rate = self.config.hyperparameters.learning_rate

        self._round = 1
        self._rounds = self.config.hyperparameters.rounds

    def initialize(self) -> None:
        """Initialize role."""
        self.model = LogisticRegression(in_features=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self) -> None:
        filename = get_dataset_filename(self.config.dataset)
        train_dataset = CompasDataset(filename=filename)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.dataset_size = len(train_dataset)

    def train(self) -> None:
        for batch_idx, (data, target, group) in enumerate(self.train_loader):
            data, target, group = (
                data.to(self.device),
                target.to(self.device),
                group.to(self.device),
            )
            self.optimizer.zero_grad()
            output = self.model(data)
            acc_loss = F.nll_loss(output, target)
            f_loss = self.regularizer.get_term(
                output=output, target=target, group=group
            )
            loss = acc_loss + f_loss
            loss.backward()
            self.optimizer.step()

    def evaluate(self) -> None:
        """Evaluate a model."""
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchCompasTrainer(config)
    t.compose()
    t.run()
