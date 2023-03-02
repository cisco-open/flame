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
"""Adult income data horizontal FL trainer for PyTorch."""

import logging
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flame.common.constants import DATA_FOLDER_PATH
from flame.common.util import install_packages
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer

install_packages(['scikit-learn'])

from sklearn.model_selection import train_test_split

from utils import MyAdultDataset, clean_dataframe, process_dataframe

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


class PyTorchAdultTrainer(Trainer):
    """PyTorch Adult/Census Income Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0

        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net(self.input_dim).to(self.device)
        self.criterion = nn.BCELoss().to(self.device)

    def load_data(self) -> None:
        """Load data."""
        data_path = os.path.join(DATA_FOLDER_PATH, "train.csv")
        data = pd.read_csv(data_path,
                           header=0,
                           skipinitialspace=True,
                           na_values="?")

        clean_dataframe(data)

        X, y, features = process_dataframe(data,
                                           target_column="income-per-year")

        xTrain, xTest, yTrain, yTest = train_test_split(X,
                                                        y,
                                                        random_state=1,
                                                        stratify=y)

        dataset = MyAdultDataset(xTrain, yTrain, features)
        self.train_loader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=20,
                                                        shuffle=True,
                                                        num_workers=4)

        self.input_dim = next(iter(self.train_loader))[0].shape[1]

    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-3,
                                    weight_decay=1e-4)

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch, max_iter=None):
        for batch_idx, (x, y) in enumerate(self.train_loader):
            if max_iter and batch_idx >= max_iter:
                break

            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.criterion(yhat[:, 0], y)
            loss.backward()

            self.optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(x)
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

    t = PyTorchAdultTrainer(config)
    t.compose()
    t.run()
