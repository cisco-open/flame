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
"""CIFAR10 SCAFFOLD Trainer."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from flame.common.util import get_dataset_filename
from flame.config import Config
from flame.mode.horizontal.scaffold.trainer import Trainer

logger = logging.getLogger(__name__)


class CNN(torch.nn.Module):
    """CNN for CIFAR-10"""

    def __init__(self):
        """Initialize."""
        super(CNN, self).__init__()
        self.num_classes = 10
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, path, train=False):
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor()])

        npz_file = np.load(path)

        self.x = npz_file["x"]
        self.y = npz_file["y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.train:
            img = (
                np.flip(img, axis=2).copy() if (np.random.rand() > 0.5) else img
            )  # Horizontal flip
            if np.random.rand() > 0.5:
                # Random cropping
                pad = 4
                extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(
                    np.float32
                )
                extended_img[:, pad:-pad, pad:-pad] = img
                dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                img = extended_img[:, dim_1 : dim_1 + 32, dim_2 : dim_2 + 32]

        img = np.moveaxis(img, 0, -1)
        img = self.transform(img)
        label = self.y[idx]
        return img, label


class PyTorchCIFAR10Trainer(Trainer):
    """PyTorch CIFAR10 Trainer"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_size = 0

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = None
        self.val_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.weight_decay = self.config.hyperparameters.weight_decay
        self.batch_size = self.config.hyperparameters.batch_size
        self._round = 1
        self._rounds = self.config.hyperparameters.rounds

    def initialize(self) -> None:
        """Initialize role."""

        self.model = CNN().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def load_data(self) -> None:
        """Load data."""
        path = get_dataset_filename(self.config.dataset)

        # data does not require transform
        train_dataset = CIFAR10Dataset(path=path, train=True)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.dataset_size = len(train_dataset)

    def train(self) -> None:
        """Train a model."""
        self.reset_batch_counter()

        for epoch in range(self.epochs):
            self.model.train()
            loss_lst = list()

            if self.training_is_done():
                break

            for data, label in self.train_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, label.reshape(-1).long()) / list(
                    label.size()
                )[0] + self.regularizer.get_term(curr_model=self.model)

                loss_lst.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=10
                )
                self.optimizer.step()

                self.increment_batch_counter()
                if self.training_is_done():
                    break

            train_loss = sum(loss_lst) / len(loss_lst)
            self.update_metrics({"Train Loss": train_loss})

    def evaluate(self) -> None:
        """Evaluate the model."""
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchCIFAR10Trainer(config)
    t.compose()
    t.run()
