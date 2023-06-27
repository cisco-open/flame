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
"""CIFAR10 SCAFFOLD Aggregator for PyTorch."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from flame.common.util import get_dataset_filename
from flame.config import Config
from flame.mode.horizontal.scaffold.top_aggregator import TopAggregator

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


class PyTorchCIFAR10Aggregator(TopAggregator):
    """PyTorch CIFAR10 Aggregator"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset = None

        self.batch_size = self.config.hyperparameters.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        """Initialize."""
        self.model = CNN().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def load_data(self) -> None:
        """Load a test dataset."""
        path = get_dataset_filename(self.config.dataset)

        dataset = CIFAR10Dataset(path=path)
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self.dataset_size = len(dataset)

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        loss_lst = list()
        labels = torch.tensor([], device=self.device)
        labels_pred = torch.tensor([], device=self.device)
        with torch.no_grad():
            for data, label in self.loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label.squeeze())
                loss_lst.append(loss.item())
                labels_pred = torch.cat([labels_pred, output.argmax(dim=1)], dim=0)
                labels = torch.cat([labels, label], dim=0)

        labels_pred = labels_pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        val_acc = accuracy_score(labels, labels_pred)
        val_loss = sum(loss_lst) / self.dataset_size

        self.update_metrics(
            {
                "Test Loss": val_loss,
                "Test Accuracy": val_acc,
                "Testset Size": self.dataset_size,
            }
        )
        logger.info(f"Test Loss: {val_loss}")
        logger.info(f"Test Accuracy: {val_acc}")
        logger.info(f"Testset Size: {self.dataset_size}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchCIFAR10Aggregator(config)
    a.compose()
    a.run()
