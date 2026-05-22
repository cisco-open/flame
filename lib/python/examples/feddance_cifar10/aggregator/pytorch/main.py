# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDance CIFAR-10 PyTorch aggregator."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator

logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FedDanceCifar10Aggregator(TopAggregator):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset: Dataset = None
        self.device = None
        self.test_loader = None
        self.batch_size = self.config.hyperparameters.batch_size or 16

    def initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)

    def load_data(self) -> None:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR10("./data", train=False, download=True, transform=tf)
        self.test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=-1)
                correct += int((pred == target).sum().item())
                total += int(target.numel())
        acc = correct / total if total else 0.0
        logger.info(f"[FEDDANCE_AGG] eval acc={acc:.4f} ({correct}/{total})")
        self.update_metrics({"test_accuracy": acc})


if __name__ == "__main__":
    from flame.launch.cli import load_config_from_argv

    config = load_config_from_argv()
    a = FedDanceCifar10Aggregator(config)
    a.compose()
    a.run()
