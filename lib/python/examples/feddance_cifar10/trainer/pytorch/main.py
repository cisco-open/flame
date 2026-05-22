# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDance CIFAR-10 PyTorch trainer.

Calls oort_loss to populate I_m and update_local_accuracy to populate A_m.
Mirrors examples/cifar10 but wires the FedDance telemetry.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from flame.config import Config
from flame.mode.horizontal.trainer import Trainer

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


class FedDanceCifar10Trainer(Trainer):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_size = 0
        self.model = None
        self.device = None
        self.train_loader = None
        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.loss_fn = torch.nn.CrossEntropyLoss
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list

    def initialize(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)

    def load_data(self) -> None:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR10("./data", train=True, download=True, transform=tf)
        indices = torch.tensor(self.trainer_indices_list)
        dataset = data_utils.Subset(dataset, indices)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=2
        )

    def train(self) -> None:
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.reset_stat_utility()
        self.reset_local_accuracy()

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        self.normalize_stat_utility(self.epochs)
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.oort_loss(output, target, epoch, batch_idx)
            loss.backward()
            self.optimizer.step()
            self.update_local_accuracy(output, target)

    def evaluate(self) -> None:
        pass


if __name__ == "__main__":
    from flame.launch.cli import load_config_from_argv

    config = load_config_from_argv()
    t = FedDanceCifar10Trainer(config)
    t.compose()
    t.run()
