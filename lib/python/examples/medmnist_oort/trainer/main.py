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
"""MedMNIST Oort trainer for PyTorch."""

import logging
import torch
import torchvision
import numpy as np

from flame.config import Config
from flame.mode.horizontal.oort.trainer import Trainer

from PIL import Image

logger = logging.getLogger(__name__)


class CNN(torch.nn.Module):
    """CNN Class"""

    def __init__(self, num_classes):
        """Initialize."""
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(6), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PathMNISTDataset(torch.utils.data.Dataset):

    def __init__(self, split, transform=None, as_rgb=False):
        npz_file = np.load("pathmnist.npz")
        self.split = split
        self.transform = transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels']
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels']
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class PyTorchMedMNistTrainer(Trainer):
    """PyTorch MedMNist Trainer"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_size = 0

        self.model = None
        # Oort requires its loss function to have 'reduction' parameter
        self.loss_fn = torch.nn.CrossEntropyLoss

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size
        self.learning_rate = self.config.hyperparameters.learning_rate
        self._round = 1
        self._rounds = self.config.hyperparameters.rounds


    def initialize(self) -> None:
        """Initialize role."""

        self.model = CNN(num_classes=9)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def load_data(self) -> None:
        """MedMNIST Pathology Dataset
        The dataset is kindly released by Jakob Nikolas Kather, Johannes Krisam, et al. (2019) in their paper 
        "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study",
        and made available through Yang et al. (2021) in 
        "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis".
        Dataset Repo: https://github.com/MedMNIST/MedMNIST
        """

        self._download()
 
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = PathMNISTDataset(split='train', transform=data_transform)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.dataset_size = len(train_dataset)

    def _download(self) -> None:
        import requests
        r = requests.get(self.config.dataset, allow_redirects=True)
        open('pathmnist.npz', 'wb').write(r.content)
    
    def train(self) -> None:
        """Train a model."""
        self.model.load_state_dict(self.weights)

        self.model.train()

        self.reset_stat_utility()

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        # calculate trainer's statistical utility from the first minibatch on epoch 1
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            # calculate statistical utility of a trainer while calculating loss
            loss = self.oort_loss(output, target.squeeze(), epoch, batch_idx)

            loss.backward()
            self.optimizer.step()

        # normalize statistical utility of a trainer based on the size of the dataset
        self.normalize_stat_utility(epoch)


    def evaluate(self) -> None:
        """Evaluate a model."""
        pass
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchMedMNistTrainer(config)
    t.compose()
    t.run()
