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
"""MedMNIST FedProx trainer for PyTorch Using Proximal Term."""




import logging

from copy import deepcopy
from flame.common.util import get_dataset_filename
from flame.config import Config
from flame.mode.horizontal.feddyn.trainer import Trainer
import torch
import torchvision
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

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

    def __init__(self, split, filename, transform=None, as_rgb=False):
        npz_file = np.load(filename)
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
        self.device = torch.device("cpu")

        self.train_loader = None
        self.val_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size
        self._round = 1
        self._rounds = self.config.hyperparameters.rounds

    def initialize(self) -> None:
        """Initialize role."""

        self.model = CNN(num_classes=9).to(self.device)
        # ensure that weight_decay = 0 for FedDyn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_data(self) -> None:
        """MedMNIST Pathology Dataset
        The dataset is kindly released by Jakob Nikolas Kather, Johannes Krisam, et al. (2019) in their paper 
        "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study",
        and made available through Yang et al. (2021) in 
        "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis".
        Dataset Repo: https://github.com/MedMNIST/MedMNIST
        """

        filename = get_dataset_filename(self.config.dataset)
 
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = PathMNISTDataset(split='train', filename=filename, transform=data_transform)
        val_dataset = PathMNISTDataset(split='val', filename=filename, transform=data_transform)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4 * torch.cuda.device_count(),
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 * torch.cuda.device_count(),
            pin_memory=True,
            drop_last=True
        )

        self.dataset_size = len(train_dataset)

    def train(self) -> None:
        """Train a model."""
        
        # save global model first
        prev_model = deepcopy(self.model)

        for epoch in range(self.epochs):
            self.model.train()
            loss_lst = list()

            for data, label in self.train_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # proximal term included in loss
                loss = self.criterion(output, label.squeeze()) + self.regularizer.get_term(curr_model = self.model, prev_model = prev_model)
                
                # back to normal stuff
                loss_lst.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_loss = sum(loss_lst) / len(loss_lst)
            self.update_metrics({"Train Loss": train_loss})

    def evaluate(self) -> None:
        """Evaluate a model."""
        self.model.eval()
        loss_lst = list()
        labels = torch.tensor([],device=self.device)
        labels_pred = torch.tensor([],device=self.device)
        with torch.no_grad():
            for data, label in self.val_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label.squeeze())
                loss_lst.append(loss.item())
                labels_pred = torch.cat([labels_pred, output.argmax(dim=1)], dim=0)
                labels = torch.cat([labels, label], dim=0)

        labels_pred = labels_pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        val_acc = accuracy_score(labels, labels_pred)

        val_loss = sum(loss_lst) / len(loss_lst)
        self.update_metrics({"Val Loss": val_loss, "Val Accuracy": val_acc, "Testset Size": len(self.val_loader)})
        logger.info(f"Test Loss: {val_loss}")
        logger.info(f"Test Accuracy: {val_acc}")
        logger.info(f"Test Set Size: {len(self.val_loader)}")
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = PyTorchMedMNistTrainer(config)
    t.compose()
    t.run()
