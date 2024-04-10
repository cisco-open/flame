# Copyright 2024 Cisco Systems, Inc. and its affiliates
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
"""HIRE_FEMNIST horizontal hierarchical FL trainer for PyTorch."""

import logging
import random
import time
import csv
import os
import os.path
import warnings

from PIL import Image

from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet
TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"

import torch
import torch.optim as optim
from flame.config import Config
from flame.mode.horizontal.coord_syncfl.trainer import Trainer
import torchvision.models as tormodels

from fedscale.dataloaders.utils_data import get_data_transform

logger = logging.getLogger(__name__)

class FEMNIST():
    """
    FEMNIST dataloader. The implementation is based on FedScale

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, meta_dir, partition_id, dataset='train', transform=None, target_transform=None, imgview=False):

        self.data_file = dataset  # 'train', 'test', 'validation'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)
        self.meta_dir = meta_dir
        self.partition_id = partition_id

        # load data and targets
        self.data, self.targets = self.load_file(self.path)
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}

        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.root, imgName))

        # avoid channel error
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    datas.append(row[1])
                    labels.append(int(row[-1]))
                line_count += 1

        return datas, labels

    def load_file(self, path):
        datas, labels = self.load_meta_data(os.path.join(
            self.meta_dir, self.data_file, 'client-'+str(self.partition_id)+'-'+self.data_file+'.csv'))

        return datas, labels

def override(method):
    return method

class PyTorchFemnistTrainer(Trainer):
    """PyTorch Femnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.data_dir = "../../../../../third_party/benchmark/dataset/data/femnist/"
        self.loss_squared = 0
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_decay = 0.2
        self.local_steps = 30
        self.meta_dir = "/tmp/flame_dataset/femnist/"
        self.partition_id = 1

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = tormodels.__dict__["resnet18"](num_classes=62).to(self.device)

    def load_data(self) -> None:
        """Load data."""
        # Generate a random parition ID
        self.partition_id = random.randint(1, 2798)

        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST(self.data_dir, self.meta_dir, self.partition_id,
                                dataset='train', transform=train_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True, timeout=0,
                num_workers=0, drop_last=True)

    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=self.device)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)

            loss_list = loss.tolist()
            loss = loss.mean()

            temp_loss = sum(loss_list) / float(len(loss_list))
            self.loss_squared = sum([l ** 2 for l in loss_list]) / float(len(loss_list))

            # only measure the loss of the first epoch
            if self.completed_steps < len(self.train_loader):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (1. - self.loss_decay) * self.epoch_train_loss + self.loss_decay * temp_loss

            # Define the backward loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.completed_steps += 1

            if self.completed_steps == self.local_steps:
                break

        logger.info(f"loss: {loss.item():.6f} \t moving_loss: {self.epoch_train_loss}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass

    @override
    def save_metrics(self):
        """Save metrics in a model registry."""

        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    @override
    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_get_aggregator = Tasklet("get_aggregator", self._get_aggregator)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_init
                >> loop(
                    task_load_data >> task_get_aggregator >> task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchFemnistTrainer(config)
    t.compose()
    t.run()
