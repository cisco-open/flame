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


import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from monai.apps import download_and_extract
from monai.metrics import compute_roc_auc
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandFlip,
    RandRotate, RandZoom, ScaleIntensity, ToTensor
)
from monai.utils import set_determinism

from ....channel_manager import ChannelManager

# monai mednist example from
# https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


class Trainer(object):
    def __init__(self, config_file: str, n_split=1, split_idx=0, rounds=1):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('param-channel')

        self._n_split = n_split
        self._split_idx = split_idx
        self._rounds = rounds

        self._split_weight = 0
        for i in range(self._n_split):
            self._split_weight += 2**i
        self._split_weight = 1 / self._split_weight

    def prepare(self):
        root_dir = '/tmp'

        resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
        md5 = "0bc7306e7427e00ad1c5526a6677552d"

        compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
        data_dir = os.path.join(root_dir, "MedNIST")
        if not os.path.exists(data_dir):
            download_and_extract(resource, compressed_file, root_dir, md5)

        set_determinism(seed=0)

        class_names = sorted(
            x for x in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, x))
        )
        num_class = len(class_names)
        image_files = [
            [
                os.path.join(data_dir, class_names[i], x)
                for x in os.listdir(os.path.join(data_dir, class_names[i]))
            ] for i in range(num_class)
        ]
        num_each = [len(image_files[i]) for i in range(num_class)]
        image_files_list = []
        image_class = []
        for i in range(num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)
        image_width, image_height = PIL.Image.open(image_files_list[0]).size

        print(f"Total image count: {num_total}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")

        # plt.subplots(3, 3, figsize=(8, 8))
        # for i, k in enumerate(np.random.randint(num_total, size=9)):
        #     im = PIL.Image.open(image_files_list[k])
        #     arr = np.array(im)
        #     plt.subplot(3, 3, i + 1)
        #     plt.xlabel(class_names[image_class[k]])
        #     plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
        # plt.tight_layout()
        # plt.show()

        val_frac = 0.1
        test_frac = 0.1
        length = len(image_files_list)
        indices = np.arange(length)
        np.random.shuffle(indices)

        test_split = int(test_frac * length)
        val_split = int(val_frac * length) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices[val_split:]

        train_x = [image_files_list[i] for i in train_indices]
        train_y = [image_class[i] for i in train_indices]
        val_x = [image_files_list[i] for i in val_indices]
        val_y = [image_class[i] for i in val_indices]
        test_x = [image_files_list[i] for i in test_indices]
        test_y = [image_class[i] for i in test_indices]

        print(
            f"Training count: {len(train_x)}, Validation count: "
            f"{len(val_x)}, Test count: {len(test_x)}"
        )

        train_transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                ToTensor(),
            ]
        )

        val_transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                ToTensor()
            ]
        )

        act = Activations(softmax=True)
        to_onehot = AsDiscrete(to_onehot=True, n_classes=num_class)

        train_ds = MedNISTDataset(train_x, train_y, train_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=300, shuffle=True, num_workers=10
        )

        val_ds = MedNISTDataset(val_x, val_y, val_transforms)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=300, num_workers=10
        )

        device = torch.device("cpu")
        model = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=num_class
        ).to(device)

        self._model = model
        self._device = device
        self._train_ds = train_ds
        self._val_loader = val_loader

        self.act = act
        self.to_onehot = to_onehot

    def train(self):
        max_epochs = 1
        val_interval = 1
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-5)

        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            self._model.train()
            epoch_loss = 0
            step = 0

            for batch_data in self._train_loader:
                step += 1
                inputs = batch_data[0].to(self._device)
                labels = batch_data[1].to(self._device)

                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(
                    f"{step}/{len(self._train_ds) // self._train_loader.batch_size}, "
                    f"train_loss: {loss.item():.4f}"
                )

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self._model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor(
                        [], dtype=torch.float32, device=self._device
                    )
                    y = torch.tensor([], dtype=torch.long, device=self._device)
                    for val_data in self._val_loader:
                        val_images = val_data[0].to(self._device)
                        val_labels = val_data[1].to(self._device)

                        y_pred = torch.cat(
                            [y_pred, self._model(val_images)], dim=0
                        )
                        y = torch.cat([y, val_labels], dim=0)
                    y_onehot = self.to_onehot(y)
                    y_pred_act = self.act(y_pred)
                    auc_metric = compute_roc_auc(y_pred_act, y_onehot)
                    del y_pred_act, y_onehot
                    metric_values.append(auc_metric)
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    if auc_metric > best_metric:
                        best_metric = auc_metric
                        best_metric_epoch = epoch + 1
                    print(
                        f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                        f" current accuracy: {acc_metric:.4f}"
                        f" best AUC: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )

        print(
            f"train completed, best_metric: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )

    def run(self):
        self.prepare()
        channel = self.cm.get('param-channel')

        i = 0
        while i < self._rounds:
            ends = channel.ends()
            if len(ends) == 0:
                time.sleep(1)
                continue

            print(f'>>> round {i+1}')

            # one aggregator is sufficient
            end = ends[0]
            state_dict = channel.recv(end)
            if not state_dict:
                continue

            self._model.load_state_dict(state_dict)
            self.train()

            # craft a message to inform aggregator
            data = (self._model.state_dict(), len(self._train_ds.image_files))
            channel.send(end, data)

            # increase round
            i += 1
