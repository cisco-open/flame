# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from monai.apps import download_and_extract
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations, AddChannel, AsDiscrete, Compose, LoadImage, ScaleIntensity,
    ToTensor
)
from monai.utils import set_determinism
from sklearn.metrics import classification_report

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


class Aggregator(object):
    def __init__(self, config_file: str, rounds=1):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('param-channel')

        self._rounds = rounds

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
        val_x = [image_files_list[i] for i in val_indices]
        test_x = [image_files_list[i] for i in test_indices]
        test_y = [image_class[i] for i in test_indices]

        print(
            f"Training count: {len(train_x)}, Validation count: "
            f"{len(val_x)}, Test count: {len(test_x)}"
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

        test_ds = MedNISTDataset(test_x, test_y, val_transforms)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=300, num_workers=10
        )

        device = torch.device("cpu")
        model = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=num_class
        ).to(device)

        self._model = model
        self._device = device
        self._test_loader = test_loader
        self._class_names = class_names

        self.act = act
        self.to_onehot = to_onehot

    def test(self):
        self._model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for test_data in self._test_loader:
                test_images = test_data[0].to(self._device)
                test_labels = test_data[1].to(self._device)

                pred = self._model(test_images).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(test_labels[i].item())
                    y_pred.append(pred[i].item())

        print(
            classification_report(
                y_true, y_pred, target_names=self._class_names, digits=4
            )
        )

    def run(self):
        self.prepare()
        channel = self.cm.get('param-channel')

        i = 0
        while i < self._rounds:
            print(f'>>> round {i+1}')
            # send out global model parameters to trainers
            for end in channel.ends():
                channel.send(end, self._model.state_dict())

            # TODO: lines below need to be abstracted for different
            # frontends (e.g., keras, pytorch, etc)
            total = 0
            state_array = []
            # receive local model parameters from trainers
            for end in channel.ends():
                msg = channel.recv(end)
                if not msg:
                    print('no data received')
                    continue

                state_dict = msg[0]
                count = msg[1]
                total += count
                state_array.append((state_dict, count))
                print(f'got {end}\'s parameters trained with {count} samples')

            if len(state_array) == 0 or total == 0:
                print('no local model parameters are obtained')
                time.sleep(1)
                continue

            count = state_array[0][1]
            rate = count / total
            global_state = state_array[0][0]

            for k, v in global_state.items():
                global_state[k] = v * rate

            for state_dict, count in state_array[1:]:
                rate = count / total

                for k in state_dict.keys():
                    global_state[k] += state_dict[k] * rate

            self._model.load_state_dict(global_state)
            self.test()
            i += 1


# example cmd: python3 -m flame.examples.mednist.aggregator.main --rounds 3
# run the above command in flame/lib/python folder
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    aggregator = Aggregator(
        'flame/examples/mednist/aggregator/config.json',
        args.rounds,
    )
    aggregator.run()
