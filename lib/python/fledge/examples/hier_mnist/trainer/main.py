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

import logging
import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from ....channel_manager import ChannelManager

logger = logging.getLogger(__name__)

# keras mnist example from https://keras.io/examples/vision/mnist_convnet/


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
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

        def get_split_array(arr):
            split_array = []
            arr_idx = 0
            for idx in range(self._n_split):
                arr_idx += int((len(arr)) * self._split_weight * (idx + 1))
                split_array.append(arr_idx)

            return split_array

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        split_array = get_split_array(x_train)
        x_train = np.split(x_train, split_array)[self._split_idx]

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        split_array = get_split_array(y_train)
        y_train = np.split(y_train, split_array)[self._split_idx]

        logger.info(f'x_train shape: {x_train.shape}')
        logger.info(f'{x_train.shape[0]} train samples')
        logger.info(f'y_train shape: {y_train.shape}')
        logger.info(f'{y_train.shape[0]} train samples')

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        self._model = model
        self._x_train = x_train
        self._y_train = y_train

    def train(self):
        batch_size = 128
        epochs = 1

        self._model.fit(
            self._x_train,
            self._y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1
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

            logger.info(f'>>> round {i+1}')

            # one aggregator is sufficient
            end = ends[0]
            weights = channel.recv(end)
            if weights is None:
                continue

            self._model.set_weights(weights)
            self.train()

            # craft a message to inform aggregator
            data = (self._model.get_weights(), len(self._x_train))
            channel.send(end, data)

            # increase round
            i += 1


# example cmd in the following:
# python3 -m fledge.examples.hier_mnist.trainer.main --config fledge/examples/hier_mnist/trainer/config_us.json --n_split 2 --rounds 3 --split_idx 0
# run the above command in fledge/lib/python folder
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument(
        '--n_split',
        type=int,
        default=1,
        help='number of splits of a training dataset'
    )
    parser.add_argument(
        '--split_idx',
        type=int,
        default=0,
        help='index of split between 0 and (n_split-1)'
    )
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    trainer = Trainer(args.config, args.n_split, args.split_idx, args.rounds)
    trainer.run()
