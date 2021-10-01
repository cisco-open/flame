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

import time

import numpy as np
from fledge.channel_manager import ChannelManager
from fledge.config import Config
from tensorflow import keras
from tensorflow.keras import layers

# keras mnist example from https://keras.io/examples/vision/mnist_convnet/


class Trainer(object):
    def __init__(self, config: Config):
        self.config = config
        self.cm = ChannelManager()
        self.cm(config)
        self.cm.join('param-channel')

        self._rounds = 5
        if 'rounds' in self.config.hyperparameters:
            self._rounds = self.config.hyperparameters['rounds']

    def prepare(self):
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print("y_train shape:", y_train.shape)
        print(y_train.shape[0], "train samples")

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

            print(f'>>> round {i+1}')

            # one aggregator is sufficient
            end = ends[0]
            weights = channel.recv(end)

            self._model.set_weights(weights)
            self.train()

            # craft a message to inform aggregator
            data = (self._model.get_weights(), len(self._x_train))
            channel.send(end, data)

            # increase round
            i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)
    trainer = Trainer(config)

    print("Starting trainer...")
    trainer.run()
