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

"""HIRE_MNIST horizontal hierarchical FL trainer for Keras."""

import logging
from random import randrange

import numpy as np
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

class KerasMnistTrainer(Trainer):
    """Keras Mnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.weights = None
        self.dataset_size = 0

        self.num_classes = 10
        self.input_shape = (28, 28, 1)

        self.model = None
        self._x_train = None
        self._y_train = None

        self.epochs = self.config.hyperparameters['epochs']
        self.batch_size = 128
        if 'batchSize' in self.config.hyperparameters:
            self.batch_size = self.config.hyperparameters['batchSize']

    def initialize(self) -> None:
        """Initialize role."""
        model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax"),
        ])

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        self.model = model

    def load_data(self) -> None:
        """Load data."""
        if self._x_train and self._y_train:
            return

        # the data, split between train and test sets
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

        split_n = 10
        index = randrange(split_n)
        # reduce train sample size to reduce the runtime
        x_train = np.split(x_train, split_n)[index]
        y_train = np.split(y_train, split_n)[index]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        self._x_train = x_train
        self._y_train = y_train

    def train(self) -> None:
        """Train a model."""
        # set model weights given from aggregator
        self.model.set_weights(self.weights)

        self.model.fit(self._x_train,
                       self._y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.1)

        # save weights and dataset size so that
        # these two pieces of info can be shared with aggregator
        self.weights = self.model.get_weights()
        self.dataset_size = len(self._x_train)

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help='config file', required=True)
    args = parser.parse_args()
    config = Config(args.config)

    t = KerasMnistTrainer(config)
    t.compose()
    t.run()
