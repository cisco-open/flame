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
"""MNIST horizontal FL aggregator for Keras."""

import logging
from random import randrange

import numpy as np
from fledge.config import Config
from fledge.mode.horizontal.aggregator import Aggregator
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class KerasMnistAggregator(Aggregator):
    """Keras Mnist Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.weights = None
        self.metrics = None
        self.model = None

        self.num_classes = 10
        self.input_shape = (28, 28, 1)

        self._x_test = None
        self._y_test = None

    def initialize(self):
        """Initialize role."""
        if not self.model:
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

        # initialize the global model weights
        self.weights = self.model.get_weights()

    def load_data(self) -> None:
        """Load a test dataset."""
        # the data, split between train and test sets
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

        split_n = 5
        index = randrange(split_n)
        # reduce test sample size to reduce the runtime
        x_test = np.split(x_test, split_n)[index]
        y_test = np.split(y_test, split_n)[index]

        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_test = np.expand_dims(x_test, -1)
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self._x_test = x_test
        self._y_test = y_test

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # set updated model weights
        self.model.set_weights(self.weights)

        score = self.model.evaluate(self._x_test, self._y_test, verbose=0)

        logger.info(f"Test loss: {score[0]}")
        logger.info(f"Test accuracy: {score[1]}")

        # set metrics after each evaluation so that the metrics can be logged
        # in a model registry.
        self.metrics = {'loss': score[0], 'accuracy': score[1]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = KerasMnistAggregator(config)
    a.compose()
    a.run()
