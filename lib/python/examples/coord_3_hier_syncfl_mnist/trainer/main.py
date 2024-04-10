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
"""HIRE_MNIST horizontal hierarchical FL trainer for Keras."""

import logging
from random import randrange
from statistics import mean

import numpy as np
from flame.config import Config
from flame.mode.horizontal.lifl_coord_syncfl.trainer import Trainer
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class KerasMnistTrainer(Trainer):
    """Keras Mnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0

        self.num_classes = 10
        self.input_shape = (28, 28, 1)

        self.model = None
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 128

    def initialize(self) -> None:
        """Initialize role."""
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model

    def load_data(self) -> None:
        """Load data."""
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        split_n = 10
        index = randrange(split_n)
        # reduce train sample size to reduce the runtime
        x_train = np.split(x_train, split_n)[index]
        y_train = np.split(y_train, split_n)[index]
        x_test = np.split(x_test, split_n)[index]
        y_test = np.split(y_test, split_n)[index]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def train(self) -> None:
        """Train a model."""
        history = self.model.fit(
            self._x_train,
            self._y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
        )

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self._x_train)

        loss = mean(history.history["loss"])
        accuracy = mean(history.history["accuracy"])
        self.update_metrics({"loss": loss, "accuracy": accuracy})

    def evaluate(self) -> None:
        """Evaluate a model."""
        score = self.model.evaluate(self._x_test, self._y_test, verbose=0)

        logger.info(f"Test loss: {score[0]}")
        logger.info(f"Test accuracy: {score[1]}")

        # update metrics after each evaluation so that the metrics can be
        # logged in a model registry.
        self.update_metrics({"test-loss": score[0], "test-accuracy": score[1]})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = KerasMnistTrainer(config)
    t.compose()
    t.run()
