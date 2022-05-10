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

"""Hybrid aggregator for Keras."""

import logging

from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

class KerasMnistAggregator(TopAggregator):
    """Keras Mnist Top Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None

        self.dataset: Dataset = None

        self.num_classes = 10
        self.input_shape = (28, 28, 1)

    def initialize(self):
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
        """Load a test dataset."""
        # Implement this if loading data is needed in aggregator
        pass

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Implement this if testing is needed in aggregator
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = KerasMnistAggregator(config)
    a.compose()
    a.run()
