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


"""HIRE_MNIST horizontal hierarchical FL middle level aggregator for Keras."""

import logging

from flame.config import Config
from flame.mode.horizontal.middle_aggregator import MiddleAggregator
from tensorflow import keras

logger = logging.getLogger(__name__)

class KerasMnistMiddleAggregator(MiddleAggregator):
    """Keras Mnist Middle Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.weights = None
        self.dataset_size = 0

    def initialize(self):
        """Initialize role."""
        pass

    def load_data(self) -> None:
        """Load a test dataset."""
        pass

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help='config file', required=True)
    args = parser.parse_args()
    config = Config(args.config)

    a = KerasMnistMiddleAggregator(config)
    a.compose()
    a.run()