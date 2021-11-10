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
"""role abstract class."""

from abc import ABC, abstractmethod


class Role(ABC):
    """Abstract base class for role implementation."""

    ###########################################################################
    # The following functions need to be implemented the child class
    ###########################################################################

    @abstractmethod
    def internal_init(self) -> None:
        """Abstract method to initialize internal state."""

    @abstractmethod
    def get(self, tag: str) -> None:
        """Abstract method to get data (model weights) from remote role(s)."""

    @abstractmethod
    def put(self, tag: str) -> None:
        """Abstract method to put data (model weights) to remote role(s)."""

    @abstractmethod
    def compose(self) -> None:
        """Abstract method to compose role with tasklets."""

    ###########################################################################
    # The following functions need to be implemented the grandchild class.
    ###########################################################################

    @abstractmethod
    def initialize(self) -> None:
        """Abstract method to initialize role."""

    @abstractmethod
    def load_data(self) -> None:
        """Abstract method to load data."""

    @abstractmethod
    def train(self) -> None:
        """Abstract method to train a model."""

    @abstractmethod
    def evaluate(self) -> None:
        """Abstract method to evaluate a model."""
