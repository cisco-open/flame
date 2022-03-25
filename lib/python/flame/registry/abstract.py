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

"""Abstract registry client."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class AbstractRegistryClient(ABC):
    """MLflow registry client."""

    #
    @abstractmethod
    def __call__(self, uri: str, job_id: str) -> None:
        """Abstract method for initializing a registry client."""

    @abstractmethod
    def setup_run(self):
        """Abstract method for setup a run."""

    @abstractmethod
    def save_metrics(
        self, epoch: int, metrics: Optional[dict[str, float]]
    ) -> None:
        """Abstract method for saving metrics in a model registry."""

    @abstractmethod
    def save_params(self, hyperparameters: Optional[dict[str, float]]) -> None:
        """Abstract method for saving hyperparameters in a model registry."""

    @abstractmethod
    def cleanup(self) -> None:
        """Abstract method for cleanning up resources."""

    @abstractmethod
    def save_model(self, name: str, model: Any) -> None:
        """Abstract method for saving a model in a model registry."""

    @abstractmethod
    def load_model(self, name: str, version: int) -> object:
        """Abstract method for loading a model from a model registry."""
