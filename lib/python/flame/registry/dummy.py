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
"""Dummmy registry client."""

from typing import Any, Optional

from flame.config import Hyperparameters, Config
from flame.registry.abstract import AbstractRegistryClient


class DummyRegistryClient(AbstractRegistryClient):
    """Dummy registry client."""

    def __call__(self, config: Config) -> None:
        """Initialize the instance."""
        pass

    def setup_run(self) -> None:
        """Set up a run."""
        pass

    def save_metrics(self, epoch: int, metrics: Optional[dict[str, float]]) -> None:
        """Save metrics in a model registry."""
        pass

    def save_params(self, hyperparameters: Optional[Hyperparameters]) -> None:
        """Save hyperparameters in a model registry."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def save_model(self, name: str, model: Any) -> None:
        """Save a model in a model registry."""
        pass

    def load_model(self, name: str, version: int) -> object:
        """Load a model from a model registry."""
        pass
