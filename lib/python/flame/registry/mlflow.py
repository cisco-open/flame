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
"""MLflow registry client."""

import logging
from sys import modules
from typing import Any, Optional

import mlflow

from .abstract import AbstractRegistryClient

logger = logging.getLogger(__name__)

# The following flavors are integrated with flame.
module_to_flavor: dict[str, str] = {
    "keras": "keras",
    "sklearn": "sklearn",
    "torch": "pytorch"
}


class MLflowRegistryClient(AbstractRegistryClient):
    """MLflow registry client."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            logger.info('Create an MLflow registry client instance')
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, uri: str, experiment: str) -> None:
        """Initialize the instance."""
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

        self.experiment = experiment

    def setup_run(self):
        """
        Set up a run for logging parameters, metrics and model.

        As start_run() is called without ``with'' block, the self.cleanup()
        method (containing mlflow.end_run()) must be called at the end.
        """
        mlflow.start_run(run_name=self.experiment)

    def save_metrics(
        self, epoch: int, metrics: Optional[dict[str, float]]
    ) -> None:
        """Save metrics in a model registry."""
        if not metrics:
            return

        mlflow.log_metrics(metrics, step=epoch)

    def save_params(self, hyperparameters: Optional[dict[str, float]]) -> None:
        """Save hyperparameters in a model registry."""
        if not hyperparameters:
            return

        mlflow.log_params(hyperparameters)

    def cleanup(self) -> None:
        """
        Clean up resources.

        This method must be called at the end if self.setup_run() is called.
        """
        mlflow.end_run()

    def save_model(self, name: str, model: Any) -> None:
        """Save a model in a model registry."""
        flavor = self._get_ml_framework_flavor()

        flavor.log_model(
            model, artifact_path="models", registered_model_name=name
        )

    def load_model(self, name: str, version: int) -> object:
        """
        Load a model.

        This method can be called without calling self.setup_run().
        """
        flavor = self._get_ml_framework_flavor()

        return flavor.load_model(model_uri=f"models:/{name}/{version}")

    def _get_ml_framework_flavor(self) -> Any:
        """Return a ml framework flavor based on the loaded module."""
        for module, flavor in module_to_flavor.items():
            if module not in modules:
                continue

            return getattr(mlflow, flavor)

        raise ValueError("machine learning framework not found")
