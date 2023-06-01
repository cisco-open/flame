# Copyright 2023 Cisco Systems, Inc. and its affiliates
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
"""Local registry client."""

from typing import Any, Optional
import logging
import os
import csv
import pickle
from collections import defaultdict
from pathlib import Path
import shutil
import time

from flame.common.util import get_ml_framework_in_use, MLFramework
from flame.config import Hyperparameters
from flame.config import Config
from flame.registry.abstract import AbstractRegistryClient

logger = logging.getLogger(__name__)

# specify folder names
METRICS = "metrics"
MODEL = "model"
PARAMS = "params"
FLAME_LOG = "flame-log"


class LocalRegistryClient(AbstractRegistryClient):
    """Local registry client."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            logger.info("Create a local registry client instance")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, config: Config) -> None:
        """Initialize the instance."""
        self.job_id = config.job.job_id
        self.task_id = config.task_id

    def setup_run(self) -> None:
        """Set up a run."""
        # set up directories
        home_dir = Path.home()
        log_dir = os.path.join(home_dir, FLAME_LOG)
        self.registry_path = os.path.join(log_dir, self.job_id, self.task_id)

        if os.path.exists(self.registry_path):
            shutil.rmtree(self.registry_path)

        for directory in [METRICS, MODEL, PARAMS]:
            os.makedirs(os.path.join(self.registry_path, directory))

        # version tracking
        self.param_version = 1
        self.model_versions = defaultdict(lambda: 1)

    def save_metrics(self, epoch: int, metrics: Optional[dict[str, float]]) -> None:
        """Save metrics in a model registry."""
        curr_time = time.time()
        for metric in metrics:
            filename = os.path.join(self.registry_path, METRICS, metric)
            exists = os.path.exists(filename)
            with open(filename, "a+") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["round", "time", metric]) if not exists else None
                csv_writer.writerow([epoch, curr_time, metrics[metric]])

    def save_params(self, hyperparameters: Optional[Hyperparameters]) -> None:
        """Save hyperparameters in a model registry."""
        with open(
            os.path.join(self.registry_path, PARAMS, str(self.param_version)), "wb"
        ) as file:
            pickle.dump(hyperparameters.dict(), file)
        self.param_version += 1

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def save_model(self, name: str, model: Any) -> None:
        """Save a model in a model registry."""
        model_folder = os.path.join(self.registry_path, MODEL, name)
        os.makedirs(model_folder, exist_ok=True)

        ml_framework = get_ml_framework_in_use()

        if ml_framework == MLFramework.PYTORCH:
            import torch

            torch.save(
                model, os.path.join(model_folder, str(self.model_versions[name]))
            )
        elif ml_framework == MLFramework.TENSORFLOW:
            model.save(os.path.join(model_folder, str(self.model_versions[name])))
        self.model_versions[name] += 1

    def load_model(self, name: str, version: int) -> object:
        """
        Load a model.

        This method can be called without calling self.setup_run().
        """
        ml_framework = get_ml_framework_in_use()
        model_path = os.path.join(self.registry_path, MODEL, name, str(version))
        if ml_framework == MLFramework.PYTORCH:
            import torch

            # the class definition for the model must be available for this
            return torch.load(model_path)
        elif ml_framework == MLFramework.TENSORFLOW:
            import tensorflow

            return tensorflow.keras.models.load_model(model_path)

        raise ModuleNotFoundError("Module for loading model not found")
