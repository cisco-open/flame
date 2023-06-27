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

"""SCAFFOLD optimizer"""
"""https://arxiv.org/abs/1910.06378"""

import logging

from diskcache import Cache

from flame.common.constants import TrainState
from flame.common.typing import ModelWeights
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.optimizer.fedavg import FedAvg
from flame.optimizer.regularizer.scaffold import ScaffoldRegularizer

logger = logging.getLogger(__name__)

DATASET_SIZES = "dataset_sizes"
GLOBAL_WEIGHTS = "glob_weights"


class Scaffold(FedAvg):
    """SCAFFOLD class."""

    def __init__(self, k):
        """Initialize SCAFFOLD instance."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for SCAFFOLD) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().__init__()

        # c-term
        self.c_glob = None

        # override parent's self.regularizer
        self.regularizer = ScaffoldRegularizer(k)
        logger.debug("Initializing scaffold")

    def save_state(self, state: TrainState, **kwargs):
        if state == TrainState.PRE:
            if DATASET_SIZES in kwargs:
                dataset_sizes = kwargs[DATASET_SIZES]

                total_samples = sum(dataset_sizes.values())
                num_trainers = len(dataset_sizes)
                self.weight_dict = {
                    end: (dataset_sizes[end] / total_samples) * num_trainers
                    for end in dataset_sizes
                }

            if GLOBAL_WEIGHTS in kwargs and self.c_glob is None:
                import torch

                glob_model_state_dict = kwargs[GLOBAL_WEIGHTS]

                # populate c_glob
                self.c_glob = {
                    k: torch.zeros_like(glob_model_state_dict[k])
                    for k in glob_model_state_dict
                }

    def do(
        self,
        base_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        **kwargs,
    ) -> ModelWeights:
        """Do aggregates models of trainers.

        Parameters
        ----------
        base_weights: weights to be used as base
        cache: a container that includes a list of weights for aggregation
        total: a number of data samples used to train weights in cache
        version: a version number of base weights
        control_cache: a cache containing the new trainer control variates (in kwargs)

        Returns
        -------
        aggregated model: type dict (pytorch)
        """
        logger.debug("calling scaffold")

        assert base_weights is not None

        if len(cache) == 0 or total == 0:
            return None

        control_cache = kwargs["control_cache"]

        # confirm scaffold aggreagtion is possible
        if len(control_cache) != len(cache):
            return None

        # aggregate control variate terms
        self.c_agg_weights = self.c_glob

        n_clnt = len(self.weight_dict)
        rate = 1 / n_clnt
        for k in list(control_cache.iterkeys()):
            tres = control_cache.pop(k)
            self.c_aggregate_fn(tres, rate, k)

        self.c_glob = self.c_agg_weights

        # reset global weights before aggregation
        self.agg_weights = base_weights

        # get unweighted mean of selected trainers
        rate = 1 / len(cache)
        for k in list(cache.iterkeys()):
            tres = cache.pop(k)
            self.aggregate_fn(tres, rate)

        return self.agg_weights

    def c_aggregate_fn(self, tres, rate, end):
        for k, v in tres.weights.items():
            tmp = v * rate
            # type of model weight may have changed on the trainer side due to scaling
            tmp = (
                tmp.to(dtype=self.c_agg_weights[k].dtype)
                if tmp.dtype != self.c_agg_weights[k].dtype
                else tmp
            )

            self.c_agg_weights[k] += tmp
