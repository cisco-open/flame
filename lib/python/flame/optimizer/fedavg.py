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
"""Federated Averaging optimizer."""
import logging

from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import (MLFramework, get_ml_framework_in_use,
                           valid_frameworks)
from .abstract import AbstractOptimizer
from .regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedAvg(AbstractOptimizer):
    """FedAvg class."""

    def __init__(self):
        """Initialize FedAvg instance."""
        self.agg_weights = None

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.aggregate_fn = self._aggregate_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")
        
        self.regularizer = Regularizer()

    def do(self,
           base_weights: ModelWeights,
           cache: Cache,
           *,
           total: int = 0,
           version: int = 0) -> ModelWeights:
        """Do aggregates models of trainers.

        Parameters
        ----------
        base_weights: weights to be used as base
        cache: a container that includes a list of weights for aggregation
        total: a number of data samples used to train weights in cache
        version: a version number of base weights

        Returns
        -------
        aggregated model: type is either list (tensorflow) or dict (pytorch)
        """
        logger.debug("calling fedavg")

        assert (base_weights is not None)

        # reset global weights before aggregation
        self.agg_weights = base_weights

        if len(cache) == 0 or total == 0:
            return None

        for k in list(cache.iterkeys()):
            # after popping, the item is removed from the cache
            # hence, explicit cache cleanup is not needed
            tres = cache.pop(k)

            rate = tres.count / total
            self.aggregate_fn(tres, rate)

        return self.agg_weights

    def _aggregate_pytorch(self, tres, rate):
        logger.debug("calling _aggregate_pytorch")

        for k, v in tres.weights.items():
            tmp = v * rate
            # tmp.dtype is always float32 or double as rate is float
            # if v.dtype is integer (int32 or int64), there is type mismatch
            # this leads to the following error when self.agg_weights[k] += tmp:
            #   RuntimeError: result type Float can't be cast to the desired
            #   output type Long
            # To handle this issue, we typecast tmp to the original type of v
            #
            # TODO: this may need to be revisited
            tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp

            self.agg_weights[k] += tmp

    def _aggregate_tensorflow(self, tres, rate):
        logger.debug("calling _aggregate_tensorflow")

        for idx in range(len(tres.weights)):
            self.agg_weights[idx] += tres.weights[idx] * rate
