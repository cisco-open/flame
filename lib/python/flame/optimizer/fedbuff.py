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
"""FedBuff optimizer.

The implementation is based on the following paper:
https://arxiv.org/pdf/2106.06639.pdf
https://arxiv.org/pdf/2111.04877.pdf

SecAgg algorithm is not the scope of this implementation.
"""
import logging
import math

from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import (MLFramework, get_ml_framework_in_use,
                           valid_frameworks)
from .abstract import AbstractOptimizer
from .regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedBuff(AbstractOptimizer):
    """FedBuff class."""

    def __init__(self):
        """Initialize FedBuff instance."""
        self.agg_goal_weights = None

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
            self.scale_add_fn = self._scale_add_agg_weights_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.aggregate_fn = self._aggregate_tensorflow
            self.scale_add_fn = self._scale_add_agg_weights_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")

        self.regularizer = Regularizer()

    def do(
        self,
        agg_goal_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        **kwargs,
    ) -> ModelWeights:
        """Do aggregates models of trainers.

        Parameters
        ----------
        agg_goal_weights: delta weights aggregated until agg goal
        cache: a container that includes a list of weights for aggregation
        total: a number of data samples used to train weights in cache
        version: a version number of base weights

        Returns
        -------
        aggregated model: type is either list (tensorflow) or dict (pytorch)
        """
        logger.debug("calling fedbuff")

        self.agg_goal_weights = agg_goal_weights
        self.is_agg_weights_none = self.agg_goal_weights is None

        if len(cache) == 0 or total == 0:
            return None

        for k in list(cache.iterkeys()):
            # after popping, the item is removed from the cache
            # hence, explicit cache cleanup is not needed
            tres = cache.pop(k)

            logger.debug(f"agg ver: {version}, trainer ver: {tres.version}")
            # rate determined based on the staleness of local model
            rate = 1 / math.sqrt(1 + version - tres.version)
            self.aggregate_fn(tres, rate)

        return self.agg_goal_weights

    def scale_add_agg_weights(
        self,
        base_weights: ModelWeights,
        agg_goal_weights: ModelWeights,
        agg_goal: int,
    ) -> ModelWeights:
        """Scale aggregated weights and add it to the original weights,
        when aggregation goal is achieved.

        Parameters
        ----------
        base_weights: original weights of the aggregator
        agg_goal_weights: weights to be scaled and added
        agg_goal: aggregation goal of FedBuff algorithm.

        Returns
        -------
        updated weights
        """
        return self.scale_add_fn(base_weights, agg_goal_weights, agg_goal)

    def _scale_add_agg_weights_pytorch(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        for k in base_weights.keys():
            base_weights[k] += agg_goal_weights[k] / agg_goal
        return base_weights

    def _scale_add_agg_weights_tensorflow(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        for idx in range(len(base_weights)):
            base_weights[idx] += agg_goal_weights[idx] / agg_goal
        return base_weights

    def _aggregate_pytorch(self, tres, rate):
        logger.debug("calling _aggregate_pytorch")

        if self.is_agg_weights_none:
            self.agg_goal_weights = {}

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

            if self.is_agg_weights_none:
                self.agg_goal_weights[k] = tmp
            else:
                self.agg_goal_weights[k] += tmp

    def _aggregate_tensorflow(self, tres, rate):
        logger.debug("calling _aggregate_tensorflow")

        if self.is_agg_weights_none:
            self.agg_goal_weights = []

        for idx in range(len(tres.weights)):
            if self.is_agg_weights_none:
                self.agg_goal_weights.append(tres.weights[idx] * rate)
            else:
                self.agg_goal_weights[idx] += tres.weights[idx] * rate
