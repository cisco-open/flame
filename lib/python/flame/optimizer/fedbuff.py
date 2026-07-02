# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
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

import numpy as np
from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import MLFramework, get_ml_framework_in_use, valid_frameworks
from .abstract import AbstractOptimizer
from .regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedBuff(AbstractOptimizer):
    """FedBuff class."""

    def __init__(self, **kwargs):
        """Initialize FedBuff instance."""
        super().__init__(**kwargs)

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
                f"supported frameworks are: {valid_frameworks}"
            )

        self.regularizer = Regularizer()

        # Set learning rate differently if asyncOORT selector is used
        try:
            self.use_oort_lr = kwargs["use_oort_lr"]
        except KeyError:
            raise KeyError("Not specified wether to use oort lr or not in config")

        # Set learning rate differently for dataset used Current
        # options: {"cifar-10", "google-speech"}
        try:
            self.dataset_name = kwargs["dataset_name"]
        except KeyError:
            raise KeyError("Dataset name not specified in the config")

        # Explicit learning rate, independent of dataset_name. When set,
        # this wins over the dataset_name lookup table below -- lets each
        # baseline declare its own rate instead of relying on fedbuff's
        # hardcoded cifar-10/google-speech table (which has no entry for
        # other examples, e.g. fwdllm/fluxtune).
        self.learning_rate = kwargs.get("learning_rate", None)

        # Set aggregation rate type between old (just staleness) and
        # new (tradeoff staleness and stat utility) Current options:
        # {"old", "new"}
        try:
            self.agg_rate_conf = kwargs["agg_rate_conf"]
        except KeyError:
            raise KeyError("Aggregation rate type not specified in the config")

    # #### FUNCTIONS TO TRADE-OFF STALENESS WITH STAT_UTILITY
    def alpha_polynomial(self, staleness, a_exp):
        return 1 / ((1 + staleness) ** a_exp)

    def alpha_exponential(self, staleness, a_exp):
        return np.exp(-a_exp * staleness)

    def beta_polynomial(self, loss, b_exp):
        return 1 - (1 / ((1 + loss) ** b_exp))

    def beta_polynomial_upshift(self, loss, b_exp):
        return 1 - (1 / ((1 + loss) ** b_exp)) + 0.5

    def beta_exponential(self, loss, b_exp):
        return 1 - np.exp(-b_exp * loss)

    def beta_exponential_custom(self, loss, b_exp):
        decay_constant = 500 / math.log(2)  # Adjusting the decay constant
        return math.exp(-loss / decay_constant)

    def weight_factor(
        self,
        scale,
        staleness,
        a_exp,
        loss,
        b_exp,
        alpha_type="polynomial",
        beta_type="polynomial",
    ):
        if alpha_type == "polynomial":
            alpha = self.alpha_polynomial(staleness, a_exp)
        elif alpha_type == "exponential":
            alpha = self.alpha_exponential(staleness, a_exp)
        else:
            raise ValueError("Invalid alpha type")

        if beta_type == "polynomial":
            beta = self.beta_polynomial(loss, b_exp)
        elif beta_type == "exponential":
            beta = self.beta_exponential(loss, b_exp)
        elif beta_type == "polynomial_upshift":
            beta = self.beta_polynomial_upshift(loss, b_exp)
        elif beta_type == "exponential_custom":
            beta = self.beta_exponential_custom(loss, b_exp)
        else:
            raise ValueError("Invalid beta type")

        # weight_factor range is [0, 1]
        return ((scale) * alpha) + ((1 - scale) * beta)

    def do(
        self,
        agg_goal_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        staleness_factor: float = 0.0,
        **kwargs,
    ) -> ModelWeights:
        """Do aggregates models of trainers.

        Parameters
        ----------
        agg_goal_weights: delta weights aggregated until agg goal
        cache: a container that includes a list of weights for
        aggregation total: a number of data samples used to train
        weights in cache version: a version number of base weights

        Returns
        -------
        aggregated model: type is either list (tensorflow) or dict
        (pytorch)
        """
        logger.debug("calling fedbuff")

        self.agg_goal_weights = agg_goal_weights
        self.is_agg_weights_none = self.agg_goal_weights is None

        if len(cache) == 0 or total == 0:
            return None

        for k in list(cache.iterkeys()):
            # after popping, the item is removed from the cache hence,
            # explicit cache cleanup is not needed
            tres = cache.pop(k)

            # rate determined based on the staleness of local model
            if self.agg_rate_conf["type"] == "old":
                rate = 1 / math.sqrt(1 + version - tres.version)

            elif self.agg_rate_conf["type"] == "new":
                # New rate that trades off staleness and statistical
                # utility

                # agg_rate_conf will be a dict with keys: {type,
                # scale, a_exp, b_exp}
                scale_val = self.agg_rate_conf["scale"]
                a_exp_val = self.agg_rate_conf["a_exp"]
                b_exp_val = self.agg_rate_conf["b_exp"]

                rate = self.weight_factor(
                    scale=scale_val,
                    staleness=(version - tres.version),
                    a_exp=a_exp_val,
                    loss=tres.stat_utility,
                    b_exp=b_exp_val,
                    alpha_type="polynomial",
                    beta_type="polynomial_upshift",
                )

            logger.info(
                f"agg ver: {version}, trainer ver: {tres.version}, "
                f"trainer stat_utility: {tres.stat_utility}, rate: {rate}, "
                f"with agg_rate_type: {self.agg_rate_conf}"
            )
            self.aggregate_fn(tres, rate)

        return self.agg_goal_weights

    def scale_add_agg_weights(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        """Scale aggregated weights and add it to the original
        weights, when aggregation goal is achieved.

        Parameters
        ----------
        base_weights: original weights of the aggregator
        agg_goal_weights: weights to be scaled and added agg_goal:
        aggregation goal of FedBuff algorithm.

        Returns
        -------
        updated weights
        """
        return self.scale_add_fn(base_weights, agg_goal_weights, agg_goal)

    def _scale_add_agg_weights_pytorch(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        logger.debug(f"base_weights.keys(): {base_weights.keys()}")

        for k in base_weights.keys():
            # agg_goal_weights are already adjusted with rate Using
            # hardcoded learning_rate for now, will pass as an
            # argument later TODO: (DG) Hyper-parameters for AsyncOORT
            # need tuning? Which all hyper-parameters apart from LR
            # need to be tuned?
            if self.learning_rate is not None:
                learning_rate = self.learning_rate
            elif self.use_oort_lr == "False":
                # for fedbuff asyncfl
                if self.dataset_name == "cifar-10":
                    learning_rate = 40.9  # Used with CIFAR-10
                elif self.dataset_name == "google-speech":
                    learning_rate = 0.075  # Used with Google speech
                else:
                    learning_rate = 1.0
                    logger.warning(
                        f"Dataset not specified. using default learning "
                        f"rate of {learning_rate} "
                        f"for FedBuff optimizer"
                    )
                logger.debug(
                    f"Dataset was {self.dataset_name}. using learning "
                    f"rate of {learning_rate} "
                    f"for FedBuff optimizer"
                )
            elif self.use_oort_lr == "True":
                # for asyncOORT asyncfl
                if self.dataset_name == "cifar-10":
                    learning_rate = 0.3  # Used with CIFAR-10
                elif self.dataset_name == "google-speech":
                    learning_rate = 0.065  # Used with Google speech
                else:
                    learning_rate = 1.0
                    logger.warning(
                        f"Dataset not specified. using default learning "
                        f"rate of {learning_rate} "
                        f"for FedBuff optimizer"
                    )
                # logger.debug(f"Dataset was {self.dataset_name}. using learning "
                #              f"rate of {learning_rate} "
                #              f"for FedBuff optimizer")
            base_weights[k] = (base_weights[k]) + (
                learning_rate * ((agg_goal_weights[k] / agg_goal))
            )
        return base_weights

    def _scale_add_agg_weights_tensorflow(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        learning_rate = self.learning_rate if self.learning_rate is not None else 1.0
        for idx in range(len(base_weights)):
            base_weights[idx] += learning_rate * (agg_goal_weights[idx] / agg_goal)
        return base_weights

    def _aggregate_pytorch(self, tres, rate):
        logger.debug("calling _aggregate_pytorch")

        if self.is_agg_weights_none:
            self.agg_goal_weights = {}

        for k, v in tres.weights.items():
            tmp = v * rate
            # tmp.dtype is always float32 or double as rate is float
            # if v.dtype is integer (int32 or int64), there is type
            # mismatch this leads to the following error when
            #   self.agg_weights[k] += tmp: RuntimeError: result type
            #   Float can't be cast to the desired output type Long To
            # handle this issue, we typecast tmp to the original type
            # of v
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
