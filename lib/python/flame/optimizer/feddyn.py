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

"""EXPERIMENTAL FedDyn optimizer"""
"""https://arxiv.org/abs/2111.04263"""

import logging
from flame.common.constants import TrainState
from diskcache import Cache
from flame.common.typing import ModelWeights
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.optimizer.regularizer.feddyn import FedDynRegularizer
from flame.optimizer.fedavg import FedAvg

logger = logging.getLogger(__name__)


class FedDyn(FedAvg):
    """FedDyn class."""

    def __init__(self, alpha):
        """Initialize FedDyn instance."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for feddyn) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().__init__()

        self.alpha = alpha
        self.local_param_dict = dict()
        self.cld_model = None

        # override parent's self.regularizer
        self.regularizer = FedDynRegularizer(self.alpha)
        logger.debug("Initializing feddyn")

    def save_state(self, state: TrainState, **kwargs):
        if state == TrainState.PRE:
            active_ends = kwargs["active_ends"]

            # adjust history terms to fit active trainers
            new_local_param_dict = dict()
            for end in active_ends:
                if end in self.local_param_dict:
                    new_local_param_dict[end] = self.local_param_dict[end]
                else:
                    # default value for no diff history so far
                    new_local_param_dict[end] = None

            self.local_param_dict = new_local_param_dict

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

        Returns
        -------
        aggregated model: type dict (pytorch)
        """
        logger.debug("calling feddyn")

        assert base_weights is not None

        # reset global weights before aggregation
        self.agg_weights = base_weights

        if len(cache) == 0 or total == 0:
            return None

        # get unweighted mean of selected trainers
        rate = 1 / len(cache)
        for k in list(cache.iterkeys()):
            tres = cache.pop(k)
            self.add_to_hist(k, tres)
            self.aggregate_fn(tres, rate)

        avg_model = self.agg_weights

        # perform unweighted mean on all hist terms
        mean_local_param = {k: 0.0 for k in avg_model}
        rate = 1 / len(self.local_param_dict)
        for end in self.local_param_dict:
            if self.local_param_dict[end] != None:
                h = self.local_param_dict[end]
                mean_local_param = {
                    k: v + rate * h[k] for (k, v) in mean_local_param.items()
                }

        # keep this model as the initial model for the next round of training
        self.cld_model = {k: avg_model[k] + mean_local_param[k] for k in avg_model}

        return avg_model

    def add_to_hist(self, end, tres):
        if end in self.local_param_dict:
            if self.local_param_dict[end] == None:
                self.local_param_dict[end] = tres.weights
            else:
                # aggregate diffs
                self.local_param_dict[end] = {
                    k: v + tres.weights[k]
                    for (k, v) in self.local_param_dict[end].items()
                }
        else:
            # case: end was not previously recorded as active trainer
            logger.debug(f"adding untracked end {end} to hist terms")
            self.local_param_dict[end] = tres.weights
