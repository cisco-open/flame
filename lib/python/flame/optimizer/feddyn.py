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
from diskcache import Cache
from ..common.typing import ModelWeights
from ..common.util import (MLFramework, get_ml_framework_in_use)
from .regularizer.feddyn import FedDynRegularizer
from .fedavg import FedAvg

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
                f"supported frameworks (for feddyn) are: {[MLFramework.PYTORCH.name.lower()]}")
        
        super().__init__()
        
        self.alpha = alpha
        self.h_t = None
        
        # override parent's self.regularizer
        self.regularizer = FedDynRegularizer(self.alpha)
        logger.debug("Initializing feddyn")
    
    def do(self,
           base_weights: ModelWeights,
           cache: Cache,
           *,
           total: int = 0,
           version: int = 0,
           **kwargs) -> ModelWeights:
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
        
        num_trainers = kwargs['num_trainers']
        num_selected = len(cache)
        
        # populate h_t
        if self.h_t is None:
            self.h_t = dict()
            for k in base_weights:
                self.h_t[k] = 0.0

        self.agg_weights = super().do(base_weights,
                                      cache,
                                      total=total,
                                      version=version)
        
        self.adapt_fn(self.agg_weights, base_weights, num_trainers, num_selected)
        
        return self.current_weights
    
    def adapt_fn(self, average, base, num_trainers, num_selected):
        
        # get delta from averaging which we use as (1/|P_t|) * sum_{k in P_t}[theta^t_k - theta^{t-1}]
        self.d_t = {k: average[k] - base[k] for k in average.keys()}
        
        # (num_selected / num_trainers) = (|P_t| / m) 
        # this acts as a conversion factor for d_t to be averaged among all active trainers
        d_mult = self.alpha * (num_selected / num_trainers)
        self.h_t = {k:self.h_t[k] - d_mult * self.d_t[k] for k in self.h_t.keys()}
        
        # here h_t needs to be multiplied by (1/alpha) before it is subtracted from the averaged model
        # although the averaged model is weighted by dataset, we take this to be the same as (1/|P_t|) * sum_{k in P_t}[theta^t_k]
        h_mult = 1.0 / self.alpha
        self.current_weights = {k:average[k] - h_mult * self.h_t[k] for k in self.h_t.keys()}
