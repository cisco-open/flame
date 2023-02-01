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
"""FedOPT optimizer.

https://arxiv.org/abs/2003.00295"""
import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import Union

from diskcache import Cache

from ..common.util import (MLFramework, get_ml_framework_in_use,
                           valid_frameworks)
from .fedavg import FedAvg

logger = logging.getLogger(__name__)


class FedOPT(FedAvg):
    """FedOPT class."""

    def __init__(self, beta_1, beta_2, eta, tau):
        """Initialize FedOPT instance."""
        super().__init__()
        self.current_weights = None
        self.d_t = None
        self.m_t = None
        self.v_t = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eta = eta
        self.tau = tau

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.adapt_fn = self._adapt_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.adapt_fn = self._adapt_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")

    def do(self,
           cache: Cache,
           *,
           base_weights=None,
           total: int = 0,
           version: int = 0) -> Union[list, dict]:
        """Do aggregates models of trainers.

        Return: aggregated model
        """
        logger.debug("calling fedopt")

        self.agg_weights = super().do(cache,
                                      base_weights=base_weights,
                                      total=total,
                                      version=version)
        if self.agg_weights is None:
            return self.current_weights

        if self.current_weights is None:
            self.current_weights = self.agg_weights
        else:
            self.adapt_fn(self.agg_weights, self.current_weights)

        return self.current_weights

    @abstractmethod
    def _delta_v_pytorch(self):
        return

    @abstractmethod
    def _delta_v_tensorflow(self):
        return

    def _adapt_pytorch(self, average, current):
        import torch
        logger.debug("calling _adapt_pytorch")

        self.d_t = {k: average[k] - current[k] for k in average.keys()}

        if self.m_t is None:
            self.m_t = {
                k: torch.zeros_like(self.d_t[k])
                for k in self.d_t.keys()
            }
        self.m_t = {
            k: self.beta_1 * self.m_t[k] + (1 - self.beta_1) * self.d_t[k]
            for k in self.m_t.keys()
        }

        if self.v_t is None:
            self.v_t = {
                k: torch.zeros_like(self.d_t[k])
                for k in self.d_t.keys()
            }
        self._delta_v_pytorch()

        self.current_weights = OrderedDict({
            k: self.current_weights[k] + self.eta * self.m_t[k] /
            (torch.sqrt(self.v_t[k]) + self.tau)
            for k in self.current_weights.keys()
        })

    def _adapt_tensorflow(self, average, current):
        import tensorflow as tf
        logger.debug("calling _adapt_tensorflow")

        self.d_t = [average[idx] - current[idx] for idx in range(len(average))]

        if self.m_t is None:
            self.m_t = [
                tf.zeros_like(self.d_t[idx]) for idx in range(len(self.d_t))
            ]
        self.m_t = [
            self.beta_1 * self.m_t[idx] + (1 - self.beta_1) * self.d_t[idx]
            for idx in range(len(self.m_t))
        ]

        if self.v_t is None:
            self.v_t = [
                tf.zeros_like(self.d_t[idx]) for idx in range(len(self.d_t))
            ]
        self._delta_v_tensorflow()

        self.current_weights = [
            self.current_weights[idx] + self.eta * self.m_t[idx] /
            (tf.sqrt(self.v_t[idx]) + self.tau)
            for idx in range(len(self.current_weights))
        ]
