# Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
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

"""Federated Averaging optimizer."""
import logging

from diskcache import Cache

from .abstract import AbstractOptimizer

logger = logging.getLogger(__name__)


class FedAvg(AbstractOptimizer):
    """FedAvg class."""

    def do(self, cache: Cache, total: int):
        """Do aggregates models of trainers.

        Return: aggregated model
        """
        logger.debug("calling fedavg")

        if len(cache) == 0 or total == 0:
            return None

        global_weights = None
        for k in list(cache.iterkeys()):
            # after popping, the item is removed from the cache
            # hence, explicit cache cleanup is not needed
            tres = cache.pop(k)

            if global_weights is None:
                global_weights = [0] * len(tres.weights)

            rate = tres.count / total
            for idx in range(len(tres.weights)):
                global_weights[idx] += tres.weights[idx] * rate

        return global_weights
