# Copyright 2024 Cisco Systems, Inc. and its affiliates
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
"""honrizontal FL middle level aggregator (EAGER aggregation)."""

import logging
import time
from copy import deepcopy

from flame.mode.horizontal.syncfl.middle_aggregator import (
    MiddleAggregator as BaseMiddleAggregator,
)
from flame.mode.message import MessageType
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)


class MiddleAggregator(BaseMiddleAggregator):
    """Middle level aggregator using eager aggregation."""

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        base_weights = deepcopy(self.weights)

        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, _ = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if MessageType.WEIGHTS in msg:
                weights = msg[MessageType.WEIGHTS]

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres

            # optimizer conducts optimization (in this case, aggregation)
            global_weights = self.optimizer.do(
               base_weights, self.cache, total=total
            )
            if global_weights is None:
                logger.debug("failed model aggregation")
                time.sleep(1)
                return

        # save global weights before updating it
        self.prev_weights = self.weights

        # set global weights
        self.weights = global_weights

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        self.dataset_size = total
