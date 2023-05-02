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
"""Oort horizontal FL top level aggregator."""

import logging
import time
from datetime import datetime
from copy import deepcopy
from typing import Any, Tuple

from flame.common.util import weights_to_device, weights_to_model_device
from flame.common.constants import DeviceType
from flame.optimizer.train_result import TrainResult
from flame.mode.message import MessageType
from flame.selector.oort import (
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_LAST_SELECTED_ROUND,
)
from ..top_aggregator import TopAggregator as BaseTopAggregator

logger = logging.getLogger(__name__)


class TopAggregator(BaseTopAggregator):
    """Oort Top level Aggregator implements an ML aggregation role."""

    def _aggregate_weights(self, tag: str) -> None:
        """
        Aggregate local model weights, accepting K trainers out of 1.3K clients
        selected from selector. Moreover, trainers' round duration is measured,
        which determine the system utility of trainers for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # receive local model parameters from trainers
        # terminate aggregating when received weights from k ends
        # (k * overcommitment is selected for training with Oort)

        end_ids = channel.ends()
        aggr_num = min(self.config.selector.kwargs["aggr_num"], len(end_ids))

        received_end_count = 0

        for msg, metadata in channel.recv_fifo(end_ids, aggr_num):
            end, _ = metadata

            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if self._round != msg[MessageType.MODEL_VERSION]:
                logger.debug(f"Stale message from {end}; skipping it")
                continue

            total = self._handle_weights_msg(msg, metadata, channel, total)

            # remove end_id if it sends a valid message with correct round info
            # break the for loop if k valid messages arrive
            received_end_count += 1
            end_ids.remove(end)
            if received_end_count == aggr_num:
                break

        # running the second loop to aggregate up to aggr_num updates from trainers
        while received_end_count < aggr_num:
            for msg, metadata in channel.recv_fifo(end_ids, 1):
                end, _ = metadata

                if not msg:
                    logger.debug(f"No data from {end}; skipping it")
                    continue

                if self._round != msg[MessageType.MODEL_VERSION]:
                    logger.debug(f"Stale message from {end}; skipping it")
                    continue

                total = self._handle_weights_msg(msg, metadata, channel, total)

                # remove end_id if it sends a valid message with correct round info
                # break the for loop if k valid messages arrive
                received_end_count += 1
                end_ids.remove(end)
                if received_end_count == aggr_num:
                    break

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights), self.cache, total=total
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = global_weights

        # update model with global weights
        self._update_model()

    def _distribute_weights(self, tag: str) -> None:
        """
        Distribute local model weights to 1.3K clients, where K is the number
        of desired trainers to select. Moreover, measure the start time of
        a round to for round duration measurement on each trainers for
        measuring their system utility that is required for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                },
            )

    def _handle_weights_msg(
        self, msg: Any, metadata: Tuple[str, datetime], channel: Any, total: int
    ) -> int:
        end = metadata[0]
        timestamp = metadata[1]

        logger.debug(f"received data from {end}")

        # calculate round duration for this end, if the round number information
        # is identical with round_start_time
        round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
        if round_start_time_tup[0] == msg[MessageType.MODEL_VERSION]:
            channel.set_end_property(
                end, PROP_ROUND_DURATION, timestamp - round_start_time_tup[1]
            )

        if MessageType.WEIGHTS in msg:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
        if MessageType.MODEL_VERSION in msg:
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            total += count
            tres = TrainResult(weights, count)
            # save training result from trainer in a disk cache
            self.cache[end] = tres

        return total
