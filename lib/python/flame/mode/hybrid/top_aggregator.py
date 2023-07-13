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
"""Hybrid FL top level aggregator."""

import logging
import time
from datetime import datetime
from copy import deepcopy
from typing import Any

from flame.common.util import weights_to_device, weights_to_model_device
from flame.common.constants import DeviceType
from flame.optimizer.train_result import TrainResult
from flame.mode.message import MessageType
from flame.mode.horizontal.top_aggregator import TopAggregator as BaseTopAggregator
from flame.mode.horizontal.syncfl.top_aggregator import (
    PROP_ROUND_START_TIME,
    PROP_ROUND_END_TIME,
)

logger = logging.getLogger(__name__)

GROUP_UNIDENTIFIED = "group_unidentified"


class TopAggregator(BaseTopAggregator):
    """Hybrid Top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.end_send_to = {}
        self.end_receive_from = {}
        self.end_idx = {}
        self.end_is_committer = {}
        self.group_ends = {}
        self.ends_group = {}

        super().internal_init()

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        self.group_ends = {}
        self.ends_group = {}

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA],
                    end,
                    channel,
                )

            if MessageType.HYBRID_METADATA in msg:
                group_id = msg[MessageType.HYBRID_METADATA]["group_id"]
                self.ends_group[end] = group_id
                if group_id not in self.group_ends:
                    self.group_ends[group_id] = [end]
                else:
                    self.group_ends[group_id].append(end)

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights),
            self.cache,
            total=total,
            num_trainers=len(channel.ends()),
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
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        selected_ends = channel.ends()
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # find receive_from, send_to, total_count, and is_committer info on all ends
        self.parse_hybrid_aggregation_info()

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending weights to {end}")

            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                    MessageType.HYBRID_METADATA: self.get_hybrid_metadata(end),
                    MessageType.IS_COMMITTER: self.end_is_committer.get(end),
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def parse_hybrid_aggregation_info(self) -> None:
        self.end_send_to = {}
        self.end_receive_from = {}
        self.end_idx = {}
        self.end_is_committer = {}

        for group, ends in self.group_ends.items():
            # Group aggregation not required for one or less trainers
            if len(ends) > 1:
                for e_idx, end in enumerate(ends):
                    self.end_send_to[end] = ends[(e_idx + 1) % len(ends)]
                    self.end_receive_from[end] = ends[(e_idx - 1) % len(ends)]
                    self.end_idx[end] = e_idx
                    self.end_is_committer[end] = (
                        True if e_idx == len(ends) - 1 else False
                    )

    def get_hybrid_metadata(self, end) -> dict[str, Any]:
        group_ends = self.group_ends.get(self.ends_group.get(end))
        group_size = len(group_ends) if type(group_ends) == list else None

        return {
            "group_id": self.ends_group.get(end, GROUP_UNIDENTIFIED),
            "sendto_id": self.end_send_to.get(end),
            "recvfrom_id": self.end_receive_from.get(end),
            "rank": self.end_idx.get(end),
            "size": group_size,
        }
