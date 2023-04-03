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

import logging
import time
from copy import deepcopy

from flame.mode.composer import Composer
from flame.mode.horizontal.middle_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_FETCH,
    TAG_UPLOAD,
    WAIT_TIME_FOR_TRAINER,
)
from flame.mode.horizontal.middle_aggregator import (
    MiddleAggregator as BaseMiddleAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)

TAG_GET_TRAINERS = "getTrainers"


class MiddleAggregator(BaseMiddleAggregator):
    def _get_trainers(self, tag) -> None:
        logger.debug("getting trainers from coordinator")

        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.COORDINATED_ENDS not in msg:
            raise ValueError("no coordinated ends message")

        self.trainers = msg[MessageType.COORDINATED_ENDS]
        logger.debug("received trainers from coordinator")

    def _distribute_weights(self, tag: str) -> None:
        logger.debug("distributing weights to trainers")

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        if len(self.trainers) == 0:
            logger.debug("no trainers found return dummy weights")
            self.trainer_no_show = True
            self._send_dummy_weights(TAG_UPLOAD)
            return

        # this call waits for at least one peer to join this channel
        self.trainer_no_show = channel.await_join(WAIT_TIME_FOR_TRAINER)
        if self.trainer_no_show:
            logger.debug("channel await join timeouted")
            # send dummy weights to unblock top aggregator
            self._send_dummy_weights(TAG_UPLOAD)
            return

        for end in self.trainers:
            logger.debug(f"sending weights to {end}")
            channel.send(
                end, {MessageType.WEIGHTS: self.weights, MessageType.ROUND: self._round}
            )

    def _aggregate_weights(self, tag: str) -> None:
        logger.debug("aggregating weights from trainers")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(self.trainers):
            end, _ = metadata

            logger.debug(f"received a message from trainer {end}")

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
            deepcopy(self.weights), self.cache, total=total
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # save global weights before updating it
        self.prev_weights = self.weights

        # set global weights
        self.weights = global_weights
        self.dataset_size = total

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)
        elif tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)
        elif tag == TAG_GET_TRAINERS:
            self._get_trainers(tag)

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_get_trainers = Tasklet(self.get, TAG_GET_TRAINERS)

            task_put_dist = Tasklet(self.put, TAG_DISTRIBUTE)
            task_put_dist.set_continue_fn(cont_fn=lambda: self.trainer_no_show)

            task_put_upload = Tasklet(self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet(self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet(self.get, TAG_FETCH)

            task_eval = Tasklet(self.evaluate)

            task_update_round = Tasklet(self.update_round)

            task_end_of_training = Tasklet(self.inform_end_of_training)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_trainers
                >> task_get_fetch
                >> task_put_dist
                >> task_get_aggr
                >> task_put_upload
                >> task_eval
                >> task_update_round
            )
            >> task_end_of_training
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_FETCH, TAG_UPLOAD, TAG_GET_TRAINERS]
