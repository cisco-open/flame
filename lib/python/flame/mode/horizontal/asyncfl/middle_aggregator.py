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
"""Asynchronous honrizontal FL middle level aggregator."""

import logging
import time
from copy import deepcopy

from ....channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from ....common.util import (MLFramework, delta_weights_pytorch,
                             delta_weights_tensorflow, get_ml_framework_in_use,
                             valid_frameworks)
from ....optimizer.train_result import TrainResult
from ...composer import Composer
from ...message import MessageType
from ...tasklet import Loop, Tasklet
from ..middle_aggregator import (TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_FETCH,
                                 TAG_UPLOAD)
from ..middle_aggregator import MiddleAggregator as SyncMidAgg

logger = logging.getLogger(__name__)

# 60 second wait time until a trainer appears in a channel
WAIT_TIME_FOR_TRAINER = 60


class MiddleAggregator(SyncMidAgg):
    """Asynchronous middle level aggregator.

    It acts as a proxy between top level aggregator and trainer.
    """

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = 0
        if 'aggGoal' in self.config.hyperparameters:
            self._agg_goal = self.config.hyperparameters['aggGoal']

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")

        if self.framework == MLFramework.PYTORCH:
            self._delta_weights_fn = delta_weights_pytorch

        elif self.framework == MLFramework.TENSORFLOW:
            self._delta_weights_fn = delta_weights_tensorflow

    def _reset_agg_goal_variables(self):
        logger.debug("reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg = channel.recv(end)

        if MessageType.WEIGHTS in msg:
            self.weights = msg[MessageType.WEIGHTS]

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        self.trainer_no_show = channel.await_join(WAIT_TIME_FOR_TRAINER)
        if self.trainer_no_show:
            logger.debug("channel await join timeouted")
            # send dummy weights to unblock top aggregator
            self._send_dummy_weights(TAG_UPLOAD)
            return

        for end in channel.ends(VAL_CH_STATE_SEND):
            logger.debug(f"sending weights to {end}")
            channel.send(
                end, {
                    MessageType.WEIGHTS: self.weights,
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round
                })

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous middle aggregator
        (..middle_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        if self._agg_goal_weights is None:
            logger.debug(f"type of weights: {type(self.weights)}")
            self._agg_goal_weights = deepcopy(self.weights)

        # receive local model parameters from a trainer who arrives first
        end, msg = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received data from {end}")

        if MessageType.WEIGHTS in msg:
            weights = msg[MessageType.WEIGHTS]

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            tres = TrainResult(weights, count, version)
            # save training result from trainer in a disk cache
            self.cache[end] = tres

            self._agg_goal_weights = self.optimizer.do(self._agg_goal_weights,
                                                       self.cache,
                                                       total=count,
                                                       version=self._round)
            # increment agg goal count
            self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            # didn't reach the aggregation goal; return
            logger.debug("didn't reach agg goal")
            logger.debug(
                f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            return

        if self._agg_goal_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # save global weights before updating it
        self.prev_weights = self.weights

        # set global weights
        self.weights = self._agg_goal_weights

        self.dataset_size = count

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        channel.send(
            end, {
                MessageType.WEIGHTS: delta_weights,
                MessageType.DATASET_SIZE: self.dataset_size,
                MessageType.MODEL_VERSION: self._round
            })
        logger.debug("sending weights done")

    def _send_dummy_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        dummy_msg = {
            MessageType.WEIGHTS: None,
            MessageType.DATASET_SIZE: 0,
            MessageType.MODEL_VERSION: self._round
        }
        channel.send(end, dummy_msg)
        logger.debug("sending dummy weights done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_reset_agg_goal_vars = Tasklet(self._reset_agg_goal_variables)

            task_put_dist = Tasklet(self.put, TAG_DISTRIBUTE)
            task_put_dist.set_continue_fn(cont_fn=lambda: self.trainer_no_show)

            task_put_upload = Tasklet(self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet(self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet(self.get, TAG_FETCH)

            task_eval = Tasklet(self.evaluate)

            task_update_round = Tasklet(self.update_round)

            task_end_of_training = Tasklet(self.inform_end_of_training)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)

        # create a loop object for asyncfl to manage concurrency as well as
        # aggregation goal
        asyncfl_loop = Loop(
            loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        task_internal_init >> task_load_data >> task_init >> loop(
            task_get_fetch >> task_reset_agg_goal_vars >> asyncfl_loop(
                task_put_dist >> task_get_aggr) >> task_put_upload >> task_eval
            >> task_update_round) >> task_end_of_training

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the middle level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_FETCH, TAG_UPLOAD]
