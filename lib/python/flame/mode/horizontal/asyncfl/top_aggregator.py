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
"""Asynchronous horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)


class TopAggregator(SyncTopAgg):
    """Asynchronous top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

    def _reset_agg_goal_variables(self):
        logger.debug("reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        if self._agg_goal_weights is None:
            logger.debug(f"type of weights: {type(self.weights)}")
            self._agg_goal_weights = deepcopy(self.weights)

        # receive local model parameters from a trainer who arrives first
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received data from {end}")

        if MessageType.WEIGHTS in msg:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            tres = TrainResult(weights, count, version)
            # save training result from trainer in a disk cache
            self.cache[end] = tres
            logger.debug(f"received {len(self.cache)} trainer updates in cache")

            self._agg_goal_weights = self.optimizer.do(
                self._agg_goal_weights, self.cache, total=count, version=self._round
            )
            # increment agg goal count
            self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            # didn't reach the aggregation goal; return
            logger.debug("didn't reach agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            return

        if self._agg_goal_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = self._agg_goal_weights

        # update model with global weights
        self._update_model()

        logger.debug(f"aggregation finished for round {self._round}")

    def _distribute_weights(self, tag: str) -> None:
        """Distributed a global model in asynchronous FL fashion.

        This method is overriden from one in synchronous top aggregator
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
        for end in channel.ends(VAL_CH_STATE_SEND):
            logger.debug(f"sending weights to {end}")
            # we use _round to indicate a model version
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round,
                },
            )

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as _:
            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

        c = self.composer
        # unlink tasklets that are chained from the parent class
        # (i.e., super().compose()).
        #
        # unlink() internally calls tasklet.reset(), which in turn
        # initialize all loop related state, which includes cont_fn.
        # therefore, if cont_fn is needed for a tasklet, set_continue_fn()
        # in Tasklet class should be used.
        c.unlink()

        loop = Loop(loop_check_fn=lambda: self._work_done)

        # create a loop object for asyncfl to manage concurrency as well as
        # aggregation goal
        asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> c.tasklet("load_data")
            >> c.tasklet("initialize")
            >> loop(
                task_reset_agg_goal_vars
                >> asyncfl_loop(task_put >> task_get)
                >> c.tasklet("train")
                >> c.tasklet("evaluate")
                >> c.tasklet("analysis")
                >> c.tasklet("save_metrics")
                >> c.tasklet("inc_round")
            )
            >> c.tasklet("inform_end_of_training")
            >> c.tasklet("save_params")
            >> c.tasklet("save_model")
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
