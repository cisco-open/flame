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
"""FedGFT horizontal FL top level aggregator."""

import logging
from datetime import datetime

from flame.common.constants import PROP_ROUND_END_TIME, PROP_ROUND_START_TIME
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.mode.composer import Composer
from flame.mode.horizontal.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.top_aggregator import \
    TopAggregator as BaseTopAggregator
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_AGGREGATE_BIAS = "aggregateBias"
TAG_DISTRIBUTE_BIAS = "distributeBias"


class TopAggregator(BaseTopAggregator):
    """FedGFT Top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for fedgft) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().internal_init()

    def _aggregate_bias(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        # store bias/dataset sizes
        dataset_sizes = dict()
        local_biases = dict()

        # receive local bias from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            if MessageType.BIAS in msg:
                local_biases[end] = msg[MessageType.BIAS]

            if MessageType.DATASET_SIZE in msg:
                dataset_sizes[end] = msg[MessageType.DATASET_SIZE]

        # optimizer complies information
        # TO DO: implement this on optimizer side
        self.optimizer.update_bias(
            dataset_sizes=dataset_sizes, local_biases=local_biases
        )

    def pre_process(self) -> None:
        """Complete this method before sending out bias (can be user-defined)."""
        # log bias
        self.update_metrics({"bias": abs(self.optimizer.get_bias())})

    def _distribute_bias(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        selected_ends = channel.ends()

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending bias to {end}")
            channel.send(
                end,
                {
                    MessageType.BIAS: self.optimizer.bias,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get_bias = Tasklet(
                "get_bias", self._aggregate_bias, TAG_AGGREGATE_BIAS
            )

            task_pre_process = Tasklet("pre_process", self.pre_process)

            task_put_bias = Tasklet(
                "put_bias", self._distribute_bias, TAG_DISTRIBUTE_BIAS
            )

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_analysis = Tasklet("analysis", self.run_analysis)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_end_of_training = Tasklet(
                "inform_end_of_training", self.inform_end_of_training
            )

            task_save_params = Tasklet("save_params", self.save_params)

            task_save_model = Tasklet("save_model", self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_put
                >> task_get_bias
                >> task_pre_process
                >> task_put_bias
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_increment_round
            )
            >> task_end_of_training
            >> task_save_params
            >> task_save_model
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_DISTRIBUTE_BIAS, TAG_AGGREGATE_BIAS]
