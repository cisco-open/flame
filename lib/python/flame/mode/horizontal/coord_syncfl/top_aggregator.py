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

from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_NOTIFY_COORDINATOR = "notifyCoordinator"  # notify orchestrator of EOT


class TopAggregator(BaseTopAggregator):
    def notify_coordinator(self) -> None:
        """Notify coordinator of the status of work."""
        coordinator_channel = self.cm.get_by_tag(TAG_NOTIFY_COORDINATOR)
        if not coordinator_channel:
            logger.debug(f"channel not found for tag {TAG_NOTIFY_COORDINATOR}")
            return

        coordinator_channel.await_join()
        end = coordinator_channel.one_end()
        coordinator_channel.send(end, {MessageType.EOT: self._work_done})

    def inform_end_of_training(self) -> None:
        """Inform all trainers and the orchestrator that the training is finished."""
        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})
        logger.debug("done broadcasting end-of-training")

        self.notify_coordinator()

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_init = Tasklet("", self.initialize)

            task_load_data = Tasklet("", self.load_data)

            task_notify_coord = Tasklet("", self.notify_coordinator)

            task_put = Tasklet("", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("", self.get, TAG_AGGREGATE)

            task_train = Tasklet("", self.train)

            task_eval = Tasklet("", self.evaluate)

            task_analysis = Tasklet("", self.run_analysis)

            task_save_metrics = Tasklet("", self.save_metrics)

            task_increment_round = Tasklet("", self.increment_round)

            task_end_of_training = Tasklet("", self.inform_end_of_training)

            task_save_params = Tasklet("", self.save_params)

            task_save_model = Tasklet("", self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_notify_coord
                >> task_put
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

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_NOTIFY_COORDINATOR]
