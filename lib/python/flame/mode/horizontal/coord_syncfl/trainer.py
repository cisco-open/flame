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
from abc import ABCMeta

from flame.common.constants import DeviceType, TrainerState
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import Composer
from flame.mode.horizontal.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_GET_AGGREGATOR = "getAggregator"


class Trainer(BaseTrainer, metaclass=ABCMeta):
    def _get_aggregator(self):
        logger.debug("calling _get_aggregator")
        channel = self.cm.get_by_tag(TAG_GET_AGGREGATOR)
        if not channel:
            logger.debug(f"channel not found with tag {TAG_GET_AGGREGATOR}")
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.COORDINATED_ENDS in msg:
            self.aggregator_id = msg[MessageType.COORDINATED_ENDS]

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        msg, _ = channel.recv(self.aggregator_id)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        self.regularizer.save_state(TrainerState.PRE_TRAIN, glob_model=self.model)
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        self._update_weights()
        self.regularizer.save_state(TrainerState.POST_TRAIN, loc_model=self.model)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        # send delta_weights to regularizer
        self.regularizer.update()

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
        }
        channel.send(self.aggregator_id, msg)
        logger.debug("sending weights done")

    def compose(self) -> None:
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_load_data = Tasklet(self.load_data)

            task_init = Tasklet(self.initialize)

            task_get_aggregator = Tasklet(self._get_aggregator)

            task_get = Tasklet(self.get, TAG_FETCH)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_put = Tasklet(self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet(self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_get_aggregator
                    >> task_get
                    >> task_train
                    >> task_eval
                    >> task_put
                    >> task_save_metrics
                )
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_GET_AGGREGATOR]
