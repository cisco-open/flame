# Copyright 2022 Cisco Systems, Inc. and its affiliates
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

"""hybrid FL trainer."""

import logging
import time

from ...mode.distributed.trainer import Trainer as DistTrainer
from ..composer import Composer
from ..message import MessageType
from ..tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_FETCH = 'fetch'
TAG_UPLOAD = 'upload'
TAG_RECEIVE = 'receive'
TAG_SEND = 'send'

class Trainer(DistTrainer):
    """Trainer inherit distributed trainer."""

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)
        elif tag == TAG_RECEIVE:
            super().get(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights from aggregator")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        while channel.empty():
            time.sleep(1)
            logger.debug("[_fetch_weights] waiting for channel ends")

        # one aggregator is sufficient
        end = channel.one_end()
        dict = channel.recv(end)
        for k, v in dict.items():
            if k == MessageType.WEIGHTS:
                self.weights = v
                self._update_model()
            elif k == MessageType.EOT:
                self._work_done = v
            elif k == MessageType.ROUND:
                self._round = v

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._upload_weights(tag)
        elif tag == TAG_SEND:
            super().put(tag)

    def _upload_weights(self, tag: str) -> None:
        logger.debug("calling _upload_weights to aggregator")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_upload_weights] channel not found with {tag}")
            return

        while channel.empty():
            time.sleep(1)
            logger.debug("[_upload_weights] waiting for channel ends")

        # one aggregator is sufficient
        end = channel.one_end()

        self._update_weights()
        if channel.get_backend_id() == self._lead_trainer:
            channel.send(end, {MessageType.WEIGHTS: self.weights, MessageType.DATASET_SIZE: self.dataset_size})
        else:
            channel.send(end, {MessageType.WEIGHTS: None, MessageType.DATASET_SIZE: 0})
        logger.debug("sending weights done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_load_data = Tasklet(self.load_data)

            task_init = Tasklet(self.initialize)

            task_get = Tasklet(self.get, TAG_FETCH)

            task_receive = Tasklet(self.get, TAG_RECEIVE)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_put = Tasklet(self.put, TAG_UPLOAD)

            task_send = Tasklet(self.put, TAG_SEND)

            task_aggregate = Tasklet(self._aggregate_weights)

            task_save_metrics = Tasklet(self.save_metrics)

            task_save_params = Tasklet(self.save_params)

            task_save_model = Tasklet(self.save_model)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            task_internal_init >> task_load_data >> task_init >> loop(
                task_get >> task_send >> task_receive >> task_aggregate 
                >> task_train >> task_eval >> task_save_metrics >> task_put 
                ) >> task_save_params >> task_save_model

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_SEND, TAG_RECEIVE]
