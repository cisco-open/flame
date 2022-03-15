# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""horizontal FL trainer."""

import logging
import time

from ...channel_manager import ChannelManager
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ..message import MessageType
from ..composer import Composer
from ..role import Role
from ..tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_FETCH = 'fetch'
TAG_UPLOAD = 'upload'


class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    #
    @abstract_attribute
    def weights(self):
        """Abstract attribute for model weights."""

    @abstract_attribute
    def dataset_size(self):
        """Abstract attribute for size of dataset used to train."""

    @abstract_attribute
    def config(self):
        """Abstract attribute for config object."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self._work_done = False

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        while len(channel._ends) == 0:
            time.sleep(1)
            logger.debug("[_fetch_weights] waiting for channel ends")

        # one aggregator is sufficient
        end = list(channel._ends.keys())[0]
        dict = channel.recv(end)
        for k, v in dict.items():
            if k == MessageType.WEIGHTS:
                self.weights = v
            elif k == MessageType.EOT:
                self._work_done = v

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        while len(channel._ends) == 0:
            time.sleep(1)
            logger.debug("[_send_weights] waiting for channel ends")

        # one aggregator is sufficient
        end = list(channel._ends.keys())[0]

        data = (self.weights, self.dataset_size)
        channel.send(end, data)
        logger.debug("sending weights done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_get = Tasklet(self.get, TAG_FETCH)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_put = Tasklet(self.put, TAG_UPLOAD)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            task_internal_init >> task_init >> loop(
                task_load_data >> task_get >> task_train >> task_eval >>
                task_put)

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD]
