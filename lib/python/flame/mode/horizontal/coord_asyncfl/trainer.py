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
from typing import Union

from flame.mode.horizontal.asyncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.asyncfl.trainer import Trainer as BaseTrainer
from flame.mode.horizontal.coord_asyncfl import CHANNEL_TRAINER_TO_MID
from flame.mode.message import MessageType

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"


class Trainer(BaseTrainer):
    """Coordinated Asynchronous FL Trainer class."""

    def get_channel(self, tag: str, *, no_wait: bool = False):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        if not no_wait:
            channel.await_join()

        return channel

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        channel = self.get_channel(TAG_FETCH)
        # overide channel's ends method
        channel.ends = self._override_channel_ends

        # we call this to wait until coordinator appears
        _ = self.get_channel(TAG_COORDINATE)

    def _override_channel_ends(self, state: Union[None, str] = None) -> list[str]:
        logger.debug(f"argument: {state}")
        channel = self.get_channel(TAG_COORDINATE, no_wait=True)
        end_id = channel.one_end()
        if not end_id:
            return []

        req = {MessageType.REQ_COORDINATED_ENDS: (CHANNEL_TRAINER_TO_MID, state, None)}
        channel.send(end_id, req)
        msg, _ = channel.recv(end_id)
        logger.debug(f"received message = {msg} from {end_id}")

        if not msg:
            return []

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            return []

        logger.debug("exited _override_channel_ends")
        return msg[MessageType.RES_COORDINATED_ENDS]

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
