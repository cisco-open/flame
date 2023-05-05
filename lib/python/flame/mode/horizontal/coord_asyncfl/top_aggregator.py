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

from flame.channel import VAL_CH_STATE_SEND
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from flame.mode.horizontal.asyncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.asyncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.mode.horizontal.coord_asyncfl import CHANNEL_TOP_TO_MID_AGG
from flame.mode.message import MessageType

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"  # coordinate with the coordinator


class TopAggregator(BaseTopAggregator):
    """Coordinated AsyncFL TopAggregator class."""

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

        data_plane_channel = self.get_channel(TAG_DISTRIBUTE)
        # overide distribute channel's ends method
        data_plane_channel.ends = self._override_data_plane_channel_ends

        # we call this to wait until coordinator appears
        _ = self.get_channel(TAG_COORDINATE)

    def _override_data_plane_channel_ends(
        self, state: Union[None, str] = None
    ) -> list[str]:
        logger.debug(f"argument: {state}")

        end_states = dict()
        data_plane_channel = self.get_channel(TAG_DISTRIBUTE, no_wait=True)
        # Note: don't call ends() method, because ends() was overriden
        #       in the internal_init(). If one calls ends(), it will create
        #       a recursive call, which will break this function.
        #       all_ends() is not overriden; so we are safe here.
        end_ids = data_plane_channel.all_ends()
        if len(end_ids) == 0:
            return []

        for end_id in end_ids:
            end_state = data_plane_channel.get_end_property(end_id, KEY_END_STATE)
            if end_state != VAL_END_STATE_RECVD:
                continue
            end_states[end_id] = end_state

        logger.debug(f"end_states = {end_states}")

        control_plane_channel = self.get_channel(TAG_COORDINATE)
        end_id = control_plane_channel.one_end()
        if not end_id:
            return []

        req = {
            MessageType.REQ_COORDINATED_ENDS: (
                CHANNEL_TOP_TO_MID_AGG,
                state,
                end_states,
            ),
        }
        control_plane_channel.send(end_id, req)
        logger.debug(f"request was sent: {req}")
        msg, _ = control_plane_channel.recv(end_id)

        logger.debug(f"received message = {msg} from {end_id}")

        if not msg:
            return []

        end_ids = msg[MessageType.RES_COORDINATED_ENDS]
        # when channel state is VAL_CH_STATE_SEND, it means that this
        # end wants to send a message. When fedbuff in the coordinator
        # receives a request, it clears VAL_END_STATE_RECVD of ends if
        # VAL_END_STATE_RECVD is set. This change needs to be
        # reflected here. All the ends received here are ones for
        # which fedbuff cleared VAL_END_STATE_RECVD property. So, we
        # also need to clear VAL_END_STATE_RECVD by setting
        # VAL_END_STATE_NONE
        if state == VAL_CH_STATE_SEND:
            for end in end_ids:
                data_plane_channel.set_end_property(
                    end, KEY_END_STATE, VAL_END_STATE_NONE
                )

        return end_ids

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        control_plane_channel = self.get_channel(TAG_COORDINATE)
        end_id = control_plane_channel.one_end()
        control_plane_channel.send(end_id, {MessageType.EOT: self._work_done})

        data_plane_channel = self.get_channel(TAG_DISTRIBUTE)
        data_plane_channel.broadcast({MessageType.EOT: self._work_done})

        logger.debug("done informing end-of-training")

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_COORDINATE]
