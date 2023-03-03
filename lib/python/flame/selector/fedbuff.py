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
"""FedBuffSelector class."""

import logging
import random

from ..channel import KEY_CH_STATE, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from ..common.typing import Scalar
from ..end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class FedBuffSelector(AbstractSelector):
    """A selector class for fedbuff-based asyncfl."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.c = kwargs['c']
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        self.round = 0

    def select(self, ends: dict[str, End],
               channel_props: dict[str, Scalar]) -> SelectorReturnType:
        """Select ends from the given ends to meet concurrency level.

        This select method chooses ends differently depending on what state
        a channel is in.
        In 'send' state, it chooses ends that are not in self.selected_ends.
        In 'recv' state, it chooses all ends from self.selected_ends.
        Essentially, if an end is in self.selected_ends, it means that we sent
        some message already to that end. For such an end, we exclude it from
        send and include it for recv in return.
        """
        logger.debug("calling fedbuff select")
        logger.debug(f"len(ends): {len(ends)}, c: {self.c}")

        concurrency = min(len(ends), self.c)
        if concurrency == 0:
            logger.debug("ends is empty")
            return {}

        self.round = channel_props['round'] if 'round' in channel_props else 0

        if KEY_CH_STATE not in channel_props:
            raise KeyError("channel property doesn't have {KEY_CH_STATE}")

        self._cleanup_recvd_ends(ends)
        results = {}
        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            results = self._handle_send_state(ends, concurrency)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
            results = self._handle_recv_state(ends, concurrency)

        else:
            state = channel_props[KEY_CH_STATE]
            raise ValueError(f"unkown channel state: {state}")

        logger.debug(f"selected ends: {self.selected_ends}")
        logger.debug(f"results: {results}")

        return results

    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected ends."""
        logger.debug("clean up recvd ends")
        logger.debug(f"ends: {ends}, selected ends: {self.selected_ends}")
        for end_id in list(self.selected_ends):
            if end_id not in ends:
                # something happened to end of end_id
                # (e.g., connection loss)
                # let's remove it from selected_ends
                logger.debug(f"no end id {end_id} in ends")
                self.selected_ends.remove(end_id)
            else:
                state = ends[end_id].get_property(KEY_END_STATE)
                logger.debug(f"end id {end_id} state: {state}")
                if state == VAL_END_STATE_RECVD:
                    ends[end_id].set_property(KEY_END_STATE,
                                              VAL_END_STATE_NONE)
                    self.selected_ends.remove(end_id)

    def _handle_send_state(self, ends: dict[str, End],
                           concurrency: int) -> SelectorReturnType:
        extra = max(0, concurrency - len(self.selected_ends))
        logger.debug(f"c: {concurrency}, ends: {ends.keys()}")
        candidates = []
        idx = 0
        # reservoir sampling
        for end_id in ends.keys():
            if end_id in self.selected_ends:
                # skip if an end is already selected
                continue

            idx += 1
            if len(candidates) < extra:
                candidates.append(end_id)
                continue

            i = random.randrange(idx)
            if i < extra:
                candidates[i] = end_id

        logger.debug(f"candidates: {candidates}")
        # add candidates to selected ends
        self.selected_ends = set(list(self.selected_ends) + candidates)

        return {end_id: None for end_id in candidates}

    def _handle_recv_state(self, ends: dict[str, End],
                           concurrency: int) -> SelectorReturnType:
        if len(self.selected_ends) == 0:
            logger.debug(f"let's select {concurrency} ends")
            self.selected_ends = set(random.sample(list(ends), concurrency))

        logger.debug(f"selected ends: {self.selected_ends}")

        return {key: None for key in self.selected_ends}
