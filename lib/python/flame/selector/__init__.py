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
"""selector abstract class."""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import logging
import time

from ..common.typing import Scalar
from ..end import End

SelectorReturnType = dict[str, Union[None, Tuple[str, Scalar]]]

logger = logging.getLogger(__name__)


class AbstractSelector(ABC):
    """Abstract base class for selector implementation."""

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.selected_ends: set = set()
        self.ordered_updates_recv_ends: list = []

    def enforce_min_start(self, ends_count: int) -> bool:
        """Return True if selection should wait due to min-start threshold."""
        threshold = (
            int(self.minInitialTrainers)
            if hasattr(self, "minInitialTrainers")
            and self.minInitialTrainers is not None
            else -1
        )
        if ends_count < threshold:
            logger.debug(
                f"Not enough ends to start selection, need at least {threshold}"
            )
            time.sleep(0.1)
            return True
        return False

    @abstractmethod
    def select(
        self, ends: dict[str, End], channel_props: dict[str, Scalar]
    ) -> SelectorReturnType:
        """Abstract method to select ends.

        Parameters
        ----------
        ends: a dictionary whose key is end id and value is End object
        channel_props: properties set in channel

        Returns
        -------
        dictionary: key is end id and value is a property (as tuple)
                    used/created during selection process; value can be none
        """

    def on_update_received(
        self, end_id: str, msg: dict, round_num: int
    ) -> None:
        """Hook: aggregator calls this when a trainer update arrives.

        Default records the end_id for later cleanup. Subclasses override to
        extract per-update metrics (e.g. FedDance pulls LOCAL_ACCURACY).
        """
        if isinstance(self.selected_ends, set):
            self.ordered_updates_recv_ends.append(end_id)

    def on_round_completed(
        self, ends: dict[str, End], round_num: int
    ) -> None:
        """Hook: aggregator calls this after aggregation finishes.

        Default frees received-ends from the in-flight set. Subclasses with
        custom legacy cleanup (_cleanup_recvd_ends) get that called too.
        """
        if isinstance(self.selected_ends, set):
            for end_id in self.ordered_updates_recv_ends:
                self.selected_ends.discard(end_id)
            self.ordered_updates_recv_ends = []
        elif hasattr(self, "_cleanup_recvd_ends"):
            self._cleanup_recvd_ends(ends)
