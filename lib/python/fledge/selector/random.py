# Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
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
"""RandomSelector class."""

import logging
import random

from ..common.typing import Scalar
from ..end import End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class RandomSelector(AbstractSelector):
    """A random selector class."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.k = kwargs['k']
        except KeyError:
            raise KeyError("k is not specified in config")

        if self.k < 0:
            self.k = 1

        self.round = 0

    def select(self, ends: dict[str, End],
               channel_props: dict[str, Scalar]) -> SelectorReturnType:
        """Return k number of ends from the given ends."""
        logger.debug("calling random select")

        k = min(len(ends), self.k)
        if k == 0:
            logger.debug("ends is empty")
            return {}

        round = channel_props['round'] if 'round' in channel_props else 0

        if len(self.selected_ends) == 0 or round > self.round:
            logger.debug(f"let's select {k} ends for new round {round}")
            self.selected_ends = random.sample(ends.keys(), k)
            self.round = round

        logger.debug(f"selected ends: {self.selected_ends}")

        return {key: None for key in self.selected_ends}
