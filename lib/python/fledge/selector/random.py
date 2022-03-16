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

from ..end import End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class RandomSelector(AbstractSelector):
    """A random selector class."""

    def __init__(self):
        try:
            self.k = getattr(self, "k")
        except:
            logger.debug("k is not specified in config")

    def select(self, ends: dict[str, End]) -> SelectorReturnType:
        """Return all ends from the given ends."""
        logger.debug("calling random select")
        if len(ends) < self.k:
            logger.debug("selected greater than total")
            self.selected_ends = ends.keys()
            return {key: None for key in self.selected_ends}

        self.selected_ends = random.sample(ends.keys(), self.k)
        return {key: None for key in self.selected_ends}