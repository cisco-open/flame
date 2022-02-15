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
"""SimpleSelector class."""

import logging

from ..end import End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class SimpleSelector(AbstractSelector):
    """A simple selector class."""

    def select(self, ends: dict[str, End]) -> SelectorReturnType:
        """Return all ends from the given ends."""
        logger.debug("calling select")

        return {key: None for key in ends.keys()}
