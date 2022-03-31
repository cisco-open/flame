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

"""DefaultSelector class."""

import logging

from ..common.typing import Scalar
from ..end import End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class DefaultSelector(AbstractSelector):
    """A default selector class."""

    def select(self, ends: dict[str, End],
               channel_props: dict[str, Scalar]) -> SelectorReturnType:
        """Return all ends from the given ends."""
        logger.debug("calling default select")

        self.selected_ends = ends.keys()
        return {key: None for key in self.selected_ends}
