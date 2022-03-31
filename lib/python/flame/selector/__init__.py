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

from ..common.typing import Scalar
from ..end import End

SelectorReturnType = dict[str, Union[None, Tuple[str, Scalar]]]


class AbstractSelector(ABC):
    """Abstract base class for selector implementation."""

    def __init__(self, **kwargs) -> None:
        """Initialize an instance with keyword-based arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.selected_ends = list()

    @abstractmethod
    def select(self, ends: dict[str, End],
               channel_props: dict[str, Scalar]) -> SelectorReturnType:
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
