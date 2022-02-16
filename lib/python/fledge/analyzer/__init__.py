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

"""Analyzer abstract class."""

from abc import abstractmethod
from typing import Any, List, Union

from ..common.typing import Metrics
from ..plugin import Plugin


class AbstractAnalyzer(Plugin):
    """Abstract base class for analyzer implementation."""

    def callback(self):
        """Return callback function."""
        return self.run

    @abstractmethod
    def run(self,
            model: Any = None,
            dataset: Union[None, List[Any]] = None) -> Union[None, Metrics]:
        """Run analysis and return results."""
