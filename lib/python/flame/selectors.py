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
"""selector provider class."""

from .config import SelectorType
from .object_factory import ObjectFactory
from .selector.default import DefaultSelector
from .selector.random import RandomSelector


class SelectorProvider(ObjectFactory):
    """Selector Provider."""

    def get(self, selector_name, **kwargs):
        """Return a end selector for a given selector name."""
        return self.create(selector_name, **kwargs)


selector_provider = SelectorProvider()
selector_provider.register(SelectorType.DEFAULT, DefaultSelector)
selector_provider.register(SelectorType.RANDOM, RandomSelector)
