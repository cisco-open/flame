# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
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
"""Model registry provider."""

from typing import Union

from .config import RegistryType
from .object_factory import ObjectFactory
from .registry.mlflow import MLflowRegistryClient


class RegistryProvider(ObjectFactory):
    """Model registry provider."""

    #
    def get(self, registry_name, **kargs) -> Union[MLflowRegistryClient]:
        """Return a registry client for a given registry name."""
        return self.create(registry_name, **kargs)


registry_provider = RegistryProvider()
registry_provider.register(RegistryType.MLFLOW, MLflowRegistryClient)
