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

"""Factory for discovery clients."""

from .discovery.local_client import LocalDiscoveryClient
from .object_factory import ObjectFactory


class DiscoveryClientProvider(ObjectFactory):
    """Discovery client provider."""

    #
    def get(self, client_name, **kargs):
        """Return a discovery client object given a name."""
        return self.create(client_name, **kargs)


discovery_client_provider = DiscoveryClientProvider()
discovery_client_provider.register('local', LocalDiscoveryClient)
