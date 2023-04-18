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

from flame.backend.mqtt import MqttBackend
from flame.backend.p2p import PointToPointBackend
from flame.config import BackendType
from flame.object_factory import ObjectFactory


class BackendProvider(ObjectFactory):
    """Backend Provider."""

    def get(self, backend_name, **kwargs):
        """Return a backend for a given backend name."""
        return self.create(backend_name, **kwargs)


backend_provider = BackendProvider()
backend_provider.register(BackendType.P2P, PointToPointBackend)
backend_provider.register(BackendType.MQTT, MqttBackend)
