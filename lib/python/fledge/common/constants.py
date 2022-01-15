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

from enum import Enum

MSG_LEN_FIELD_SIZE = 4
DEFAULT_RUN_ASYNC_WAIT_TIME = 10  # 10 seconds

# backend related constants
MQTT_TOPIC_PREFIX = '/fledge'
UNIX_SOCKET_PATH = '/tmp/local_registry.socket'


class BackendEvent(Enum):
    """Enum class for BackendEvent."""

    DISCONNECT = 1
