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
"""Constant definitions."""

from enum import Enum

MSG_LEN_FIELD_SIZE = 4
DEFAULT_RUN_ASYNC_WAIT_TIME = 10  # 10 seconds

# backend related constants
MQTT_TOPIC_PREFIX = "/flame"
UNIX_SOCKET_PATH = "/tmp/local_registry.socket"

# default data folder
DATA_FOLDER_PATH = "/flame/data"

EMPTY_PAYLOAD = b""


class CommType(Enum):
    """Enum class for communication type."""

    BROADCAST = 1
    UNICAST = 2


class DeviceType(Enum):
    """Enum class for device."""

    CPU = 1
    GPU = 2


class TrainState(Enum):
    """Enum class for train state."""

    PRE = "pre"
    DURING = "during"
    POST = "post"
