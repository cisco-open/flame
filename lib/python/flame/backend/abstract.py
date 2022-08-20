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
"""Abstract backend."""

from abc import ABC, abstractmethod

from ..common.constants import CommType


class AbstractBackend(ABC):
    """AbstractBackend class."""

    @abstractmethod
    def configure(self, broker: str, job_id: str, task_id: str):
        """Configure the backend."""
        pass

    @abstractmethod
    def eventq(self):
        """Return a event queue object."""
        pass

    @abstractmethod
    def loop(self):
        """Return loop instance of asyncio."""
        pass

    @abstractmethod
    def uid(self):
        """Return backend id."""
        pass

    @abstractmethod
    def join(self, channel) -> None:
        """Join a channel."""
        pass

    @abstractmethod
    def create_tx_task(self,
                       channel_name: str,
                       end_id: str,
                       comm_type=CommType.UNICAST) -> bool:
        """Create asyncio task for transmission."""
        pass

    @abstractmethod
    def attach_channel(self, channel):
        """Attach a channel to backend."""
        pass
