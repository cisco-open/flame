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
"""End class."""

import asyncio
from typing import Union

from .common.typing import Scalar

KEY_END_STATE = 'state'
VAL_END_STATE_RECVD = 'recvd'
VAL_END_STATE_NONE = ''


class End(object):
    """End class."""

    def __init__(self, end_id: str) -> None:
        """Initialize end instance."""
        self.end_id = end_id
        self.rxq = asyncio.Queue()
        self.txq = asyncio.Queue()
        self.peek_buf = None

        # a property is a key-value pair where key is a name of property
        # and value is a scalar that describes or quantifies the property.
        # One or more properties can be used when an end is evaluated.
        # For example, properties can be used to select an end.
        self.properties: dict[str, Scalar] = dict()

    def set_property(self, key: str, value: Scalar) -> None:
        """Set a property of an end."""
        self.properties[key] = value

    def get_property(self, key: str) -> Union[None, Scalar]:
        """Return a property of a given property name."""
        return self.properties[key] if key in self.properties else None

    def reset_property(self, key: str) -> None:
        """Remove a property from an end."""
        if key in self.properties:
            del self.properties[key]

    def cleanup_properties(self):
        """Clean up all properties."""
        self.properties = dict()

    async def put(self, payload: bytes) -> None:
        """Put a payload into a tx queue."""
        await self.txq.put(payload)

    async def get(self) -> bytes:
        """Get a payload from a rx queue."""
        if self.peek_buf is not None:
            data = self.peek_buf
            self.peek_buf = None
            return data

        payload = await self.rxq.get()
        self.rxq.task_done()

        return payload

    async def peek(self) -> Union[None, bytes]:
        """Peek item in a rxq."""
        if self.peek_buf is not None:
            return self.peek_buf

        if self.rxq.empty():
            return None

        self.peek_buf = await self.get()

        return self.peek_buf

    def get_rxq(self) -> asyncio.Queue:
        """Return a rx queue."""
        return self.rxq

    def get_txq(self) -> asyncio.Queue:
        """Return a tx queue."""
        return self.txq

    def is_rxq_empty(self) -> bool:
        """Return true if rxq is empty; otherwise, false."""
        return self.rxq.empty()

    def is_txq_empty(self) -> bool:
        """Return true if txq is empty; otherwise, false."""
        return self.txq.empty()
