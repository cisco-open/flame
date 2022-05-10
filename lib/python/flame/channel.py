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

"""Channel."""

import asyncio

import cloudpickle

from .common.constants import CommType
from .common.typing import Scalar
from .common.util import run_async
from .config import GROUPBY_DEFAULT_GROUP
from .end import End


class Channel(object):
    """Channel class."""

    def __init__(self,
                 backend,
                 selector,
                 job_id: str,
                 name: str,
                 me='',
                 other='',
                 groupby=GROUPBY_DEFAULT_GROUP):
        """Initialize instance."""
        self._backend = backend
        self._selector = selector
        self._job_id = job_id
        self._name = name
        self._my_role = me
        self._other_role = other
        self._groupby = groupby
        self.properties = dict()

        # _ends must be accessed within backend's loop
        self._ends: dict[str, End] = dict()

        async def _setup_bcast_tx():
            self._bcast_queue = asyncio.Queue()

            # attach this channel to backend
            self._backend.attach_channel(self)

            # create tx task for broadcast queue
            self._backend.create_tx_task(self._name, "", CommType.BROADCAST)

        _, _ = run_async(_setup_bcast_tx(), self._backend.loop())

    def job_id(self) -> str:
        """Return job id."""
        return self._job_id

    def name(self) -> str:
        """Return channel name."""
        return self._name

    def my_role(self) -> str:
        """Return my role's name."""
        return self._my_role

    def other_role(self) -> str:
        """Return other role's name."""
        return self._other_role

    def groupby(self) -> str:
        """Return groupby tag."""
        return self._groupby

    def set_property(self, key: str, value: Scalar) -> None:
        """Set property of channel.

        Parameters
        ----------
        key: string
        value: any of boolean, bytes, float, int, or string
        """
        self.properties[key] = value

    """
    ### The following are not asyncio methods
    ### But access to _ends variable should take place in the backend loop
    ### Therefore, when necessary, coroutine is defined inside each method
    ### and the coroutine is executed via run_async()
    """

    def empty(self) -> bool:
        """Return True/False on whether channels has ends or not."""

        async def inner() -> bool:
            return len(self._ends) == 0

        result, _ = run_async(inner(), self._backend.loop())

        return result

    def one_end(self) -> str:
        """Return one end out of all ends."""
        return self.ends()[0]

    def ends(self) -> list[str]:
        """Return a list of end ids."""

        async def inner():
            selected = self._selector.select(self._ends, self.properties)

            id_list = list()
            for end_id, kv in selected.items():
                id_list.append(end_id)
                if not kv:
                    continue

                (key, value) = kv
                self._ends[end_id].set_property(key, value)

            return id_list

        result, _ = run_async(inner(), self._backend.loop())
        return result

    def broadcast(self, message):
        """Broadcast a message in a blocking call fashion."""

        async def _put():
            payload = cloudpickle.dumps(message)
            await self._bcast_queue.put(payload)

        _, status = run_async(_put(), self._backend.loop())

    def send(self, end_id, message):
        """Send a message to an end in a blocking call fashion."""

        async def _put():
            if not self.has(end_id):
                # can't send message to end_id
                return

            payload = cloudpickle.dumps(message)
            await self._ends[end_id].put(payload)

        _, status = run_async(_put(), self._backend.loop())

        return status

    def recv(self, end_id):
        """Receive a message from an end in a blocking call fashion."""

        async def _get():
            if not self.has(end_id):
                # can't receive message from end_id
                return None

            payload = await self._ends[end_id].get()
            return payload

        payload, status = run_async(_get(), self._backend.loop())

        return cloudpickle.loads(payload) if payload and status else None

    def join(self):
        """Join the channel."""
        self._backend.join(self)

    """
    ### The following are asyncio methods of backend loop
    ### Therefore, they must be called in the backend loop
    """

    async def add(self, end_id):
        """Add an end to the channel and allocate rx and tx queues for it."""
        if self.has(end_id):
            return

        self._ends[end_id] = End(end_id)

        # create tx task in the backend for the channel
        self._backend.create_tx_task(self._name, end_id)

    async def remove(self, end_id):
        """Remove an end from the channel."""
        if not self.has(end_id):
            return

        rxq = self._ends[end_id].get_rxq()
        del self._ends[end_id]

        # put bogus data to unblock a get() call
        await rxq.put(b'')

    def has(self, end_id: str) -> bool:
        """Check if an end is in the channel."""
        return end_id in self._ends

    def get_rxq(self, end_id: str) -> asyncio.Queue:
        """Return a rx queue associated wtih an end."""
        return self._ends[end_id].get_rxq()

    def get_txq(self, end_id: str) -> asyncio.Queue:
        """Return a tx queue associated wtih an end."""
        return self._ends[end_id].get_txq()

    def broadcast_q(self):
        """Return a broadcast queue object."""
        return self._bcast_queue

    def get_backend_id(self) -> str:
        """Return backend id."""
        return self._backend.uid()
