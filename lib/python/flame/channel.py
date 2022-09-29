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
import logging
from typing import Any, Tuple, Union

import cloudpickle
from aiostream import stream

from .common.constants import EMPTY_PAYLOAD, CommType
from .common.typing import Scalar
from .common.util import run_async
from .config import GROUPBY_DEFAULT_GROUP
from .end import End

logger = logging.getLogger(__name__)


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
        self.await_join_event = None

        # access _ends with caution.
        # in many cases, _ends must be accessed within a backend's loop
        self._ends: dict[str, End] = dict()

        async def _setup():
            self.await_join_event = asyncio.Event()

            self._bcast_queue = asyncio.Queue()

            # attach this channel to backend
            self._backend.attach_channel(self)

            # create tx task for broadcast queue
            self._backend.create_tx_task(self._name, "", CommType.BROADCAST)

        _, _ = run_async(_setup(), self._backend.loop())

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
        """Return True if channels has no end. Otherwise, return False."""

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

    def ends_digest(self) -> str:
        """Compute a digest of ends."""
        list_ends = self.ends()
        if len(list_ends) == 0:
            return ""

        # convert my end id (string) into an array
        digest = [c for c in self._backend.uid()]
        for end_id in list_ends:
            digest = [chr(ord(a) ^ ord(b)) for a, b in zip(digest, end_id)]

        return "".join(digest)

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
            logger.debug(f"length of payload = {len(payload)}")
            await self._ends[end_id].put(payload)

        _, status = run_async(_put(), self._backend.loop())

        return status

    def recv(self, end_id) -> Any:
        """Receive a message from an end in a blocking call fashion."""
        logger.debug(f"will receive data from {end_id}")

        async def _get():
            if not self.has(end_id):
                # can't receive message from end_id
                return None

            payload = None
            try:
                payload = await self._ends[end_id].get()
            except KeyError:
                return None

            return payload

        payload, status = run_async(_get(), self._backend.loop())

        return cloudpickle.loads(payload) if payload and status else None

    def recv_fifo(self, end_ids: list[str]) -> Tuple[str, Any]:
        """Receive a message per end from a list of ends.

        The message arrival order among ends is not fixed.
        Messages are yielded in a FIFO manner.
        This method is not thread-safe.
        """

        async def _get(end_id) -> Tuple[str, Any]:
            if not self.has(end_id):
                # can't receive message from end_id
                yield end_id, None

            payload = None
            try:
                payload = await self._ends[end_id].get()
            except KeyError:
                yield end_id, None

            yield end_id, payload

        async def _streamer(tmpq):
            runs = [_get(end_id) for end_id in end_ids]

            merged = stream.merge(*runs)
            async with merged.stream() as streamer:
                async for result in streamer:
                    await tmpq.put(result)

        # a temporary aysncio queue to store messages in a FIFO manner.
        # we define this varialbe to make sure it is visiable
        # in both _inner1() and _inner2()
        tmpq = None

        async def _inner1():
            nonlocal tmpq
            # tmpq must be created in the _backend loop
            tmpq = asyncio.Queue()
            _ = asyncio.create_task(_streamer(tmpq))

        async def _inner2():
            return await tmpq.get()

        # first, create an asyncio task to fetch messages and put a temp queue
        # _inner1 works as if it is a non-blocking call
        # because a task is created within it
        _, _ = run_async(_inner1(), self._backend.loop())

        # the _inner2() coroutine fetches a message from the temp queue
        # we call this coroutine the number of end_ids by iterating end_ids
        for _ in end_ids:
            result, status = run_async(_inner2(), self._backend.loop())
            (end_id, payload) = result
            msg = cloudpickle.loads(payload) if payload and status else None
            yield end_id, msg

    def peek(self, end_id):
        """Peek rxq of end_id and return data if queue is not empty."""

        async def _peek():
            if not self.has(end_id):
                # can't peek message from end_id
                return None

            payload = await self._ends[end_id].peek()
            return payload

        payload, status = run_async(_peek(), self._backend.loop())

        return cloudpickle.loads(payload) if payload and status else None

    def join(self):
        """Join the channel."""
        self._backend.join(self)

    def await_join(self):
        """Wait for at least one peer joins a channel."""

        async def _inner():
            logger.debug("waiting for join")
            await self.await_join_event.wait()
            logger.debug("at least one peer joined")

        _, _ = run_async(_inner(), self._backend.loop())

    def is_rxq_empty(self, end_id: str) -> bool:
        """Return true if rxq is empty; otherwise, false."""
        return self._ends[end_id].is_rxq_empty()

    def is_txq_empty(self, end_id: str) -> bool:
        """Return true if txq is empty; otherwise, false."""
        return self._ends[end_id].is_txq_empty()

    def cleanup(self):
        """Clean up resources allocated for the channel."""

        async def _inner():
            # we add this check because not all backends implement cleanup()
            # TODO: once all backends implement cleanup(), remove this check
            if not hasattr(self._backend, 'cleanup'):
                return

            # we use EMPTY_PAYLOAD as signal to finish tx tasks

            # put EMPTY_PAYLOAD to broadcast queue
            await self._bcast_queue.put(EMPTY_PAYLOAD)

            for end_id, end in self._ends.items():
                # put EMPTY_PAYLOAD to unicast queue for each end
                await end.put(EMPTY_PAYLOAD)

            await self._backend.cleanup()

        _, _ = run_async(_inner(), self._backend.loop())

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

        if not self.await_join_event.is_set():
            # set the event true
            self.await_join_event.set()

    async def remove(self, end_id):
        """Remove an end from the channel."""
        if not self.has(end_id):
            return

        rxq = self._ends[end_id].get_rxq()
        del self._ends[end_id]

        # put bogus data to unblock a get() call
        await rxq.put(EMPTY_PAYLOAD)

        if len(self._ends) == 0:
            # clear (or unset) the event
            self.await_join_event.clear()

    def has(self, end_id: str) -> bool:
        """Check if an end is in the channel."""
        return end_id in self._ends

    def get_rxq(self, end_id: str) -> Union[None, asyncio.Queue]:
        """Return a rx queue associated wtih an end."""
        if not self.has(end_id):
            return None

        return self._ends[end_id].get_rxq()

    def get_txq(self, end_id: str) -> Union[None, asyncio.Queue]:
        """Return a tx queue associated wtih an end."""
        if not self.has(end_id):
            return None

        return self._ends[end_id].get_txq()

    def broadcast_q(self):
        """Return a broadcast queue object."""
        return self._bcast_queue

    def get_backend_id(self) -> str:
        """Return backend id."""
        return self._backend.uid()
