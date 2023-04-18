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
from datetime import datetime
from typing import Any, Union

import cloudpickle
from aiostream import stream

from flame.common.constants import EMPTY_PAYLOAD, CommType
from flame.common.typing import Scalar
from flame.common.util import run_async
from flame.config import GROUPBY_DEFAULT_GROUP
from flame.end import KEY_END_STATE, VAL_END_STATE_RECVD, End

logger = logging.getLogger(__name__)

KEY_CH_STATE = "state"
VAL_CH_STATE_RECV = "recv"
VAL_CH_STATE_SEND = "send"


class Channel(object):
    """Channel class."""

    def __init__(
        self,
        backend,
        selector,
        job_id: str,
        name: str,
        me="",
        other="",
        groupby=GROUPBY_DEFAULT_GROUP,
    ):
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

        # dict showing active, awaiting recv fifo tasks on each ends
        self._active_recv_fifo_tasks: set(str) = set()

        async def _setup():
            self.await_join_event = asyncio.Event()

            self._bcast_queue = asyncio.Queue()
            self._rx_queue = asyncio.Queue()

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

    def set_end_property(self, end_id: str, key: str, value: Scalar) -> None:
        """Set property of an end."""
        self._ends[end_id].set_property(key, value)

    def get_end_property(self, end_id, key) -> Scalar:
        """Get property of an end."""
        return self._ends[end_id].get_property(key)

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

    def one_end(self, state: Union[None, str] = None) -> str:
        """Return one end out of all ends."""
        return self.ends(state)[0]

    def ends(self, state: Union[None, str] = None) -> list[str]:
        """Return a list of end ids."""
        if state == VAL_CH_STATE_RECV or state == VAL_CH_STATE_SEND:
            self.properties[KEY_CH_STATE] = state

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

    def all_ends(self):
        """Return a list of all end ids (needed in FedDyn to compute alpha values)."""
        return list(self._ends.keys())

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

    def recv(self, end_id) -> tuple[Any, datetime]:
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

        if self.has(end_id):
            # set a property that says a message was received for the end
            self._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)

        # dissect the payload into msg and timestamp
        msg, timestamp = (
            (cloudpickle.loads(payload[0]), payload[1])
            if payload and status
            else (None, None)
        )

        # set cleanup ready event
        self._backend.set_cleanup_ready(end_id)

        return msg, timestamp

    def recv_fifo(
        self, end_ids: list[str], first_k: int = 0
    ) -> tuple[Any, tuple[str, datetime]]:
        """Receive a message per end from a list of ends.

        The message arrival order among ends is not fixed.
        Messages are yielded in a FIFO manner.
        This method is not thread-safe.

        Parameters
        ----------
        end_ids: a list of ends to receive a message from
        first_k: an integer argument to restrict the number of ends
                 to receive a messagae from. The default value (= 0)
                 means that we'd like to receive messages from all
                 ends in the list. If first_k > len(end_ids),
                 first_k is set to len(end_ids).

        Returns
        -------
        The function yields a pair: end id and message
        """
        logger.debug(f"first_k = {first_k}, len(end_ids) = {len(end_ids)}")

        first_k = min(first_k, len(end_ids))
        if first_k <= 0:
            # a negative value in first_k is an error
            # we handle it by setting first_k as the length of the array
            first_k = len(end_ids)

        self.first_k = first_k

        async def _put_message_to_rxq_inner():
            _ = asyncio.create_task(self._streamer_for_recv_fifo(end_ids))

        async def _get_message_inner():
            return await self._rx_queue.get()

        # first, create an asyncio task to fetch messages and put a temp queue
        # _put_message_to_rxq_inner works as if it is a non-blocking call
        # because a task is created within it
        _, _ = run_async(_put_message_to_rxq_inner(), self._backend.loop())

        # the _get_message_inner() coroutine fetches a message from the temp
        # queue; we call this coroutine first_k times
        for _ in range(first_k):
            result, status = run_async(_get_message_inner(), self._backend.loop())
            (end_id, payload) = result
            logger.debug(f"get payload for {end_id}")

            if self.has(end_id):
                logger.debug(f"channel got a msg for {end_id}")
                # set a property to indicate that a message was received
                # for the end
                self._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
            else:
                logger.debug(f"channel has no end id {end_id} for msg")

            msg, timestamp = (
                (cloudpickle.loads(payload[0]), payload[1])
                if payload and status
                else (None, None)
            )
            metadata = (end_id, timestamp)

            # set cleanup ready event
            self._backend.set_cleanup_ready(end_id)

            yield msg, metadata

    async def _streamer_for_recv_fifo(self, end_ids: list[str]):
        """Read messages in a FIFO fashion.

        This method reads messages from queues associated with each end
        and puts first_k number of the messages into a queue;
        The remaining messages are saved back into a variable (peek_buf)
        of their corresponding end so that they can be read later.
        """

        async def _get_inner(end_id) -> tuple[str, Any]:
            if not self.has(end_id):
                # can't receive message from end_id
                yield end_id, None

            payload = None
            try:
                payload = await self._ends[end_id].get()
            except KeyError:
                yield end_id, None

            yield end_id, payload

        runs = []
        for end_id in end_ids:
            if not self.has(end_id) or end_id in self._active_recv_fifo_tasks:
                continue
            else:
                runs.append(_get_inner(end_id))
                self._active_recv_fifo_tasks.add(end_id)

                logger.debug(f"active task added for {end_id}")
                logger.debug(f"{str(self._active_recv_fifo_tasks)}")

        # DO NOT CHANGE self.count as a local variable
        # with aiostream, local variable update looks incorrect.
        # but with an instance variable , the variable update is
        # done correctly.
        self.count = 0
        merged = stream.merge(*runs)
        async with merged.stream() as streamer:
            logger.debug(f"0) cnt: {self.count}, first_k: {self.first_k}")
            async for result in streamer:
                (end_id, payload) = result
                logger.debug(f"1) end id: {end_id}, cnt: {self.count}")

                self.count += 1
                logger.debug(f"2) end id: {end_id}, cnt: {self.count}")
                if self.count <= self.first_k:
                    logger.debug(f"3) end id: {end_id}, cnt: {self.count}")
                    await self._rx_queue.put(result)
                    self._active_recv_fifo_tasks.remove(end_id)

                    logger.debug(f"active task removed for {end_id}")
                    logger.debug(f"{str(self._active_recv_fifo_tasks)}")
                else:
                    logger.debug(f"4) end id: {end_id}, cnt: {self.count}")
                    # We already put the first_k number of messages into
                    # a queue.
                    #
                    # Now we need to save the remaining messages which
                    # were already taken out from each end's rcv queue.
                    # In order not to lose those messages, we use peek_buf
                    # in end object.

                    # WARNING: peek_buf must be none; if not, we called
                    # peek() somewhere else and then called recv_fifo()
                    # before recv() was called.
                    # To detect this potential issue, assert is given here.
                    assert self._ends[end_id].peek_buf is None

                    self._ends[end_id].peek_buf = payload
                    self._active_recv_fifo_tasks.remove(end_id)

                    logger.debug(f"active task removed for {end_id}")
                    logger.debug(f"{str(self._active_recv_fifo_tasks)}")

    def peek(self, end_id):
        """Peek rxq of end_id and return data if queue is not empty."""

        async def _peek():
            if not self.has(end_id):
                # can't peek message from end_id
                return None

            payload = await self._ends[end_id].peek()
            return payload

        payload, status = run_async(_peek(), self._backend.loop())

        msg, timestamp = (
            (cloudpickle.loads(payload[0]), payload[1])
            if payload and status
            else (None, None)
        )

        return msg, timestamp

    def drain_messages(self):
        """Drain messages from rx queues of ends."""
        for end_id in list(self._ends.keys()):
            while True:
                msg, _ = self.peek(end_id)
                if not msg:
                    break

                # drain message from end so that cleanup ready event is set
                _ = self.recv(end_id)

    def join(self):
        """Join the channel."""
        self._backend.join(self)

    def leave(self):
        """Clean up resources allocated in the channel and leave it."""
        self.drain_messages()

        self._backend.leave(self)

    def await_join(self, timeout=None) -> bool:
        """Wait for at least one peer joins a channel.

        If timeout value is set, it will wait until timeout occurs.
        Returns a boolean value to indicate whether timeout occurred or not.

        Parameters
        ----------
        timeout: a timeout value; default: None
        """

        async def _inner() -> bool:
            """Return True if timeout occurs; otherwise False."""
            logger.debug("waiting for join")
            try:
                await asyncio.wait_for(self.await_join_event.wait(), timeout)
            except asyncio.TimeoutError:
                logger.debug("timeout occurred")
                return True
            logger.debug("at least one peer joined")
            return False

        timeouted, _ = run_async(_inner(), self._backend.loop())
        logger.debug(f"timeouted = {timeouted}")
        return timeouted

    def is_rxq_empty(self, end_id: str) -> bool:
        """Return true if rxq is empty; otherwise, false."""
        return self._ends[end_id].is_rxq_empty()

    def is_txq_empty(self, end_id: str) -> bool:
        """Return true if txq is empty; otherwise, false."""
        return self._ends[end_id].is_txq_empty()

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

        # set the event true
        # it's okay to call set() without checking its condition
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
