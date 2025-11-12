# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Channel."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Union
import time
import cloudpickle
from aiostream import stream
from flame.common.constants import EMPTY_PAYLOAD, CommType
from flame.common.typing import Scalar
from flame.common.util import run_async
from flame.config import TrainerAvailState, GROUPBY_DEFAULT_GROUP
from flame.end import KEY_END_STATE, VAL_END_STATE_RECVD, PROP_END_AVL_STATE, End
from flame.mode.message import MessageType
from flame.mode.role import Role
import gzip
import zstandard as zstd
import sys

logger = logging.getLogger(__name__)

KEY_CH_STATE = "state"
VAL_CH_STATE_RECV = "recv"
VAL_CH_STATE_SEND = "send"
VAL_CH_STATE_HTBT_RECV = "heartbeat_recv"
VAL_CH_STATE_HTBT_SEND = "heartbeat_send"

KEY_CH_SELECT_REQUESTER = "requester"

END_LAST_AVAIL_TS = "end_last_avail_ts"
END_LAST_UNAVAIL_TS = "end_last_unavail_ts"
PROP_TOTAL_AVAIL_DURATION = "total_avail_duration"
PROP_TOTAL_UNAVAIL_DURATION = "total_unavail_duration"


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
        self.mc = Role.mc

        self.trainer_unavail_list = None

        # access _ends with caution. in many cases, _ends must be
        # accessed within a backend's loop
        self._ends: dict[str, End] = dict()

        # separate data structure to track end state across
        # disconnections
        self._end_state_info = dict()

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
        key: string value: any of boolean, bytes, float, int, or
        string
        """
        self.properties[key] = value

    def set_end_property(self, end_id: str, key: str, value: Scalar) -> None:
        """Set property of an end."""
        if self.has(end_id):
            self._ends[end_id].set_property(key, value)
            logger.debug(f"SET property {key} with val {value} for end_id {end_id}")
        else:
            logger.debug(
                f"Failed to SET property {key} with {value} as end_id {end_id}"
                f" not found"
            )

    def get_end_property(self, end_id, key) -> Scalar:
        """Get property of an end."""
        if self.has(end_id):
            val = self._ends[end_id].get_property(key)
            logger.debug(f"GOT property {key} with val {val} for end_id {end_id}")
            return val
        else:
            logger.debug(
                f"Failed to GET property {key} as end_id {end_id}" f" not found"
            )
            return None

    """
    ### The following are not asyncio methods ### But access to _ends
    variable should take place in the backend loop ### Therefore, when
    necessary, coroutine is defined inside each method ### and the
    coroutine is executed via run_async()
    """

    def empty(self) -> bool:
        """Return True if channels has no end. Otherwise, return
        False."""

        async def inner() -> bool:
            return len(self._ends) == 0

        result, _ = run_async(inner(), self._backend.loop())

        return result

    def one_end(self, state: Union[None, str] = None) -> Union[None, str]:
        """Return one end out of all ends."""
        end_list = self.ends(state)
        return end_list[0] if len(end_list) > 0 else None

    def ends(
        self, state: Union[None, str] = None, task_to_perform: str = "train"
    ) -> list[str]:
        """Return a list of end ids."""
        logger.debug(
            f"ends() for channel name: {self._name}, "
            f"current self._ends: {self._ends}"
        )
        if state == VAL_CH_STATE_RECV or state == VAL_CH_STATE_SEND:
            self.properties[KEY_CH_STATE] = state

        self.properties[KEY_CH_SELECT_REQUESTER] = self.get_backend_id()

        async def inner():
            # while hasattr(self._selector, "k") and len(self._ends) < self._selector.k:

            #     logger.debug(f"sleeping till ends enough, current ends: {len(self._ends)}")
            #     time.sleep(5)
            if self.trainer_unavail_list is not None:
                selected = self._selector.select(
                    self._ends,
                    self.properties,
                    self.trainer_unavail_list,
                    task_to_perform,
                )
                logger.debug(f"selected: {selected}")
                if len(selected) is 0:
                    return
            else:
                selected = self._selector.select(
                    self._ends, self.properties, task_to_perform
                )
                logger.debug(f"selected: {selected}")
                if len(selected) is 0:
                    return
            logger.debug(
                f"selected for task {task_to_perform} and returned from select(): {selected}"
            )

            id_list = list()
            for end_id, kv in selected.items():
                id_list.append(end_id)
                logger.debug(f"appended end_id {end_id} to id_list, kv is {kv}")
                if not kv:
                    continue

                (key, value) = kv
                logger.debug(
                    f"Setting property for end_id {end_id} "
                    f"using (key,val) = ({key},{value})"
                )
                self._ends[end_id].set_property(key, value)
                logger.debug(
                    f"Updated end_id {end_id} property to key: {key}, "
                    f"value: {value} in self._ends"
                )
            logger.debug(f"Going to return id_list: {id_list}")
            return id_list

        result, _ = run_async(inner(), self._backend.loop())
        logger.debug(f"Going to return result: {result}")
        return result

    def all_ends(self):
        """Return a list of all end ids (needed in FedDyn to compute
        alpha values)."""
        return list(self._ends.keys())

    def cleanup_recvd_ends(self):
        """Performs cleanup of end states in the selector. Usually
        only performed after aggregation of a round completes"""

        # TODO: (DG) This function is named to cleanup recvd ends, but
        # can extend beyond just "recvd" state. We might also want to
        # send a subset of ends here not the entire self._ends?

        self._selector._cleanup_recvd_ends(self._ends)
        logger.debug("Cleaned up ends successfully")

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
            self.mc.accumulate("bytes", "broadcast", len(payload))
            await self._bcast_queue.put(payload)

        _, status = run_async(_put(), self._backend.loop())

    def send(self, end_id, message):
        """Send a message to an end in a blocking call fashion."""

        async def _put():
            if not self.has(end_id):
                # can't send message to end_id
                return

            payload = cloudpickle.dumps(message)
            # payload2 = gzip.compress(payload)
            # compressor = zstd.ZstdCompressor()
            # payload2 = compressor.compress(payload)
            self.mc.accumulate("bytes", "send", len(payload))
            logger.debug(f"size of payload = {sys.getsizeof(payload)}")
            await self._ends[end_id].put(payload)

        _, status = run_async(_put(), self._backend.loop())

        return status

    def recv(self, end_id) -> tuple[Any, datetime]:
        # NOTE (DG): This isnt being used in horizontal top-agg async,
        # checked

        """Receive a message from an end in a blocking call
        fashion."""
        logger.debug(f"will receive data from {end_id}")

        async def _get():
            if not self.has(end_id):
                # can't receive message from end_id
                return None

            payload = None
            try:
                payload = await self._ends[end_id].get()
                if payload:
                    # ignore timestamp for measuring bytes received
                    # payload = gzip.decompress(payload)
                    # decompressor = zstd.ZstdDecompressor()
                    # payload = decompressor.decompress(payload)
                    self.mc.accumulate("bytes", "recv", len(payload[0]))
            except KeyError:
                return None

            return payload

        payload, status = run_async(_get(), self._backend.loop())

        if self.has(end_id):
            # set a property that says a message was received for the
            # end
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

        The message arrival order among ends is not fixed. Messages
        are yielded in a FIFO manner. This allows recv_fifo to return
        (msg, metadata) asynchronously to the caller method. This
        method is not thread-safe.

        Parameters
        ----------
        end_ids: a list of ends to receive a message from first_k: an
        integer argument to restrict the number of ends
                 to receive a messagae from. The default value (= 0)
                 means that we'd like to receive messages from all
                 ends in the list. If first_k > len(end_ids), first_k
                 is set to len(end_ids).

        Returns
        -------
        The function yields a pair: end id and message
        """
        logger.debug(
            f"Receive fifo: first_k = {first_k}, len(end_ids) = {len(end_ids)}"
        )

        first_k = min(first_k, len(end_ids))
        if first_k <= 0:
            # a negative value in first_k is an error we handle it by
            # setting first_k as the length of the array
            logger.debug(f"first_k < 0 with value {first_k}")
            first_k = len(end_ids)

        self.first_k = first_k
        logger.debug(f"self.first_k: {self.first_k}")

        if self.first_k == 0:
            # we got an empty end id list
            logger.debug("Got an empty end id list, will yield None")
            yield None, ("", datetime.now())

        async def _put_message_to_rxq_inner():
            logger.debug("Created task for recv_fifo in put_msg_to_rxq_inner")
            _ = asyncio.create_task(self._streamer_for_recv_fifo(end_ids))

        async def _get_message_inner():
            logger.debug("In _get_msg_inner(), will await until getting a message")
            return await self._rx_queue.get()

        # first, create an asyncio task to fetch messages and put a
        # temp queue _put_message_to_rxq_inner works as if it is a
        # non-blocking call because a task is created within it
        _, _ = run_async(_put_message_to_rxq_inner(), self._backend.loop())

        # the _get_message_inner() coroutine fetches a message from
        # the temp queue; we call this coroutine first_k times
        for _ in range(first_k):
            result, status = run_async(_get_message_inner(), self._backend.loop())
            logger.debug(f"After getting message, status: {status}")
            (end_id, payload) = result
            logger.debug(f"get payload for {end_id}")

            if self.has(end_id):
                logger.debug(f"channel got a msg for {end_id}")
                # set a property to indicate that a message was
                # received for the end
                self._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
            else:
                logger.debug(f"channel {self._name} has no end id {end_id} for msg")

            msg, timestamp = (
                (cloudpickle.loads(payload[0]), payload[1])
                if payload and status
                else (None, None)
            )
            metadata = (end_id, timestamp)

            if msg is not None:
                if MessageType.MODEL_VERSION in msg:
                    logger.debug(f"msg of type MODEL_VERSION recvd for end {end_id}")
                elif MessageType.HEARTBEAT in msg:
                    logger.debug(f"msg of type HEARTBEAT recvd for end {end_id}")
                    # TODO: (DG) Check if it helps here- can reset
                    # ends state to VAL_END_STATE_HEARTBEAT
                else:
                    logger.debug(f"msg of type UNKNOWN recvd for end {end_id}")
            else:
                # TODO: (DG) It comes here even for channel leave
                # notifications. Need a cleaner processing for it
                # later
                logger.warning(
                    f"Tried to populate None message (maybe leave notification) from end_id {end_id}. Will not yield msg and metadata until it gets a valid MODEL_VERSION msg"
                )

            # set cleanup ready event
            self._backend.set_cleanup_ready(end_id)

            yield msg, metadata

    async def _streamer_for_recv_fifo(self, end_ids: list[str]):
        """Read messages in a FIFO fashion.

        This method reads messages from queues associated with each
        end and puts first_k number of the messages into a queue; The
        remaining messages are saved back into a variable (peek_buf)
        of their corresponding end so that they can be read later.
        """

        async def _get_inner(end_id) -> tuple[str, Any]:
            if not self.has(end_id):
                # can't receive message from end_id
                logger.debug(f"Cannot receive message from end_id {end_id}")
                yield end_id, None

            payload = None
            try:
                logger.debug(
                    f"channel {self._name} awaiting get() on end_id {end_id} in self.ends"
                )
                payload = await self._ends[end_id].get()
                if payload:
                    # ignore timestamp for measuring bytes received
                    self.mc.accumulate("bytes", "recv", len(payload[0]))
            except KeyError:
                yield end_id, None

            logger.debug(f"_get_inner() invoked for end_id: {end_id}")
            yield end_id, payload

        runs = []
        for end_id in end_ids:
            if not self.has(end_id) or end_id in self._active_recv_fifo_tasks:
                continue
            else:
                runs.append(_get_inner(end_id))
                self._active_recv_fifo_tasks.add(end_id)

                logger.debug(
                    f"active task added for {end_id}, runs length: {len(runs)}"
                )
                logger.debug(
                    f"self._active_recv_fifo_tasks: {str(self._active_recv_fifo_tasks)}"
                )

        merged = stream.merge(*runs)
        async with merged.stream() as streamer:
            async for result in streamer:
                (end_id, _) = result

                await self._rx_queue.put(result)
                self._active_recv_fifo_tasks.remove(end_id)
                logger.debug(f"active task removed for {end_id}")

    def peek(self, end_id):
        """Peek rxq of end_id and return data if queue is not
        empty."""

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
            msg_delete_count = 0
            while True:
                msg, _ = self.peek(end_id)
                if not msg:
                    break

                # drain message from end so that cleanup ready event
                # is set
                _ = self.recv(end_id)
                msg_delete_count += 1
            logger.debug(f"Drained {msg_delete_count} messages from end_id {end_id}")

    def join(self):
        """Join the channel."""
        logger.info(f"calling channel join for {self._name}")

        self._backend.join(self)

    def leave(self):
        """Clean up resources allocated in the channel and leave
        it."""
        logger.debug(f"calling channel leave for {self._name}")

        self.drain_messages()

        self._backend.leave(self)

        logger.info(f"channel leave done for {self._name}")

    def update_trainer_state(self, state: TrainerAvailState, timestamp: str):
        """Update the state of an end in the channel."""
        logger.info(f"calling channel update state for {self._name}")

        self._backend.update_trainer_state(self, state, timestamp)

    def await_join(self, timeout=None) -> bool:
        """Wait for at least one peer joins a channel.

        If timeout value is set, it will wait until timeout occurs.
        Returns a boolean value to indicate whether timeout occurred
        or not.

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
        logger.info(f"timeouted = {timeouted}")
        return timeouted

    def is_rxq_empty(self, end_id: str) -> bool:
        """Return true if rxq is empty; otherwise, false."""
        return self._ends[end_id].is_rxq_empty()

    def is_txq_empty(self, end_id: str) -> bool:
        """Return true if txq is empty; otherwise, false."""
        return self._ends[end_id].is_txq_empty()

    """
    ### The following are asyncio methods of backend loop ###
    Therefore, they must be called in the backend loop
    """

    async def add(self, end_id):
        """Add an end to the channel and allocate rx and tx queues for
        it."""
        if self.has(end_id):
            return

        self._ends[end_id] = End(end_id)

        logger.debug(f"Adding end {end_id} to channel {self._name}")

        # create tx task in the backend for the channel
        self._backend.create_tx_task(self._name, end_id)

        # set the event true it's okay to call set() without checking
        # its condition
        self.await_join_event.set()

        # Set END_LAST_AVAIL_TS to current timestamp.
        current_ts = datetime.now()
        self.set_end_property(end_id=end_id, key=END_LAST_AVAIL_TS, value=current_ts)

        # NOTE: Also set in end_state_info for further use
        if end_id not in self._end_state_info.keys():
            self._end_state_info[end_id] = {}
        self._end_state_info[end_id][END_LAST_AVAIL_TS] = current_ts

        end_last_avail_ts = self.get_end_property(end_id=end_id, key=END_LAST_AVAIL_TS)
        logger.debug(
            f"Get after setting END_LAST_AVAIL_TS was {end_last_avail_ts} "
            f"for end_id {end_id}"
        )

        # Check if END_LAST_UNAVAIL_TS is None. If it is, skip adding
        # TOTAL_UNAVAIL_DURATION. If its not None, update the
        # TOTAL_UNAVAIL_DURATION.
        # NOTE: Since end_id was deleted from _ends, get its state
        # info from end_state_info which persists information across
        # connections and disconnections

        if END_LAST_UNAVAIL_TS in self._end_state_info[end_id].keys():
            end_last_unavail_ts = self._end_state_info[end_id][END_LAST_UNAVAIL_TS]
        else:
            end_last_unavail_ts = None
        logger.debug(
            f"Got END_LAST_UNAVAIL_TS: {end_last_unavail_ts} " f"for end_id {end_id}"
        )

        if end_last_unavail_ts is not None:
            logger.debug(f"END_LAST_UNAVAIL_TS was not None for end_id {end_id}")
            # subtraction of two datetime objects returns a timedelta
            last_time_unavail_duration = datetime.now() - end_last_unavail_ts

            # Get total unavail duration from end_state_info
            if PROP_TOTAL_UNAVAIL_DURATION in self._end_state_info[end_id].keys():
                elapsed_end_unavail_duration = self._end_state_info[end_id][
                    PROP_TOTAL_UNAVAIL_DURATION
                ]
            else:
                elapsed_end_unavail_duration = None

            if elapsed_end_unavail_duration is None:
                elapsed_end_unavail_duration = timedelta(
                    hours=0, minutes=0, seconds=0, microseconds=0
                )
            total_end_unavail_duration = (
                elapsed_end_unavail_duration + last_time_unavail_duration
            )

            # set as end_id property here to be used in asyncoort
            # selector later.
            self.set_end_property(
                end_id=end_id,
                key=PROP_TOTAL_UNAVAIL_DURATION,
                value=total_end_unavail_duration,
            )
            # NOTE: Also keeping it updated in end_node_info
            self._end_state_info[end_id][
                PROP_TOTAL_UNAVAIL_DURATION
            ] = total_end_unavail_duration
            logger.debug(
                f"Updated total unavail duration to "
                f"{total_end_unavail_duration} for end_id {end_id}. Added "
                f"{last_time_unavail_duration} to elapsed unavail duration "
                f"{elapsed_end_unavail_duration}"
            )
        else:
            logger.debug(
                f"None got as END_LAST_UNAVAIL_TS: {end_last_unavail_ts} "
                f"for end_id {end_id}"
            )

    async def remove(self, end_id):
        """Remove an end from the channel."""
        logger.debug(f"Removing end {end_id} from channel {self._name}")
        if not self.has(end_id):
            logger.debug(
                f"Noting to remove since end {end_id} not in channel {self._name}"
            )
            return

        rxq = self._ends[end_id].get_rxq()
        txq = self._ends[end_id].get_txq()
        del self._ends[end_id]

        # put bogus data to unblock a get() call
        await rxq.put(EMPTY_PAYLOAD)

        # put bogus data to let tx_task finish
        await txq.put(EMPTY_PAYLOAD)

        if len(self._ends) == 0:
            # clear (or unset) the event
            self.await_join_event.clear()

        # NOTE: USE end_state_info since end has been deleted from
        # _ends and no properties can be get OR set.

        # Set END_LAST_UNAVAIL_TS to current timestamp.
        self._end_state_info[end_id][END_LAST_UNAVAIL_TS] = datetime.now()
        end_last_unavail_ts = self._end_state_info[end_id][END_LAST_UNAVAIL_TS]
        logger.debug(
            f"Updated END_LAST_UNAVAIL_TS to {end_last_unavail_ts} "
            f"for end_id {end_id} in end_state_info"
        )

        # Check if END_LAST_AVAIL_TS is None. If it is, skip adding
        # TOTAL_AVAIL_DURATION. If its not None, update the
        # TOTAL_AVAIL_DURATION.
        end_last_avail_ts = self._end_state_info[end_id][END_LAST_AVAIL_TS]
        logger.debug(
            f"Got END_LAST_AVAIL_TS: {end_last_avail_ts} "
            f"for end_id {end_id} in end_state_info"
        )
        if end_last_avail_ts is not None:
            logger.debug(f"END_LAST_AVAIL_TS was not None for end_id {end_id}")
            last_time_avail_duration = datetime.now() - end_last_avail_ts
            if PROP_TOTAL_AVAIL_DURATION in self._end_state_info[end_id].keys():
                elapsed_end_avail_duration = self._end_state_info[end_id][
                    PROP_TOTAL_AVAIL_DURATION
                ]
            else:
                elapsed_end_avail_duration = None

            if elapsed_end_avail_duration is None:
                elapsed_end_avail_duration = timedelta(
                    hours=0, minutes=0, seconds=0, microseconds=0
                )
            total_end_avail_duration = (
                elapsed_end_avail_duration + last_time_avail_duration
            )
            self._end_state_info[end_id][
                PROP_TOTAL_AVAIL_DURATION
            ] = total_end_avail_duration
            logger.debug(
                f"Updated total unavail duration to "
                f"{total_end_avail_duration} for end_id {end_id}. Added "
                f"{last_time_avail_duration} to elapsed unavail duration "
                f"{elapsed_end_avail_duration}"
            )
        else:
            logger.debug(
                f"None got as END_LAST_AVAIL_TS: {end_last_avail_ts} "
                f"for end_id {end_id}"
            )

        # inform selector to cleanup its send/recieve state to allow
        # quicker addition next time it joins
        logger.debug(
            "Also removing existing trainer update send/recv state from selector"
        )
        self._selector._cleanup_removed_ends(end_id)

    async def update_state(
        self, end_id: str, new_state: TrainerAvailState, timestamp: str
    ):
        """Update the state of an end in the channel."""
        if not self.has(end_id):
            logger.info(f"End {end_id} not in channel {self._name}")
            return

        old_end_state = self._ends[end_id].get_property(PROP_END_AVL_STATE)
        self._ends[end_id].set_property(PROP_END_AVL_STATE, new_state)
        logger.debug(
            f"Updated new_state of end {end_id} in channel {self._name} to state: {self._ends[end_id].get_property(PROP_END_AVL_STATE)} from timestamp: {timestamp}"
        )

        # If new updated_state is UN_AVL, reset end state to unblock
        # any train or eval tasks sent to that trainer
        if new_state == TrainerAvailState.UN_AVL:
            logger.debug(
                f"Since new_state for trainer {end_id} is {new_state}, will remove from selected and all_selected"
            )
            self._selector.remove_from_selected_ends(self._ends, end_id)
            self._selector._cleanup_removed_ends(end_id)
        elif (
            old_end_state == TrainerAvailState.AVL_TRAIN
            and new_state == TrainerAvailState.AVL_EVAL
        ):
            # TODO: (DG) This is a temporary fix to reset the state of
            # a trainer from AVL_TRAIN to AVL_EVAL. Check actual last
            # task sent before resetting.
            logger.debug(
                f"Trainer {end_id} moved from state {old_end_state} to {new_state}, removing the end from selected and all_selected. If any train tasks are pending, they should be re-set, but not needed for pending eval tasks. Doing it anyway for now."
            )
            self._selector.remove_from_selected_ends(self._ends, end_id)
            self._selector._cleanup_removed_ends(end_id)

        # set cleanup ready event
        self._backend.set_cleanup_ready(end_id)

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

    def set_curr_unavailable_trainers(self, trainer_unavail_list: list):
        self.trainer_unavail_list = trainer_unavail_list
