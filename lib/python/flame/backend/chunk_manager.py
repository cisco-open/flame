# Copyright 2023 Cisco Systems, Inc. and its affiliates
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
"""Chunk Manager."""

import logging
from datetime import datetime
from queue import Empty, Queue
from threading import Thread

from flame.backend.chunk_store import ChunkStore
from flame.channel import Channel
from flame.common.util import run_async
from flame.proto import backend_msg_pb2 as msg_pb2

logger = logging.getLogger(__name__)

KEY_BACKEND = "backend"
KEY_CHANNEL = "channel"
KEY_END_ID = "end_id"

QUEUE_TIMEOUT = 5  # 5 seconds


class ChunkThread(Thread):
    """ChunkThread class."""

    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ):
        """Initialize an instance."""
        # call parent constructure method
        super().__init__(group, target, name, daemon=daemon)

        self._backend = kwargs[KEY_BACKEND]
        self._channel = kwargs[KEY_CHANNEL]
        self._end_id = kwargs[KEY_END_ID]

        self.queue = Queue()
        self.chunk_store = ChunkStore()
        self._done = False

    def stop(self):
        """Set a flag to stop the thread."""
        self._done = True

    def insert(self, msg: msg_pb2.Data) -> None:
        """Put a message into queue in the chunk thread."""
        self.queue.put(msg)

    def run(self):
        """Override run function of Thread.

        The function assembles chunks into a full-size message and passes
        the message to its designated receive queue.
        """

        async def inner(end_id: str, data: bytes, timestamp: datetime):
            logger.debug(f"fully assembled data size = {len(data)}")

            rxq = self._channel.get_rxq(end_id)
            if rxq is None:
                logger.debug(f"rxq not found for {end_id}")

                # set cleanup ready event for a given end id
                await self._backend.set_cleanup_ready_async(end_id)
                return

            await rxq.put((data, timestamp))

        while not self._done:
            try:
                msg = self.queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                continue

            timestamp = datetime.now()

            # assemble is done in a chunk thread so that it won't block
            # asyncio task
            status = self.chunk_store.assemble(msg)
            if not status:
                # reset chunk_store if message is wrong
                self.chunk_store.reset()

                # set cleanup ready event for a given end id
                self._backend.set_cleanup_ready(msg.end_id)
            else:
                if not self.chunk_store.eom:
                    # not an end of message, hence, can't get a payload
                    # out of chunk store yet

                    # set cleanup ready event for a given end id
                    self._backend.set_cleanup_ready(msg.end_id)
                    continue

                payload = self.chunk_store.get_data()
                # now push payload to a target receive queue.
                _, status = run_async(
                    inner(msg.end_id, payload, timestamp), self._backend.loop()
                )

                # message was completely assembled, reset the chunk store
                self.chunk_store.reset()

        logger.debug(f"finished chunk thread for {self._end_id}")


class ChunkManager(object):
    """ChunkStore class."""

    def __init__(self, backend):
        """Initialize an instance."""
        self._backend = backend
        self._chunk_threads: dict[str, ChunkThread] = {}

    def handle(self, msg: msg_pb2.Data, channel: Channel) -> None:
        """Process msg."""
        if msg.end_id not in self._chunk_threads:
            kwargs = {
                KEY_BACKEND: self._backend,
                KEY_CHANNEL: channel,
                KEY_END_ID: msg.end_id,
            }
            chunk_thd = ChunkThread(kwargs=kwargs, daemon=True)
            self._chunk_threads[msg.end_id] = chunk_thd
            chunk_thd.start()

        chunk_thd = self._chunk_threads[msg.end_id]
        chunk_thd.insert(msg)

    def stop(self, end_id):
        """Stop chunk thread associated with end id."""
        if end_id in self._chunk_threads:
            chunk_thd = self._chunk_threads[end_id]
            chunk_thd.stop()
            del self._chunk_threads[end_id]
