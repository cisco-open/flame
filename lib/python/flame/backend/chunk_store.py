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
"""ChunkStore."""

import logging
from threading import Thread
from typing import Tuple, Union

from ..common.constants import EMPTY_PAYLOAD
from ..common.util import run_async
from ..proto import backend_msg_pb2 as msg_pb2

DEFAULT_CHUNK_SIZE = 1048576  # 1MB

logger = logging.getLogger(__name__)


class ChunkStore(object):
    """ChunkStore class."""

    def __init__(self, loop=None, channel=None):
        """Initialize an instance."""
        self._loop = loop
        self._channel = channel

        # for fragment
        self.pidx = 0
        self.cidx = DEFAULT_CHUNK_SIZE

        # for assemble
        self.recv_buf = list()

        # for both fragment and assemble
        self.data = b''
        self.seqno = -1
        self.eom = False

    def set_data(self, data: bytes) -> None:
        """Set data in chunk store."""
        logger.debug(f"setting data of size {len(data)}")
        self.data = data

        # reset variables since new data is set
        self.pidx = 0
        self.cidx = DEFAULT_CHUNK_SIZE

    def get_chunk(self) -> Tuple[Union[bytes, None], int, bool]:
        """
        Return a chunk of data.

        The method returns None as the first part of the triplet
        if its internal index is pointing beyond the end of data.
        Otherwise, it returns a chunk every time it is called.
        """
        data_len = len(self.data)

        # no more chunk
        if self.pidx >= data_len:
            return None, -1, False

        if self.cidx < data_len:
            data = self.data[self.pidx:self.cidx]
            eom = False
        else:
            data = self.data[self.pidx:]
            eom = True

        seqno = self.seqno + 1
        self.seqno = seqno
        self.pidx = self.cidx
        self.cidx += DEFAULT_CHUNK_SIZE

        logger.debug(f"chunk {seqno}: {len(data)}")
        return data, seqno, eom

    def assemble(self, msg: msg_pb2.Data) -> bool:
        """Assemble message.

        This method pushes message payload into a receive buffer.
        If eom (end of message) is set, bytes in the array are joined.
        Then, the assembled data will be put into a receive queue.

        The join operation can be exepnsive if the data size is large.
        We run the join operation in a separate thread in order to unblock
        asyncio tasks as quickly as possible.
        """
        # out of order delivery
        if self.seqno + 1 != msg.seqno:
            logger.warning(f'out-of-order seqno from {msg.end_id}')
            return False

        logger.debug(f"chunk {msg.seqno}: {len(msg.payload)}")
        # add payload to a recv buf
        self.recv_buf.append(msg.payload)
        self.seqno = msg.seqno
        self.eom = msg.eom

        if self.eom:
            # we assemble payloads in the recv buf array
            # only if eom is set to True.
            # In this way, we only pay byte concatenation cost once
            _thread = Thread(target=self._assemble,
                             args=(msg.end_id, ),
                             daemon=True)
            _thread.start()

        return True

    def _assemble(self, end_id: str) -> None:
        # This code must be executed in a separate thread
        self.data = EMPTY_PAYLOAD.join(self.recv_buf)

        async def inner():
            logger.debug(f'fully assembled data size = {len(self.data)}')

            rxq = self._channel.get_rxq(end_id)
            if rxq is None:
                logger.debug(f"rxq not found for {end_id}")
                return

            await rxq.put(self.data)

        _, status = run_async(inner(), self._loop)
