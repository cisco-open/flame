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
from typing import Tuple, Union

from ..common.constants import EMPTY_PAYLOAD
from ..proto import backend_msg_pb2 as msg_pb2

DEFAULT_CHUNK_SIZE = 1048576  # 1MB

logger = logging.getLogger(__name__)


class ChunkStore(object):
    """ChunkStore class."""

    def __init__(self):
        """Initialize an instance."""
        self.reset()

    def reset(self):
        """Reset the state of chunk store."""
        # for fragment
        self.pidx = 0
        self.cidx = DEFAULT_CHUNK_SIZE

        # for assemble
        self.recv_buf = list()

        # for both fragment and assemble
        self.data = EMPTY_PAYLOAD
        self.seqno = -1
        self.eom = False

    def get_data(self) -> Union[bytes, None]:
        """Return data."""
        return self.data

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
            self.data = EMPTY_PAYLOAD.join(self.recv_buf)

        return True
