import logging
from typing import Tuple, Union

from ..proto import backend_msg_pb2 as msg_pb2

DEFAULT_CHUNK_SIZE = 4194304  # 4MB

logger = logging.getLogger(__name__)


class ChunkStore(object):
    def __init__(self):
        # for fragment
        self.pidx = 0
        self.cidx = DEFAULT_CHUNK_SIZE

        # for both fragment and assemble
        self.data = b''
        self.seqno = -1
        self.eom = False

    def set_data(self, data: bytes) -> None:
        self.data = data

        # reset variables since new data is set
        self.pidx = 0
        self.cidx = DEFAULT_CHUNK_SIZE

    def get_chunk(self) -> Tuple[Union[bytes, None], int, bool]:
        """
        get_chunk() returns None as the first part of the triplet
        if its internal index is pointing beyond the end of message.
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

        return data, seqno, eom

    def assemble(self, msg: msg_pb2.Data) -> bool:
        # out of order delivery
        if self.seqno + 1 != msg.seqno:
            logger.warning(f'out-of-order seqno from {msg.end_id}')
            return False

        self.data += msg.payload
        self.seqno = msg.seqno
        self.eom = msg.eom

        return True
