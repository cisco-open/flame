import struct

from google.protobuf.any_pb2 import Any

from .constants import HEADER_LEN


async def _recv_msg(reader):
    data = await reader.read(HEADER_LEN)
    if len(data) == 0:
        return None

    msg_len = struct.unpack('>I', data)[0]

    data = await reader.read(msg_len)
    any = Any().FromString(data)

    return any


async def _send_msg(writer, msg):
    any = Any()
    any.Pack(msg)

    data = any.SerializeToString()
    data = struct.pack('>I', len(data)) + data

    writer.write(data)
    await writer.drain()
