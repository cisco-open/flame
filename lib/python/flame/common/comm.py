# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct

from google.protobuf.any_pb2 import Any

from .constants import MSG_LEN_FIELD_SIZE


async def _recv_msg(reader):
    data = await reader.read(MSG_LEN_FIELD_SIZE)
    if len(data) == 0:
        return None

    msg_len = struct.unpack('>I', data)[0]

    data = await reader.readexactly(msg_len)
    any = Any().FromString(data)

    return any


async def _send_msg(writer, msg):
    any = Any()
    any.Pack(msg)

    data = any.SerializeToString()
    data = struct.pack('>I', len(data)) + data

    writer.write(data)

    try:
        await writer.drain()
    except ConnectionResetError:
        # connection is reset; let the caller detect it via reader and cleanup
        pass
