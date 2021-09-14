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

import asyncio

from ..common.comm import _send_msg
from ..proto import backend_msg_pb2 as msg_pb2
from .local import LocalBackend

# TODO: BACKEND_PORT may need to be updated or configurable
BACKEND_PORT = 20000
LISTENING_IP = '0.0.0.0'
ENDPOINT_TOKEN_LEN = 2


class PointToPointBackend(LocalBackend):
    async def _setup_server(self):
        self._server = await asyncio.start_server(
            self._handle, LISTENING_IP, BACKEND_PORT
        )

        self._backend = f'{LISTENING_IP}:{BACKEND_PORT}'

        name = self._server.sockets[0].getsockname()
        print(f'serving on {name}')

        async with self._server:
            await self._server.serve_forever()

    async def _connect(self, endpoint):
        tokens = endpoint.split(':')
        if len(tokens) != ENDPOINT_TOKEN_LEN:
            return None, None

        addr = tokens[0]
        port = int(tokens[1])
        reader, writer = await asyncio.open_connection(addr, port)

        # register backend id to end
        msg = msg_pb2.Connect()
        msg.end_id = self._id

        await _send_msg(writer, msg)

        # create a coroutine task
        coro = self._rx_task(reader, writer)
        _ = asyncio.create_task(coro)

        return reader, writer
