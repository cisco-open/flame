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


import asyncio
import uuid

from ..common.comm import _recv_msg, _send_msg
from ..common.constants import DEFAULT_RUN_ASYNC_WAIT_TIME, BackendEvent
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from .abstract import AbstractBackend

UNIX_SOCKET_PATH_TEMPLATE = '/tmp/flame-{}.socket'


class LocalBackend(AbstractBackend):
    '''
    LocalBackend only allows a singleton instance
    '''
    _instance = None

    _initialized = False
    _endpoints = None
    _channels = None
    # inform channel manager of events occurring at backend
    _eventq = None

    _loop = None
    _thread = None
    _server = None

    _id = None
    _backend = None

    def __new__(cls):
        if cls._instance is None:
            print('creating a LocalBackend instance')
            cls._instance = super(LocalBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._endpoints = {}
        self._channels = {}
        self._id = str(uuid.uuid1())

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

            coro = self._monitor_connections()
            _ = asyncio.create_task(coro)

            coro = self._setup_server()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _ = run_async(coro, self._loop, 1)

        self._initialized = True

    async def _monitor_connections(self):
        while True:
            for end_id, (reader, _) in list(self._endpoints.items()):
                if not reader.at_eof():
                    continue

                # connection is closed
                await self._close(end_id)

            await asyncio.sleep(1)

    async def _register_end(self, reader, writer):
        any_msg = await _recv_msg(reader)

        if not any_msg.Is(msg_pb2.Connect.DESCRIPTOR):
            # not expected message
            await self.__close(writer)
            return

        conn_msg = msg_pb2.Connect()
        any_msg.Unpack(conn_msg)

        print(f'registering end: {conn_msg.end_id}')
        self._endpoints[conn_msg.end_id] = (reader, writer)

    async def _handle_msg(self, reader):
        any_msg = await _recv_msg(reader)

        if any_msg is None:
            return

        if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
            msg = msg_pb2.Notify()
            any_msg.Unpack(msg)

            if msg.channel_name not in self._channels:
                return

            channel = self._channels[msg.channel_name]
            await channel.add(msg.end_id)

        elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
            msg = msg_pb2.Data()
            any_msg.Unpack(msg)

            channel = self._channels[msg.channel_name]
            rxq = channel.get_rxq(msg.end_id)
            await rxq.put(msg.payload)

        else:
            print('unknown message')

    async def _handle(self, reader, writer):
        await self._register_end(reader, writer)

        # create a coroutine task
        coro = self._rx_task(reader, writer)
        _ = asyncio.create_task(coro)

    async def _setup_server(self):
        self._backend = UNIX_SOCKET_PATH_TEMPLATE.format(self._id)
        self._server = await asyncio.start_unix_server(self._handle,
                                                       path=self._backend)

        name = self._server.sockets[0].getsockname()
        print(f'serving on {name}')

        async with self._server:
            await self._server.serve_forever()

    async def _connect(self, endpoint):
        reader, writer = await asyncio.open_unix_connection(f'{endpoint}')

        # register backend id to end
        msg = msg_pb2.Connect()
        msg.end_id = self._id

        await _send_msg(writer, msg)

        # create a coroutine task
        coro = self._rx_task(reader, writer)
        _ = asyncio.create_task(coro)

        return reader, writer

    async def _send_data(self, writer, channel_name, data):
        msg = msg_pb2.Data()
        msg.end_id = self._id
        msg.channel_name = channel_name
        msg.payload = data

        await _send_msg(writer, msg)

    async def _put_txq(self, channel, end_id, data):
        await channel._txq.put((end_id, data))

    async def __close(self, writer):
        if writer.is_closing():
            return

        writer.close()
        await writer.wait_closed()

    async def _close(self, end_id):
        if end_id not in self._endpoints:
            return

        try:
            _, writer = self._endpoints[end_id]
            await self.__close(writer)
        finally:
            del self._endpoints[end_id]

            await self._eventq.put((BackendEvent.DISCONNECT, end_id))
            print(f'connection to {end_id} closed')

    def uid(self):
        '''
        return backend id
        '''
        return self._id

    def endpoint(self):
        return self._backend

    def eventq(self):
        return self._eventq

    def connect(self, end_id, endpoint):
        if end_id in self._endpoints:
            return True

        coro = self._connect(endpoint)
        result, status = run_async(coro, self._loop,
                                   DEFAULT_RUN_ASYNC_WAIT_TIME)
        (reader, writer) = result
        if status:
            self._endpoints[end_id] = (reader, writer)

        return status

    def notify(self, end_id, channel_name):
        if end_id not in self._endpoints:
            return False

        _, writer = self._endpoints[end_id]

        async def _notify():
            msg = msg_pb2.Notify()
            msg.end_id = self._id
            msg.channel_name = channel_name

            await _send_msg(writer, msg)

        _, status = run_async(_notify(), self._loop,
                              DEFAULT_RUN_ASYNC_WAIT_TIME)

        return status

    def close(self):
        for end_id in list(self._endpoints):
            coro = self._close(end_id)
            _ = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)

    def loop(self):
        return self._loop

    def attach_channel(self, channel):
        self._channels[channel.name()] = channel

    def create_tx_task(self, channel_name, end_id):
        if (channel_name not in self._channels
                or not self._channels[channel_name].has(end_id)):
            return False

        channel = self._channels[channel_name]

        # if in_active_loop:
        coro = self._tx_task(channel, end_id)
        _ = asyncio.create_task(coro)

        return True

    async def _rx_task(self, reader, writer):
        '''
        _rx_task() must be created upon connect or accept
        '''
        while not reader.at_eof():  # while connection is alive
            await self._handle_msg(reader)

        await self.__close(writer)

    async def _tx_task(self, channel, end_id):
        '''
        _tx_task() must be created per tx queue right after end_id is added to
        channel (e.g., channel.add(end_id))
        '''
        name = channel.name()
        txq = channel.get_txq(end_id)

        reader, writer = self._endpoints[end_id]

        while not reader.at_eof():  # while connection is alive
            data = await txq.get()
            await self._send_data(writer, name, data)
            txq.task_done()

        await self._close(end_id)
