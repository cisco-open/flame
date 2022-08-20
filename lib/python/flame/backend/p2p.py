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
"""PointToPoint Backend."""

import asyncio
import logging
import socket
from typing import AsyncIterable, Iterable, Tuple

import grpc

from ..common.constants import (DEFAULT_RUN_ASYNC_WAIT_TIME, BackendEvent,
                                CommType)
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from ..proto import backend_msg_pb2_grpc as msg_pb2_grpc
from ..proto import meta_pb2, meta_pb2_grpc
from .abstract import AbstractBackend
from .chunk_store import ChunkStore

logger = logging.getLogger(__name__)

ENDPOINT_TOKEN_LEN = 2
HEART_BEAT_DURATION = 30


class BackendServicer(msg_pb2_grpc.BackendRouteServicer):
    """Implements functionallity of backend route server."""

    def __init__(self, p2pbe) -> None:
        """Initialize."""
        self.p2pbe = p2pbe

    async def notify_end(self, req: msg_pb2.Notify,
                         unused_context) -> msg_pb2.Notify:
        """Implement a method to handle Notification request."""
        return await self.p2pbe._handle_notification(req)

    async def send_data(self, req_iter: AsyncIterable[msg_pb2.Data],
                        unused_context) -> msg_pb2.BackendID:
        """Implement a method to handle send_data request stream."""
        # From server perspective, the server receives data from client.
        logger.warn(f"address of req_iter = {hex(id(req_iter))}")
        logger.warn(f"address of context = {hex(id(unused_context))}")

        async for msg in req_iter:
            await self.p2pbe._handle_data(msg)

        return msg_pb2.BackendID(end_id=self.p2pbe._id)

    async def recv_data(self, req: msg_pb2.BackendID,
                        context) -> AsyncIterable[msg_pb2.Data]:
        """Implement a method to handle recv_data request."""
        # From server perspective, the server sends data to client.
        dck_task = asyncio.create_task(self._dummy_context_keeper(context))
        self.p2pbe._set_writer(req.end_id, context)

        await dck_task

    async def _dummy_context_keeper(self, context):
        """Sleep 1000 sec (an abitrary big value) to keep context alive."""
        while True:
            await asyncio.sleep(1000)


class PointToPointBackend(AbstractBackend):
    """PointToPointBackend class.

    PointToPoint backend is EXPERIMENTAL.
    """

    def __init__(self):
        """Initialize an instance."""
        self._instance = None
        self._initialized = False
        # queue to inform channel manager of backend's events
        self._eventq = None
        self._loop = None

        # variables for functionality
        self._id = None
        self._channels = None
        self._broker = None
        self._msg_chunks = None
        if self._initialized:
            return

        self._backend = None

        self._endpoints = {}
        self.end_to_rwop = {}
        self._channels = {}

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

            coro = self._monitor_end_termination()
            _ = asyncio.create_task(coro)

            coro = self._setup_server()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)

        if not success:
            raise SystemError('initialization failure')

        self._initialized = True

    async def _monitor_end_termination(self):
        # TODO: handle how to monitor grpc channel status
        # while True:
        #     for end_id, (reader, _) in list(self._endpoints.items()):
        #         if not reader.at_eof():
        #             continue

        #         # connection is closed
        #         await self._eventq.put((BackendEvent.DISCONNECT, end_id))
        #         await self._close(end_id)

        #     await asyncio.sleep(1)
        pass

    async def _setup_server(self):
        server = grpc.aio.server()
        msg_pb2_grpc.add_BackendRouteServicer_to_server(
            BackendServicer(self), server)

        ip_addr = socket.gethostbyname(socket.gethostname())
        port = server.add_insecure_port(f'{ip_addr}:0')

        self._backend = f'{ip_addr}:{port}'
        logger.info(f'serving on {self._backend}')
        await server.start()
        await server.wait_for_termination()

    def configure(self, broker: str, job_id: str, task_id: str):
        """Configure the backend."""
        self._msg_chunks: dict[str, ChunkStore] = {}

        self._broker = broker
        self._job_id = job_id
        self._id = task_id

    def eventq(self):
        """Return a event queue object."""
        return self._eventq

    def loop(self):
        """Return loop instance of asyncio."""
        return self._loop

    def uid(self):
        """Return backend id."""
        return self._id

    def join(self, channel) -> None:
        """Join a channel."""

        async def _join_inner():
            coro = self._register_channel(channel)
            _ = asyncio.create_task(coro)

        coro = _join_inner()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not success:
            raise SystemError('join failure')

    def create_tx_task(self,
                       channel_name: str,
                       end_id: str,
                       comm_type=CommType.UNICAST) -> bool:
        """Create asyncio task for transmission."""
        if (channel_name not in self._channels
                or (not self._channels[channel_name].has(end_id)
                    and comm_type != CommType.BROADCAST)):
            return False

        channel = self._channels[channel_name]

        coro = self._tx_task(channel, end_id, comm_type)
        _ = asyncio.create_task(coro)

    def attach_channel(self, channel) -> None:
        """Attach a channel to backend."""
        self._channels[channel.name()] = channel

    async def _register_channel(self, channel) -> None:
        logger.info("calling _register_channel")
        with grpc.insecure_channel(self._broker) as grpc_channel:
            stub = meta_pb2_grpc.MetaRouteStub(grpc_channel)
            meta_info = meta_pb2.MetaInfo(
                job_id=channel.job_id(),
                ch_name=channel.name(),
                me=channel.my_role(),
                other=channel.other_role(),
                endpoint=self._backend,
            )

            meta_resp = stub.RegisterMetaInfo(meta_info)
            if meta_resp:
                meta_pb2_success = meta_pb2.MetaResponse.Status.SUCCESS
                if meta_resp.status != meta_pb2_success:
                    logger.debug("registration failed")
                    raise SystemError('registration failure')

                for endpoint in meta_resp.endpoints:
                    logger.info(f"endpoint: {endpoint}")
                    await self._connect_and_notify(endpoint, channel.name())

            while True:
                meta_resp = stub.HeartBeat(meta_info)
                logger.debug(f"meta_resp from heart beat: {meta_resp}")
                await asyncio.sleep(HEART_BEAT_DURATION)

    async def _connect_and_notify(self, endpoint: str, ch_name: str) -> None:
        grpc_ch = grpc.aio.insecure_channel(endpoint)
        stub = msg_pb2_grpc.BackendRouteStub(grpc_ch)

        await self.notify(ch_name, msg_pb2.NotifyType.JOIN, stub, grpc_ch)

    async def notify(self, channel_name, notify_type, stub, grpc_ch) -> bool:
        """Send a notify message to an end by using stub."""
        if channel_name not in self._channels:
            logger.debug(f'channel {channel_name} not found')
            return False

        msg = msg_pb2.Notify(end_id=self._id,
                             channel_name=channel_name,
                             type=notify_type)

        try:
            resp = await stub.notify_end(msg)
        except grpc.aio.AioRpcError:
            logger.warn("can't proceed as grpc channel is unavailable")
            return False

        logger.debug(f"resp = {resp}")
        _ = await self._handle_notification(resp, stub, grpc_ch)

        return True

    async def _handle_notification(
            self,
            msg: msg_pb2.Notify,
            stub: msg_pb2_grpc.BackendRouteStub = None,
            grpc_ch: grpc.aio.Channel = None) -> msg_pb2.Notify:
        if msg.end_id == self._id:
            # This case happens when message is broadcast to a self-loop
            # e.g., distributed topology
            logger.debug('message sent to self; do nothing')
            return msg_pb2.Notify(type=msg_pb2.NotifyType.ACK)

        if msg.channel_name not in self._channels:
            logger.debug('channel not found')
            return msg_pb2.Notify(type=msg_pb2.NotifyType.ACK)

        channel = self._channels[msg.channel_name]

        if msg.type == msg_pb2.NotifyType.JOIN and not channel.has(msg.end_id):
            if stub is not None:
                reader = stub.recv_data(msg_pb2.BackendID(end_id=self._id))
                logger.debug(f"type of reader = {type(reader)}")
                # (w, x, y, z) - w: reader, x: writer for server (context)
                # y: writer for client (stub), z: a grpc channel
                # grpc channel is saved to prevent it from
                # being garbage-collected
                self._endpoints[msg.end_id] = (reader, None, stub, grpc_ch)
                _ = asyncio.create_task(self._rx_task(msg.end_id, reader))

            # add end to the channel
            await channel.add(msg.end_id)

            # this is the first time to see this end,
            # so let's notify my presence to the end
            logger.debug('acknowledge notification')
            return msg_pb2.Notify(end_id=self._id,
                                  channel_name=msg.channel_name,
                                  type=msg_pb2.NotifyType.JOIN)

        elif msg.type == msg_pb2.NotifyType.LEAVE:
            # it's sufficient to remove end from channel
            # no extra action (e.g., unsubscribe) needed
            await channel.remove(msg.end_id)

        return msg_pb2.Notify(type=msg_pb2.NotifyType.ACK)

    async def _handle_data(self, msg: msg_pb2.Data) -> None:
        if msg.end_id == self._id:
            # This case happens when message is broadcast to a self-loop
            # e.g., distributed topology
            logger.debug('message sent to self; do nothing')
            return

        payload, fully_assembled = self.assemble_chunks(msg)

        if fully_assembled:
            logger.debug(f'fully assembled data size = {len(payload)}')

            channel = self._channels[msg.channel_name]
            rxq = channel.get_rxq(msg.end_id)
            if rxq is None:
                logger.debug(f"rxq not found for {msg.end_id}")
                return

            await rxq.put(payload)

    def assemble_chunks(self, msg: msg_pb2.Data) -> Tuple[bytes, bool]:
        """Assemble message chunks to build a whole message."""
        if msg.end_id not in self._msg_chunks:
            self._msg_chunks[msg.end_id] = ChunkStore()

        chunk_store = self._msg_chunks[msg.end_id]
        if not chunk_store.assemble(msg):
            # clean up wrong message
            del self._msg_chunks[msg.end_id]
            return b'', False

        if chunk_store.eom:
            data = chunk_store.data
            del self._msg_chunks[msg.end_id]
            return data, True
        else:
            return b'', False

    async def _tx_task(self, channel, end_id, comm_type: CommType):
        """Conducts data transmission in a loop.

        _tx_task() must be created per tx queue right after end_id is added to
        channel (e.g., channel.add(end_id)).
        In case of a tx task for broadcast queue, a broadcaset queue must be
        created first.
        """
        if comm_type == CommType.BROADCAST:
            await self._broadcast_task(channel)
        else:
            await self._unicast_task(channel, end_id)

    async def _broadcast_task(self, channel):
        """Broadcast messages.

        p2p backend doesn't support broadcast natively.
        broadcast is simply a collection of unicast transmissions.
        """
        txq = channel.broadcast_q()

        while True:
            data = await txq.get()
            end_ids = list(channel._ends.keys())
            logger.debug(f"end ids for bcast = {end_ids}")
            for end_id in end_ids:
                try:
                    await self.send_chunks(end_id, channel.name(), data)
                except Exception as ex:
                    ex_name = type(ex).__name__
                    logger.warn(f"An exception of type {ex_name} occurred")

                    await self._eventq.put((BackendEvent.DISCONNECT, end_id))
                    del self._endpoints[end_id]
            txq.task_done()

    async def _unicast_task(self, channel, end_id):
        txq = channel.get_txq(end_id)

        while True:
            data = await txq.get()

            try:
                await self.send_chunks(end_id, channel.name(), data)
            except Exception as ex:
                ex_name = type(ex).__name__
                logger.warn(f"An exception of type {ex_name} occurred")

                await self._eventq.put((BackendEvent.DISCONNECT, end_id))
                del self._endpoints[end_id]
                txq.task_done()
                # This break ends a tx_task for end_id
                break

            txq.task_done()

        logger.warn(f"unicast task for {end_id} terminated")

    async def send_chunks(self, other: str, ch_name: str, data: bytes) -> None:
        """Send data chunks to an end."""
        _, svr_writer, clt_writer, _ = self._endpoints[other]

        # TODO: keep regenerating messages can be expensive; revisit this later
        if clt_writer is not None:
            await clt_writer.send_data(
                self._generate_data_messages(ch_name, data))
        elif svr_writer is not None:
            for msg in self._generate_data_messages(ch_name, data):
                logger.debug(f"svr writer sending msg {msg.seqno}")
                await svr_writer.write(msg)
        else:
            logger.debug("writer not found}")

    def _generate_data_messages(self, ch_name: str,
                                data: bytes) -> Iterable[msg_pb2.Data]:
        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            msg = msg_pb2.Data(end_id=self._id,
                               channel_name=ch_name,
                               payload=data,
                               seqno=seqno,
                               eom=eom)

            yield msg

    def _set_writer(self, end_id: str,
                    context: grpc.aio.ServicerContext) -> None:
        if end_id in self._endpoints:
            logger.debug(f"{end_id} is already registered")
            return

        logger.debug(f"{end_id}: context = {context}")

        self._endpoints[end_id] = (None, context, None, None)

    async def _rx_task(self, end_id: str, reader) -> None:
        while True:
            try:
                msg = await reader.read()
            except grpc.aio.AioRpcError:
                logger.info(f"AioRpcError occurred for {end_id}")
                break

            if msg == grpc.aio.EOF:
                break

            await self._handle_data(msg)

        # grpc channel is unavailable
        # so, clean up an entry for end_id from _endpoints dict
        await self._eventq.put((BackendEvent.DISCONNECT, end_id))
        del self._endpoints[end_id]

        logger.info(f"cleaned up {end_id} info from _endpoints")
