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
import time
from typing import AsyncIterable, Iterable

import grpc

from ..common.constants import (DEFAULT_RUN_ASYNC_WAIT_TIME, EMPTY_PAYLOAD,
                                BackendEvent, CommType)
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from ..proto import backend_msg_pb2_grpc as msg_pb2_grpc
from ..proto import meta_pb2, meta_pb2_grpc
from .abstract import AbstractBackend
from .chunk_store import ChunkStore

logger = logging.getLogger(__name__)

ENDPOINT_TOKEN_LEN = 2
HEART_BEAT_DURATION = 30  # for metaserver

PEER_HEART_BEAT_PERIOD = 20  # 20 seconds
PEER_HEART_BEAT_WAIT_TIME = 1.5 * PEER_HEART_BEAT_PERIOD
HEART_BEAT_UPDATE_SKIP_TIME = PEER_HEART_BEAT_PERIOD / 4

GRPC_MAX_MESSAGE_LENGTH = 1073741824  # 1GB


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
        async for msg in req_iter:
            self.p2pbe._set_heart_beat(msg.end_id)
            # if the message is not a heart beat message,
            # the message needs to be processed.
            if msg.seqno != -1 or msg.eom is False or msg.channel_name != "":
                await self.p2pbe._handle_data(msg)

        return msg_pb2.BackendID(end_id=self.p2pbe._id)

    async def recv_data(self, req: msg_pb2.BackendID,
                        context) -> AsyncIterable[msg_pb2.Data]:
        """Implement a method to handle recv_data request."""
        # From server perspective, the server sends data to client.
        dck_task = asyncio.create_task(self._dummy_context_keeper(context))
        await self.p2pbe._set_writer(req.end_id, context)
        self.p2pbe._set_heart_beat(req.end_id)

        await dck_task

    async def _dummy_context_keeper(self, context):
        """Block the call with an event to keep context alive."""
        unreachable_event = asyncio.Event()

        # blocked here forever
        await unreachable_event.wait()
        logger.debug("must not reach here until termination!")


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
        self._channels = {}
        self._livecheck = {}
        self.delayed_channel_add = {}
        self.tx_tasks = []

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

            coro = self._setup_server()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)

        if not success:
            raise SystemError('initialization failure')

        self._initialized = True

    async def _setup_server(self):
        server = grpc.aio.server(options=[('grpc.max_send_message_length',
                                           GRPC_MAX_MESSAGE_LENGTH),
                                          ('grpc.max_receive_message_length',
                                           GRPC_MAX_MESSAGE_LENGTH)])
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
        task = asyncio.create_task(coro)
        self.tx_tasks.append(task)

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
                group=channel.groupby(),
                endpoint=self._backend,
            )

            meta_resp = stub.RegisterMetaInfo(meta_info)
            if meta_resp:
                meta_pb2_success = meta_pb2.MetaResponse.Status.SUCCESS
                if meta_resp.status != meta_pb2_success:
                    logger.debug("registration failed")
                    raise SystemError('registration failure')

                for endpoint in meta_resp.endpoints:
                    logger.debug(f"connecting to endpoint: {endpoint}")
                    await self._connect_and_notify(endpoint, channel.name())

            while True:
                meta_resp = stub.HeartBeat(meta_info)
                logger.debug(f"meta_resp from heart beat: {meta_resp}")
                await asyncio.sleep(HEART_BEAT_DURATION)

    async def _connect_and_notify(self, endpoint: str, ch_name: str) -> None:
        grpc_ch = grpc.aio.insecure_channel(
            endpoint,
            options=[('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
                     ('grpc.max_receive_message_length',
                      GRPC_MAX_MESSAGE_LENGTH)])
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
            logger.debug("can't proceed as grpc channel is unavailable")
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
            if stub is not None:  # this is client
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
            else:  # this is server
                # server needs to wait for writer context so that it can be ready
                # to send messages. here we can't call "channel.add(msg.end_id)."
                # therefore, we save info for adding end to a channel here.
                # we do actuall addition in _set_writer() method.
                if msg.end_id not in self.delayed_channel_add:
                    self.delayed_channel_add[msg.end_id] = []
                self.delayed_channel_add[msg.end_id].append(channel)

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

        if msg.end_id not in self._msg_chunks:
            channel = self._channels[msg.channel_name]
            self._msg_chunks[msg.end_id] = ChunkStore(self._loop, channel)

        chunk_store = self._msg_chunks[msg.end_id]
        if not chunk_store.assemble(msg) or chunk_store.eom:
            # clean up if message is wrong or completely assembled
            del self._msg_chunks[msg.end_id]

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

        logger.debug("_tx_task is done")

    async def _broadcast_task(self, channel):
        """Broadcast messages.

        p2p backend doesn't support broadcast natively.
        broadcast is simply a collection of unicast transmissions.
        """
        txq = channel.broadcast_q()

        while True:
            data = await txq.get()
            if data == EMPTY_PAYLOAD:
                txq.task_done()
                logger.debug("broadcast task got an empty msg from queue")
                break

            end_ids = list(channel._ends.keys())
            logger.debug(f"end ids for bcast = {end_ids}")
            for end_id in end_ids:
                try:
                    await self.send_chunks(end_id, channel.name(), data)
                except Exception as ex:
                    ex_name = type(ex).__name__
                    logger.debug(f"An exception of type {ex_name} occurred")

                    await self._cleanup_end(end_id)
            txq.task_done()

    async def _unicast_task(self, channel, end_id):
        txq = channel.get_txq(end_id)

        while True:
            try:
                data = await asyncio.wait_for(txq.get(),
                                              PEER_HEART_BEAT_PERIOD)
            except asyncio.TimeoutError:
                if end_id not in self._endpoints:
                    logger.debug(f"end_id {end_id} not in _endpoints")
                    break

                _, _, clt_writer, _ = self._endpoints[end_id]
                if clt_writer is None:
                    continue

                def heart_beat():
                    # the condition for heart beat message:
                    #    channel_name = ""
                    #    seqno = -1
                    #    eom = True
                    msg = msg_pb2.Data(end_id=self._id,
                                       channel_name="",
                                       payload=EMPTY_PAYLOAD,
                                       seqno=-1,
                                       eom=True)

                    yield msg

                logger.debug("sending heart beat to server")
                await clt_writer.send_data(heart_beat())
                continue

            if data == EMPTY_PAYLOAD:
                txq.task_done()
                logger.debug("unicast tx task got an empty msg from queue")
                break

            try:
                await self.send_chunks(end_id, channel.name(), data)
            except Exception as ex:
                ex_name = type(ex).__name__
                logger.debug(f"An exception of type {ex_name} occurred")

                await self._cleanup_end(end_id)
                txq.task_done()
                # This break ends a tx_task for end_id
                break

            txq.task_done()

        logger.debug(f"unicast task for {end_id} terminated")

    async def send_chunks(self, other: str, ch_name: str, data: bytes) -> None:
        """Send data chunks to an end."""
        _, svr_writer, clt_writer, _ = self._endpoints[other]

        # TODO: keep regenerating messages can be expensive; revisit this later
        if clt_writer is not None:
            await clt_writer.send_data(
                self._generate_data_messages(ch_name, data))
        elif svr_writer is not None:
            for msg in self._generate_data_messages(ch_name, data):
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
                               payload=chunk,
                               seqno=seqno,
                               eom=eom)

            yield msg

    async def _set_writer(self, end_id: str,
                          context: grpc.aio.ServicerContext) -> None:
        if end_id in self._endpoints:
            logger.debug(f"{end_id} is already registered")
            return

        logger.debug(f"{end_id}: context = {context}")

        self._endpoints[end_id] = (None, context, None, None)

        if end_id not in self.delayed_channel_add:
            return

        for channel in self.delayed_channel_add[end_id]:
            # add end to the channel
            await channel.add(end_id)

        del self.delayed_channel_add[end_id]

    async def _rx_task(self, end_id: str, reader) -> None:
        while True:
            try:
                msg = await reader.read()
            except grpc.aio.AioRpcError:
                logger.debug(f"AioRpcError occurred for {end_id}")
                break

            if msg == grpc.aio.EOF:
                logger.debug("got grpc.aio.EOF")
                break

            await self._handle_data(msg)

        # grpc channel is unavailable
        # so, clean up an entry for end_id from _endpoints dict
        await self._cleanup_end(end_id)

        logger.debug(f"cleaned up {end_id} info from _endpoints")

    async def _cleanup_end(self, end_id):
        await self._eventq.put((BackendEvent.DISCONNECT, end_id))
        if end_id in self._endpoints:
            del self._endpoints[end_id]
        if end_id in self._livecheck:
            self._livecheck[end_id].cancel()
            del self._livecheck[end_id]

    def _set_heart_beat(self, end_id) -> None:
        logger.debug(f"heart beat data message for {end_id}")
        if end_id not in self._livecheck:
            self._livecheck[end_id] = LiveChecker(self, end_id,
                                                  PEER_HEART_BEAT_WAIT_TIME)

        self._livecheck[end_id].reset()

    async def cleanup(self):
        """Clean up resources in backend."""
        # wait for tx tasks to be finished
        logger.debug(f"waiting for {len(self.tx_tasks)} tasks to be done")
        await asyncio.gather(*self.tx_tasks)
        logger.debug("all done")


class LiveChecker:
    """LiveChecker class."""

    def __init__(self, p2pbe, end_id, timeout) -> None:
        """Initialize an instance."""
        self._p2pbe = p2pbe
        self._end_id = end_id
        self._timeout = timeout

        self._task = None
        self._last_reset = time.time()

    async def _check(self):
        await asyncio.sleep(self._timeout)
        await self._p2pbe._cleanup_end(self._end_id)
        logger.debug(f"live check timeout occured for {self._end_id}")

    def cancel(self) -> None:
        """Cancel a task."""
        if self._task is None or self._task.cancelled():
            return

        self._task.cancel()
        logger.debug(f"cancelled task for {self._end_id}")

    def reset(self) -> None:
        """Reset a task."""
        now = time.time()
        if now - self._last_reset < HEART_BEAT_UPDATE_SKIP_TIME:
            # this is to prevent too frequent reset
            logger.debug("too frequent reset request; skip it")
            return

        self._last_reset = now

        self.cancel()

        self._task = asyncio.ensure_future(self._check())

        logger.debug(f"set future for {self._end_id}")
