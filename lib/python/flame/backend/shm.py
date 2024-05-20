# Copyright 2024 Cisco Systems, Inc. and its affiliates
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
"""LIFL's Shared Memory Backend."""

import asyncio
import logging
import socket
import os, sys
import time

from typing import Union
from multiprocessing import shared_memory
from datetime import datetime
import threading
from shared_memory_dict import SharedMemoryDict

from flame.backend.abstract import AbstractBackend
from flame.channel import Channel
from flame.common.constants import DEFAULT_RUN_ASYNC_WAIT_TIME, EMPTY_PAYLOAD, CommType
from flame.common.util import background_thread_loop, run_async
from flame.proto import backend_msg_pb2 as msg_pb2

logger = logging.getLogger(__name__)

# We set environment variable to use multiprocessing.Lock
# on write operations of shared memory  
os.environ["SHARED_MEMORY_USE_LOCK"] = "1"

SHM_DICT_SIZE = 1024

class LIFLSharedMemoryBackend(AbstractBackend):
    """SharedMemoryBackend class.

    SharedMemory backend is EXPERIMENTAL.
    """

    def __init__(self):
        """Initialize an instance."""
        self._instance = None
        self._initialized = False
        self._loop = None

        # variables for functionality
        self._id = None
        self._channels = None
        self._broker = None
        if self._initialized:
            return

        self._backend = None

        self._channels = dict()
        self.tx_tasks = dict()
        self._cleanup_ready = dict()

        self._is_shm_buf_created = dict()

        # TODO: make them configurable
        self.sockmap_server_ip   = "127.0.0.1"
        self.sockmap_server_port = 10105
        self.rpc_server_ip     = "127.0.0.1"
        self.rpc_server_port   = 10106

        with background_thread_loop() as loop:
            self._loop = loop

        async def _setup_rx_queue():
            self._rx_queue = asyncio.Queue()

        _, _ = run_async(_setup_rx_queue(), self._loop)

        self._initialized = True

    def configure(self, broker: str, job_id: str, task_id: str):
        """Configure the backend."""
        self._broker = broker
        self._job_id = job_id
        self._id = task_id

        self._setup_ebpf()

        logger.debug("Creating rx_thread.")
        rx_thread = threading.Thread(target=self._rx_task)
        rx_thread.daemon = True  # Daemonize the thread
        rx_thread.start()

        async def _create_handle_data_task():
            _ = asyncio.create_task(self._handle_data())

        _, success = run_async(_create_handle_data_task(), self._loop)
        if not success:
            raise SystemError("_create_handle_data_task failure")

        logger.debug("Configuration of the shm backend is completed.")

    def _setup_ebpf(self):
        """Register to sockmap manager."""
        self.sockmap_sock = self._sockmap_client(self.sockmap_server_ip, self.sockmap_server_port)
        self.rpc_sock = self._rpc_client(self.rpc_server_ip, self.rpc_server_port)

        self.pid = os.getpid()
        self.sock_fd = self.sockmap_sock.fileno()        
        self.fn_id = bytes.fromhex(self._id)[-2:]

        skmsg_md = [self.pid, self.sock_fd, self.fn_id]
        skmsg_md_bytes = b''.join([skmsg_md[0].to_bytes(4, byteorder = 'little'), skmsg_md[1].to_bytes(4, byteorder = 'little'), skmsg_md[2]])

        logger.debug(f"skmsg_md: PID {self.pid}; socket FD {self.sock_fd}; end ID {self.fn_id}")
        logger.debug(f"skmsg_md_bytes: {skmsg_md_bytes}")

        self.rpc_sock.send(skmsg_md_bytes)
        self.rpc_sock.close()

    def _create_socket(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as msg:
            logger.info(f"Failed to create socket. Error code: {str(msg[0])}, Error message: {msg[1]}")
            sys.exit()

        return sock

    def _rpc_client(self, remote_ip, port):
        sock = self._create_socket()

        sock.connect((remote_ip, port))
        logger.info(f"Connected to RPC server {remote_ip}:{port}")

        return sock

    def _sockmap_client(self, remote_ip, port):
        sock = self._create_socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        sock.connect((remote_ip, port))
        logger.info(f"Connected to sockmap server {remote_ip}:{port}")

        return sock

    def loop(self):
        """Return loop instance of asyncio."""
        return self._loop

    def uid(self):
        """Return backend id."""
        return self._id

    def join(self, channel) -> None:
        """Join a channel."""

        shm_ends = SharedMemoryDict(name = channel.name() + "-" + channel.my_role(), size = SHM_DICT_SIZE)
        shm_ends[self._id] = 1

        logger.debug(f"{len(shm_ends)} {channel.name()}-{channel.my_role()} shm ends: {shm_ends}")

        async def _join_inner():
            while True:
                await asyncio.sleep(0.5)

                peer_shm_ends = SharedMemoryDict(name = channel.name() + "-" + channel.other_role(), size = SHM_DICT_SIZE)
                logger.debug(f"{len(peer_shm_ends)} peer shm ends: {peer_shm_ends}")

                for peer_end_id in peer_shm_ends.keys():
                    if channel.has(peer_end_id) is False:
                        print("New shm end detected:", peer_end_id)

                        await channel.add(peer_end_id)

        async def _create_join_inner_task():
            _ = asyncio.create_task(_join_inner())

        _, success = run_async(_create_join_inner_task(), self._loop)
        if not success:
            raise SystemError("_create_join_inner_task failure")

    def leave(self, channel) -> None:
        """Leave a given channel.
        
        TODO: notify the sockmap manager to remove the entry from eBPF map
        """
        logger.info("Clean up shared memory buffers.")

        for end in channel.all_ends():
            shm_buf = shared_memory.SharedMemory(name = end)
            shm_buf.close()
            if end == self._id:
                shm_buf.unlink()

        # NOTE: this method may recreate the shm dict.
        shm_ends = SharedMemoryDict(name = channel.name() + "-" + channel.my_role(), size = SHM_DICT_SIZE)
        del shm_ends[self._id]

        if len(shm_ends) == 0:
            shm_ends.shm.close()
            shm_ends.shm.unlink()
            del shm_ends

        # NOTE: this method may recreate the shm dict.
        other_ends = SharedMemoryDict(name = channel.name() + "-" + channel.other_role(), size = SHM_DICT_SIZE)
        other_ends.shm.close()

    def create_tx_task(
        self, channel_name: str, end_id: str, comm_type=CommType.UNICAST
    ) -> bool:
        """Create asyncio task for transmission."""
        if channel_name not in self._channels or (
            not self._channels[channel_name].has(end_id)
            and comm_type != CommType.BROADCAST
        ):
            return False

        channel = self._channels[channel_name]

        coro = self._tx_task(channel, end_id, comm_type)
        task = asyncio.create_task(coro)
        if channel_name not in self.tx_tasks:
            self.tx_tasks[channel_name] = list()
        self.tx_tasks[channel_name].append(task)

    def attach_channel(self, channel) -> None:
        """Attach a channel to backend."""
        self._channels[channel.name()] = channel

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

        shm backend doesn't support broadcast natively.
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
            logger.debug(f"end ids for {channel.name()} bcast = {end_ids}")
            for end_id in end_ids:
                try:
                    self.send_data(end_id, channel.name(), data)
                except Exception as ex:
                    ex_name = type(ex).__name__
                    logger.debug(f"An exception of type {ex_name} occurred")
                    await self._cleanup_end(end_id)

            txq.task_done()

        logger.debug(f"broadcast task for {channel.name()} terminated")

    async def _unicast_task(self, channel, end_id):
        txq = channel.get_txq(end_id)

        while True:
            data = await txq.get()
            if data == EMPTY_PAYLOAD:
                txq.task_done()
                logger.debug("broadcast task got an empty msg from queue")
                break

            try:
                self.send_data(end_id, channel.name(), data)
            except Exception as ex:
                ex_name = type(ex).__name__
                logger.debug(f"An exception of type {ex_name} occurred: {ex}")
                await self._cleanup_end(end_id)

            txq.task_done()

        logger.debug(f"unicast task for {end_id} terminated")

    def send_data(self, other: str, ch_name: str, data: bytes) -> None:
        """Send data references to an end."""
        msg = self._generate_data_messages(ch_name, other, data)

        next_fn_id = bytes.fromhex(other)[-2:]
        next_fn_id_int = int.from_bytes(next_fn_id, byteorder = 'little')

        skmsg_md_bytes = b"".join([next_fn_id_int.to_bytes(4, byteorder = 'little'), msg.SerializeToString()])

        self.sockmap_sock.sendall(skmsg_md_bytes)

    def _generate_data_messages(self, ch_name: str, other: str, data: bytes) -> msg_pb2.Data:
        self.set_data(data, other)

        msg = msg_pb2.Data(
            end_id=self._id, # end_id == buf_id in shm backend
            channel_name=ch_name,
            payload=b"",
            seqno=len(data), # Length of data instead of seq no.
            eom=True,
        )

        return msg

    def _rx_task(self) -> None:
        logger.debug(f"{self._id} creating _rx_task()")

        while True:
            logger.debug(f"{self._id} receiving msg")
            skmsg_md = self.sockmap_sock.recv(1024)
            logger.debug(f"{self._id} received request: {skmsg_md}")

            msg = msg_pb2.Data.FromString(skmsg_md[4:])

            # NOTE: Parse SK_MSG; Check if SK_MSG is allowed or not
            truncated_target_end_id = int.from_bytes(skmsg_md[0:3], "little")
            truncated_self_id = int.from_bytes(bytes.fromhex(self._id)[-2:], byteorder = 'little')

            if truncated_target_end_id != truncated_self_id:
                logger.warning(f"{self._id} (truncated_self_id) received unexpected msg! target_end_id: {truncated_target_end_id}")

            _, _ = run_async(self._rx_queue.put(msg), self._loop)

    async def _handle_data(self) -> None:
        logger.debug("Starting _handle_data loop")

        while True:
            logger.debug("Awaiting for msg from rx queue")
            msg = await self._rx_queue.get()

            if msg.end_id == self._id:
                # This case happens when message is broadcast to a self-loop
                # e.g., distributed topology
                logger.debug("message sent to self; do nothing")
                return

            if msg.channel_name in self._channels:
                channel = self._channels[msg.channel_name]
                await self.handle(msg, channel)

    async def set_cleanup_ready_async(self, end_id: str) -> None:
        """Set cleanup ready event for a given end id.

        This should be called in self._loop thread.
        This is not yet implemented.
        """
        pass

    def set_cleanup_ready(self, end_id: str) -> None:
        """Set cleanup ready event for a given end id.

        This should be called in self._loop thread.
        This is not yet implemented.
        """
        pass

    async def _cleanup_end(self, end_id):
        """Clean up resoure in the backend related to end_id.

        This should be called in self._loop thread.
        This is not yet implemented.
        """
        pass

    def get_data(self, other, msg_size) -> Union[bytes, None]:
        """Return the data buffer of shared memory."""
        read_buf = shared_memory.SharedMemory(other + "-" + self._id)
        data = bytes(read_buf.buf[:msg_size])

        return data

    def set_data(self, data: bytes, other: str) -> None:
        """Write data to shared memory buffer."""
        key = self._id + "-" + other

        if key not in self._is_shm_buf_created:
            wrt_buf = shared_memory.SharedMemory(name = self._id + "-" + other, create = True, size = len(data))
            self._is_shm_buf_created[self._id + "-" + other] = True
        else:
            wrt_buf = shared_memory.SharedMemory(name = self._id + "-" + other)

        wrt_buf.buf[:len(data)] = data

    async def handle(self, msg: msg_pb2.Data, channel: Channel) -> None:
        """Process msg."""
        payload = self.get_data(msg.end_id, msg.seqno)

        timestamp = datetime.now()

        rxq = channel.get_rxq(msg.end_id)
        if rxq is None:
            logger.debug(f"rxq not found for {msg.end_id}")

            # NOTE: We piggyback the handshake ack in the skmsg
            # Thus, there is no need to send an explicit ack.
            await channel.add(msg.end_id)
            rxq = channel.get_rxq(msg.end_id)

            # TODO: need to differentiate from the cleanup event.
            # set cleanup ready event for a given end id
            # await self.set_cleanup_ready_async(msg.end_id)
            # return

        await rxq.put((payload, timestamp))