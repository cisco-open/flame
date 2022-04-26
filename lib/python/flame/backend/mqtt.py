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
"""MQTT backend."""

import asyncio
import logging
import time
from collections import deque
from enum import IntEnum
from typing import Tuple

import paho.mqtt.client as mqtt
from google.protobuf.any_pb2 import Any
from paho.mqtt.client import MQTTv5

from ..channel import Channel
from ..common.constants import (DEFAULT_RUN_ASYNC_WAIT_TIME, MQTT_TOPIC_PREFIX,
                                BackendEvent, CommType)
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from .abstract import AbstractBackend
from .chunk_store import ChunkStore

END_STATUS_ON = 'online'
END_STATUS_OFF = 'offline'

# wait time of 10 sec
# clean up resources allocated for terminated end
# if no message arrives after the wait time
MQTT_TIME_WAIT = 10  # 10 sec
MIN_CHECK_PERIOD = 1  # 1 sec
MQTT_LOOP_CHECK_PERIOD = 0.1  # 100ms

logger = logging.getLogger(__name__)


# info for mqtt qos is found at
# https://www.hivemq.com/blog/mqtt-essentials-part-6-mqtt-quality-of-service-levels/
class MqttQoS(IntEnum):
    """Enum for MQTT QoS value."""

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class MqttBackend(AbstractBackend):
    """MqttBackend class. It only allows a singleton instance."""

    # variables for class initialization
    _instance = None
    _initialized = False
    _eventq = None  # inform channel manager of events occurring at backend
    _loop = None

    # variables for functionality
    _id = None
    _channels = None
    _broker = None
    _mqtt_client = None
    _last_payload_sig = None
    _msg_chunks = None
    _cleanup_waits = None

    def __new__(cls):
        """Create an instance if not exist."""
        if cls._instance is None:
            logger.info('creating an MqttBackend instance')
            cls._instance = super(MqttBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize an instance."""
        if self._initialized:
            return

        self._channels = {}
        self._cleanup_waits = {}

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

            coro = self._monitor_end_termination()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not success:
            raise SystemError('initialization failure')

        self._initialized = True

    async def _monitor_end_termination(self):
        period = MQTT_TIME_WAIT * 0.5
        period = MIN_CHECK_PERIOD if period <= MIN_CHECK_PERIOD else period

        while True:
            for end_id, expiry in list(self._cleanup_waits.items()):
                if time.time() >= expiry:
                    await self._eventq.put((BackendEvent.DISCONNECT, end_id))
                    del self._cleanup_waits[end_id]

            await asyncio.sleep(period)

    def configure(self, broker: str, job_id: str, task_id: str):
        """Configure the backend."""
        self._msg_chunks: dict[str, ChunkStore] = {}

        self._broker = broker
        self._job_id = job_id
        self._id = task_id

        self._mqtt_client = mqtt.Client(self._id, protocol=MQTTv5)

        self._health_check_topic = f'{MQTT_TOPIC_PREFIX}/{self._job_id}'

        async def _setup_mqtt_client():
            _ = asyncio.create_task(self._rx_task())

            self._mqtt_client.on_connect = self.on_connect
            self._mqtt_client.on_message = self.on_message
            self._mqtt_client.will_set(self._health_check_topic,
                                       payload=f'{self._id}:{END_STATUS_OFF}',
                                       qos=MqttQoS.EXACTLY_ONCE)

            _ = AsyncioHelper(self._loop, self._mqtt_client)

            self._mqtt_client.connect(self._broker)
            self._mqtt_client.subscribe(self._health_check_topic)

        coro = _setup_mqtt_client()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not success:
            logger.error('failed to set up mqtt client')
            raise ConnectionError

    def join(self, channel: Channel):
        """Join a channel by subscribing to topics."""
        self.notify(channel.name())

        # format for broadcast topic to subscribe:
        #   /flame/<job_id>/<channel_name>/<groupby>/broadcast/<other_role>/+
        # format for unicast topic to subscribe:
        #  /flame/<job_id>/<channel_name>/<groupby>/unicast/<other_role>/+/<my_role>/<my_end_id>
        sep = '/'
        for comm_type in CommType:
            topic = sep.join([
                MQTT_TOPIC_PREFIX,
                self._job_id,
                channel.name(),
                channel.groupby(),
                comm_type.name,
                channel.other_role(),
                '+',
            ])

            if comm_type == CommType.UNICAST:
                topic = sep.join([topic, channel.my_role(), self._id])

            self.subscribe(topic)

    def _handle_health_message(self, message):
        health_data = str(message.payload.decode("utf-8"))
        # the correct format of health data: <end_id>:<status>
        (end_id, status) = health_data.split(':')[0:2]
        logger.debug(f'end: {end_id}, status: {status}')
        if end_id == self._id or status == END_STATUS_ON:
            # nothing to do
            return

        expiry = time.time() + MQTT_TIME_WAIT
        self._cleanup_waits[end_id] = expiry
        logger.debug(f'end: {end_id}, expiry time: {expiry}')

    async def _handle_notification(self, any_msg):
        msg = msg_pb2.Notify()
        any_msg.Unpack(msg)

        if msg.channel_name not in self._channels:
            logger.debug('channel not found')
            return

        channel = self._channels[msg.channel_name]

        if not channel.has(msg.end_id):
            # this is the first time to see this end,
            # so let's notify my presence to the end
            logger.debug('acknowledge notification')
            self.notify(msg.channel_name)

        await channel.add(msg.end_id)

    async def _handle_data(self, any_msg: Any) -> None:
        msg = msg_pb2.Data()
        any_msg.Unpack(msg)

        # update is needed only if end_id's termination is detected
        if msg.end_id in self._cleanup_waits:
            # update end_id to _cleanup_waits dictionary
            expiry = time.time() + MQTT_TIME_WAIT
            self._cleanup_waits[msg.end_id] = expiry

        payload, fully_assembled = self.assemble_chunks(msg)

        channel = self._channels[msg.channel_name]

        if fully_assembled:
            logger.debug(f'fully assembled data size = {len(payload)}')
            rxq = channel.get_rxq(msg.end_id)
            await rxq.put(payload)

    async def _rx_task(self):
        self._rx_deque = deque()
        self._rx_deque.append(self._loop.create_future())

        while True:
            message = await self._rx_deque[0]
            self._rx_deque.popleft()

            if message.topic == self._health_check_topic:
                self._handle_health_message(message)
                continue

            logger.debug(
                f'_rx_task - topic: {message.topic}; len: {len(message.payload)}'
            )

            any_msg = Any().FromString(message.payload)

            if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
                await self._handle_notification(any_msg)
            elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
                await self._handle_data(any_msg)
            else:
                logger.warning('unknown message type')

    def uid(self):
        """Return backend id."""
        return self._id

    def eventq(self):
        """Return a event queue object."""
        return self._eventq

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """on_connect publishes a health check message to a mqtt broker."""
        logger.debug('calling on_connect')

        # publish health data; format: <end_id>:<status>
        # status is either END_STATUS_ON or END_STATUS_OFF
        client.publish(self._health_check_topic,
                       payload=f'{self._id}:{END_STATUS_ON}',
                       qos=MqttQoS.EXACTLY_ONCE)

    def on_message(self, client, userdata, message):
        """on_message receives message."""
        logger.debug(
            f'on_message - topic: {message.topic}; len: {len(message.payload)}'
        )

        idx = len(self._rx_deque) - 1
        # set result at the end of the queue
        self._rx_deque[idx].set_result(message)
        # add one extra future in the queue
        self._rx_deque.append(self._loop.create_future())

    def assemble_chunks(self, msg: msg_pb2.Data) -> Tuple[bytes, bool]:
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

    def subscribe(self, topic):
        """Subscribe to a topic."""
        logger.debug(f'subscribe topic: {topic}')
        self._mqtt_client.subscribe(topic, qos=MqttQoS.EXACTLY_ONCE)

    def notify(self, channel_name):
        if channel_name not in self._channels:
            logger.debug(f'channel {channel_name} not found')
            return False

        channel = self._channels[channel_name]

        topic = self.topic_for_pub(channel)

        msg = msg_pb2.Notify()
        msg.end_id = self._id
        msg.channel_name = channel_name

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        logger.debug(f'notify: publish topic: {topic}')
        self._mqtt_client.publish(topic, payload, qos=MqttQoS.EXACTLY_ONCE)

        return True

    def loop(self):
        """Return loop instance of asyncio."""
        return self._loop

    def attach_channel(self, channel):
        """Attach a channel to backend."""
        self._channels[channel.name()] = channel

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

        return True

    def topic_for_pub(self,
                      ch: Channel,
                      other_id: str = "",
                      comm_type=CommType.BROADCAST):
        """Return a proper topic for a given channel."""
        sep = '/'

        if comm_type == CommType.BROADCAST:
            topic = sep.join([
                MQTT_TOPIC_PREFIX,
                ch.job_id(),
                ch.name(),
                ch.groupby(),
                CommType.BROADCAST.name,
                ch.my_role(),
                self._id,
            ])
        elif comm_type == CommType.UNICAST:
            topic = sep.join([
                MQTT_TOPIC_PREFIX,
                ch.job_id(),
                ch.name(),
                ch.groupby(),
                CommType.UNICAST.name,
                ch.my_role(),
                self._id,
                ch.other_role(),
                other_id,
            ])
        else:
            raise ValueError(f'unknown CommType {comm_type}')

        return topic

    async def _tx_task(self, channel, end_id, comm_type: CommType):
        """Conducts data transmission in a loop.

        _tx_task() must be created per tx queue right after end_id is added to
        channel (e.g., channel.add(end_id)).
        In case of a tx task for broadcast queue, a broadcaset queue must be
        created first.
        """
        if comm_type == CommType.BROADCAST:
            txq = channel.broadcast_q()
        else:
            txq = channel.get_txq(end_id)

        topic = self.topic_for_pub(channel, end_id, comm_type)

        while True:
            data = await txq.get()
            self.send_chunks(topic, channel.name(), data)
            txq.task_done()

    def send_chunks(self, topic, ch_name: str, data: bytes) -> None:
        """Send data chunks."""
        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            self.send_chunk(topic, ch_name, chunk, seqno, eom)

    def send_chunk(self, topic: str, channel_name: str, data: bytes,
                   seqno: int, eom: bool) -> None:
        """Send a chunk."""
        msg = msg_pb2.Data()
        msg.end_id = self._id
        msg.channel_name = channel_name
        msg.payload = data
        msg.seqno = seqno
        msg.eom = eom

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        info = self._mqtt_client.publish(topic,
                                         payload,
                                         qos=MqttQoS.EXACTLY_ONCE)

        while not info.is_published():
            logger.debug('waiting for publish completion')
            self._mqtt_client.loop(MQTT_LOOP_CHECK_PERIOD)

        logger.debug(f'sending chunk {seqno} to {topic} is done')


class AsyncioHelper:
    """Asyncio helper class.

    Asyncio MQTT client example from
    https://github.com/eclipse/paho.mqtt.python/blob/master/examples/loop_asyncio.py
    """

    def __init__(self, loop, client):
        self.loop = loop
        self.client = client
        self.client.on_socket_open = self.on_socket_open
        self.client.on_socket_close = self.on_socket_close
        self.client.on_socket_register_write = self.on_socket_register_write
        self.client.on_socket_unregister_write = self.on_socket_unregister_write

    def on_socket_open(self, client, userdata, sock):
        logger.debug('Socket opened')

        def cb():
            logger.debug('Socket is readable, calling loop_read')
            client.loop_read()

        self.loop.add_reader(sock, cb)
        self.misc = self.loop.create_task(self.misc_loop())

    def on_socket_close(self, client, userdata, sock):
        logger.debug('Socket closed')
        self.loop.remove_reader(sock)
        self.misc.cancel()

    def on_socket_register_write(self, client, userdata, sock):
        logger.debug('Watching socket for writability.')

        def cb():
            logger.debug('Socket is writable, calling loop_write')
            client.loop_write()

        self.loop.add_writer(sock, cb)

    def on_socket_unregister_write(self, client, userdata, sock):
        logger.debug('Stop watching socket for writability.')
        self.loop.remove_writer(sock)

    async def misc_loop(self):
        logger.debug('misc_loop started')
        while self.client.loop_misc() == mqtt.MQTT_ERR_SUCCESS:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
        logger.debug('misc_loop finished')
