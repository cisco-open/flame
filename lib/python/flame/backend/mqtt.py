# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""MQTT backend."""

import asyncio
import logging
import time
from collections import deque
from enum import IntEnum

import paho.mqtt.client as mqtt
from flame.backend.abstract import AbstractBackend
from flame.backend.chunk_manager import ChunkManager
from flame.backend.chunk_store import ChunkStore
from flame.channel import Channel
from flame.common.constants import (
    DEFAULT_RUN_ASYNC_WAIT_TIME,
    EMPTY_PAYLOAD,
    MQTT_TOPIC_PREFIX,
    CommType,
)
from flame.common.util import background_thread_loop, run_async
from flame.proto import backend_msg_pb2 as msg_pb2
from google.protobuf.any_pb2 import Any
from paho.mqtt.client import MQTTv5

END_STATUS_ON = "online"
END_STATUS_OFF = "offline"

# wait time of 10 sec clean up resources allocated for terminated end if no
# message arrives after the wait time NOTE: DG increased wait_time to 4h to
# handle trainer init stalls and other waits during unavailability unaware
# training
MQTT_TIME_WAIT = 14400  # 14400 sec
MIN_CHECK_PERIOD = 1  # 1 sec
MQTT_LOOP_CHECK_PERIOD = 1  # 1 sec
TOPIC_SEP = "/"

logger = logging.getLogger(__name__)


# info for mqtt qos is found at
# https://www.hivemq.com/blog/mqtt-essentials-part-6-mqtt-quality-of-service-levels/
class MqttQoS(IntEnum):
    """Enum for MQTT QoS value."""

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class MqttBackend(AbstractBackend):
    """MqttBackend class."""

    def __init__(self):
        """Initialize an instance."""
        # variables for class initialization
        self._instance = None
        self._initialized = False
        self._loop = None

        # variables for functionality
        self._id = None
        self._channels = None
        self._broker = None
        self._mqtt_client = None
        self._last_payload_sig = None
        self._cleanup_waits = None
        # TODO: (DG) check if _cleanup_ready is being used correctly Refer
        # p2p.py for implementation
        self._cleanup_ready = set()
        self._is_connected = False
        self._reconnecting = False

        if self._initialized:
            return

        self._channels = {}
        self._cleanup_waits = {}

        with background_thread_loop() as loop:
            self._loop = loop

        # initialize a chunk manager
        self.chunk_mgr = ChunkManager(self)

        async def _init_loop_stuff():
            coro = self._monitor_end_termination()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not success:
            raise SystemError("initialization failure")

        self._initialized = True

    async def _monitor_end_termination(self):
        period = MQTT_TIME_WAIT * 1
        period = MIN_CHECK_PERIOD if period <= MIN_CHECK_PERIOD else period

        while True:
            for end_id, expiry in list(self._cleanup_waits.items()):
                if time.time() >= expiry:
                    logger.info(f"end termination check timed out: {end_id}")
                    # linear iteration is okay because there are not many
                    # channels per role in general
                    for _, channel in self._channels.items():
                        # remove the end id from the channel
                        await channel.remove(end_id)

                    del self._cleanup_waits[end_id]

            await asyncio.sleep(period)

    def configure(self, broker: str, job_id: str, task_id: str):
        logger.debug(f"Inside configure {task_id}")
        """Configure the backend."""
        self._broker = broker
        self._job_id = job_id
        self._id = task_id

        self._mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1, self._id, protocol=MQTTv5
        )

        self._health_check_topic = f"{MQTT_TOPIC_PREFIX}/{self._job_id}"

        async def _setup_mqtt_client():
            _ = asyncio.create_task(self._rx_task())

            self._mqtt_client.on_connect = self.on_connect
            self._mqtt_client.on_disconnect = self.on_disconnect
            self._mqtt_client.on_message = self.on_message
            self._mqtt_client.will_set(
                self._health_check_topic,
                payload=f"{self._id}:{END_STATUS_OFF}",
                qos=MqttQoS.EXACTLY_ONCE,
            )

            _ = AsyncioHelper(self._loop, self._mqtt_client)

            # Connect with extended keepalive timeout (300s instead of default 60s)
            # This prevents disconnections during heavy weight distribution to many trainers
            # where synchronous send operations can temporarily block the event loop
            keepalive_timeout = 300  # 5 minutes
            self._mqtt_client.connect(self._broker, keepalive=keepalive_timeout)
            logger.info(f"MQTT client connecting to {self._broker} with keepalive={keepalive_timeout}s")
            self._mqtt_client.subscribe(self._health_check_topic)

        coro = _setup_mqtt_client()
        _, success = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not success:
            logger.error("failed to set up mqtt client")
            raise ConnectionError

    def _topics_for_notify(self, channel: Channel) -> list[str]:
        notify_topics = list()

        # format for broadcast topic to subscribe:
        #   /flame/<job_id>/<channel_name>/<groupby>/broadcast/<other_role>/+
        # format for unicast topic to subscribe:
        #  /flame/<job_id>/<channel_name>/<groupby>/unicast/<other_role>/+/<my_role>/<my_end_id>
        for comm_type in CommType:
            topic = TOPIC_SEP.join(
                [
                    MQTT_TOPIC_PREFIX,
                    self._job_id,
                    channel.name(),
                    channel.groupby(),
                    comm_type.name,
                    channel.other_role(),
                    "+",
                ]
            )

            if comm_type == CommType.UNICAST:
                topic = TOPIC_SEP.join([topic, channel.my_role(), self._id])

            notify_topics.append(topic)

        return notify_topics

    def join(self, channel: Channel) -> None:
        """Join a channel by subscribing to topics."""
        for topic in self._topics_for_notify(channel):
            self.subscribe(topic)

        # notify after subscription to topics is finished
        self.notify(channel.name(), msg_pb2.NotifyType.JOIN)

    def leave(self, channel: Channel) -> None:
        """Leave a channel.

        Send leave notify message and unsubscribe from topics.
        """
        logger.debug(
            f"Sending notification of type LEAVE from " f"channel {channel.name()}"
        )
        self.notify(channel.name(), msg_pb2.NotifyType.LEAVE)

        logger.debug("Going to start unsubscribing topics")

        # unsubscribe from topics after notify is finished
        for topic in self._topics_for_notify(channel):
            logger.info(
                f"Initiating unsubscribe for topic {topic} "
                f"on channel {channel.name()}"
            )
            self.unsubscribe(topic)
        pass

    def update_trainer_state(
        self, channel: Channel, state: str, timestamp: str
    ) -> None:
        """Update a trainer state in the backend."""
        logger.info(
            f"Sending notification of type STATE_UPDATE from "
            f"channel {channel.name()} for state {state} at timestamp {timestamp}"
        )
        # notify after subscription to topics is finished
        self.notify(channel.name(), msg_pb2.NotifyType.STATE_UPDATE, (state, timestamp))

    def _handle_health_message(self, message):
        health_data = str(message.payload.decode("utf-8"))
        # the correct format of health data: <end_id>:<status>
        (end_id, status) = health_data.split(":")[0:2]
        logger.debug(f"end: {end_id}, status: {status}")
        if end_id == self._id or status == END_STATUS_ON:
            # nothing to do
            return

        expiry = time.time() + MQTT_TIME_WAIT
        self._cleanup_waits[end_id] = expiry
        logger.debug(f"end: {end_id}, expiry time: {expiry}")

    async def _handle_notification(self, any_msg):
        msg = msg_pb2.Notify()
        any_msg.Unpack(msg)

        if msg.end_id == self._id:
            # This case happens when message is broadcast to a self-loop e.g.,
            # distributed topology
            logger.debug("message sent to self; do nothing")
            return

        if msg.channel_name not in self._channels:
            logger.debug("channel not found")
            return

        channel = self._channels[msg.channel_name]

        if msg.type == msg_pb2.NotifyType.JOIN and not channel.has(msg.end_id):
            # this is the first time to see this end, so let's notify my
            # presence to the end
            logger.debug(f"Acknowledge join notification from {msg.end_id}")
            self.notify(msg.channel_name, msg_pb2.NotifyType.JOIN)

            # add end to the channel
            await channel.add(msg.end_id)
        elif msg.type == msg_pb2.NotifyType.LEAVE:
            logger.debug(f"Got channel leave message from {msg.end_id}")
            await channel.remove(msg.end_id)
        elif msg.type == msg_pb2.NotifyType.STATE_UPDATE:
            logger.info(f"Got state update message from {msg.end_id}")
            await channel.update_state(msg.end_id, msg.info.state, msg.info.timestamp)

    async def _handle_data(self, any_msg: Any) -> None:
        msg = msg_pb2.Data()
        any_msg.Unpack(msg)

        if msg.end_id == self._id:
            # This case happens when message is broadcast to a self-loop e.g.,
            # distributed topology
            logger.debug("message sent to self; do nothing")
            return

        # update is needed only if end_id's termination is detected
        if msg.end_id in self._cleanup_waits:
            # update end_id to _cleanup_waits dictionary
            expiry = time.time() + MQTT_TIME_WAIT
            self._cleanup_waits[msg.end_id] = expiry

        channel = self._channels[msg.channel_name]
        logger.debug(f"Message being sent to chunk_mgr for end: {msg.end_id}")
        self.chunk_mgr.handle(msg, channel)

    async def _rx_task(self):
        self._rx_deque = deque()
        self._rx_deque.append(self._loop.create_future())
        logger.debug("inside _rx_task")
        while True:
            message = await self._rx_deque[0]
            self._rx_deque.popleft()

            if message.topic == self._health_check_topic:
                logger.debug("topic same")
                self._handle_health_message(message)
                continue

            logger.debug(
                f"_rx_task - topic: {message.topic}; len: {len(message.payload)}"
            )

            any_msg = Any().FromString(message.payload)

            if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
                await self._handle_notification(any_msg)
            elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
                await self._handle_data(any_msg)
            else:
                logger.warning("unknown message type")

    def uid(self):
        """Return backend id."""
        return self._id

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """on_connect publishes a health check message to a mqtt
        broker."""
        logger.info(f"MQTT client connected: rc={rc}")
        self._is_connected = True
        self._reconnecting = False

        # publish health data; format: <end_id>:<status> status is either
        # END_STATUS_ON or END_STATUS_OFF
        temp = client.publish(
            self._health_check_topic,
            payload=f"{self._id}:{END_STATUS_ON}",
            qos=MqttQoS.EXACTLY_ONCE,
        )
        if temp:
            logger.debug(f"on_connect temp: {temp}")

    def on_disconnect(self, client, userdata, rc, properties=None):
        """on_disconnect handles disconnection from MQTT broker.
        
        Common causes of disconnection:
        1. Keepalive timeout: Broker disconnects if no keepalive received within timeout period
           - Original issue: Default 60s keepalive + blocking send operations = timeout
           - Now using 300s keepalive + async sends with event loop yields
        2. Network issues: Temporary connectivity problems
        3. Broker resource limits: Max connections, rate limits, memory pressure
        4. Heavy load: Multiple simultaneous large message transmissions blocking event loop
        """
        logger.warning(f"MQTT client disconnected: rc={rc}")
        self._is_connected = False
        
        if rc != mqtt.MQTT_ERR_SUCCESS and not self._reconnecting:
            # Unexpected disconnect, attempt reconnection
            logger.info("Attempting to reconnect to MQTT broker...")
            self._reconnecting = True
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self._reconnecting = False

    def on_message(self, client, userdata, message):
        """on_message receives message."""
        logger.debug(f"topic: {message.topic}; len: {len(message.payload)}")
        idx = len(self._rx_deque) - 1

        if self._rx_deque[idx].cancelled():
            # this is because _rx_task is cancelled rx_task is cancelled when
            # the program exits; nothing to do
            return

        # set result at the end of the queue
        self._rx_deque[idx].set_result(message)
        # add one extra future in the queue
        self._rx_deque.append(self._loop.create_future())
        logger.debug(f"deque size = {len(self._rx_deque)}")

    def subscribe(self, topic) -> None:
        """Subscribe to a topic."""
        logger.debug(f"subscribe topic: {topic}")
        self._mqtt_client.subscribe(topic, qos=MqttQoS.EXACTLY_ONCE)

    def unsubscribe(self, topic) -> None:
        """Unsubscribe from a topic."""
        logger.debug(f"unsubscribe topic: {topic}")
        self._mqtt_client.unsubscribe(topic)

    def notify(self, channel_name, notify_type, notify_info=None) -> bool:
        """Broadcast a notify message to a channel."""
        if channel_name not in self._channels:
            logger.debug(f"channel {channel_name} not found")
            return False

        channel = self._channels[channel_name]

        topic = self.topic_for_pub(channel)

        msg = msg_pb2.Notify()
        msg.end_id = self._id
        msg.channel_name = channel_name
        msg.type = notify_type

        # Add notify_info if it contains state and timestamp
        if notify_info is not None:
            msg.info.state = notify_info[0]
            msg.info.timestamp = notify_info[1]

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()
        logger.debug(f"notify: publish topic: {topic}")
        
        # Check connection before publishing
        if not self._is_connected:
            logger.warning(f"Cannot send notify: MQTT client not connected")
            return False
        
        try:
            self._mqtt_client.publish(topic, payload, qos=MqttQoS.EXACTLY_ONCE)
            return True
        except Exception as e:
            logger.error(f"Failed to send notify message: {e}")
            return False

    def loop(self):
        """Return loop instance of asyncio."""
        return self._loop

    def attach_channel(self, channel):
        """Attach a channel to backend."""
        self._channels[channel.name()] = channel

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
        _ = asyncio.create_task(coro)

        return True

    def topic_for_pub(
        self, ch: Channel, other_id: str = "", comm_type=CommType.BROADCAST
    ):
        """Return a proper topic for a given channel."""
        if comm_type == CommType.BROADCAST:
            topic = TOPIC_SEP.join(
                [
                    MQTT_TOPIC_PREFIX,
                    ch.job_id(),
                    ch.name(),
                    ch.groupby(),
                    CommType.BROADCAST.name,
                    ch.my_role(),
                    self._id,
                ]
            )
        elif comm_type == CommType.UNICAST:
            topic = TOPIC_SEP.join(
                [
                    MQTT_TOPIC_PREFIX,
                    ch.job_id(),
                    ch.name(),
                    ch.groupby(),
                    CommType.UNICAST.name,
                    ch.my_role(),
                    self._id,
                    ch.other_role(),
                    other_id,
                ]
            )
        else:
            raise ValueError(f"unknown CommType {comm_type}")

        return topic

    async def _tx_task(self, channel, end_id, comm_type: CommType):
        """Conducts data transmission in a loop.

        _tx_task() must be created per tx queue right after end_id is added to
        channel (e.g., channel.add(end_id)). In case of a tx task for broadcast
        queue, a broadcaset queue must be created first.
        """
        if comm_type == CommType.BROADCAST:
            txq = channel.broadcast_q()
        else:
            txq = channel.get_txq(end_id)

        topic = self.topic_for_pub(channel, end_id, comm_type)

        while True:
            data = await txq.get()
            if data == EMPTY_PAYLOAD:
                txq.task_done()
                logger.debug("task got an empty msg from queue")
                break

            await self.send_chunks_async(topic, channel.name(), data)
            txq.task_done()

    def send_chunks(self, topic, ch_name: str, data: bytes) -> None:
        """Send data chunks (synchronous version - deprecated, use send_chunks_async)."""
        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            self.send_chunk(topic, ch_name, chunk, seqno, eom)

    async def send_chunks_async(self, topic, ch_name: str, data: bytes) -> None:
        """Send data chunks asynchronously.
        
        This async version allows the event loop to process other tasks
        (including keepalive messages) between chunk transmissions, preventing
        MQTT broker disconnections during heavy load.
        """
        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            await self.send_chunk_async(topic, ch_name, chunk, seqno, eom)

    def send_chunk(
        self, topic: str, channel_name: str, data: bytes, seqno: int, eom: bool
    ) -> None:
        """Send a chunk with retry logic for disconnections."""
        msg = msg_pb2.Data()
        msg.end_id = self._id
        msg.channel_name = channel_name
        msg.payload = data
        msg.seqno = seqno
        msg.eom = eom

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        max_retries = 5
        retry_delay = 1.0  # initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Check if client is connected before attempting to publish
                if not self._is_connected:
                    logger.warning(
                        f"Client not connected (attempt {attempt + 1}/{max_retries}), "
                        f"waiting for reconnection..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                    continue
                
                info = self._mqtt_client.publish(topic, payload, qos=MqttQoS.EXACTLY_ONCE)
                
                # Wait for publish to complete with timeout
                timeout_counter = 0
                max_wait_iterations = 100  # 100 seconds max wait
                
                while not info.is_published():
                    if timeout_counter >= max_wait_iterations:
                        raise TimeoutError(f"Publish timeout after {max_wait_iterations} seconds")
                    
                    logger.debug(f"waiting for publish completion: rc = {info.rc}")
                    retval = self._mqtt_client.loop(MQTT_LOOP_CHECK_PERIOD)
                    logger.debug(f"retval from loop = {retval}")
                    timeout_counter += 1
                
                logger.debug(f"sending chunk {seqno} to {topic} is done")
                return  # Success, exit retry loop
                
            except RuntimeError as e:
                if "not currently connected" in str(e):
                    logger.warning(
                        f"Publish failed due to disconnection (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to send chunk {seqno} to {topic} after {max_retries} attempts"
                        )
                        raise
                else:
                    # Different RuntimeError, re-raise immediately
                    raise
            except TimeoutError as e:
                logger.warning(
                    f"Publish timeout (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(
                        f"Failed to send chunk {seqno} to {topic} after {max_retries} attempts due to timeout"
                    )
                    raise
            except Exception as e:
                logger.error(f"Unexpected error sending chunk {seqno} to {topic}: {e}")
                raise
        
        # If we exit the loop without returning, all retries failed
        raise RuntimeError(f"Failed to send chunk {seqno} to {topic} after {max_retries} retries")

    async def send_chunk_async(
        self, topic: str, channel_name: str, data: bytes, seqno: int, eom: bool
    ) -> None:
        """Send a chunk asynchronously with retry logic for disconnections.
        
        This async version allows the event loop to process keepalive messages
        during chunk transmission, preventing disconnections on heavy load.
        """
        msg = msg_pb2.Data()
        msg.end_id = self._id
        msg.channel_name = channel_name
        msg.payload = data
        msg.seqno = seqno
        msg.eom = eom

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        max_retries = 5
        retry_delay = 1.0  # initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Check if client is connected before attempting to publish
                if not self._is_connected:
                    logger.warning(
                        f"Client not connected (attempt {attempt + 1}/{max_retries}), "
                        f"waiting for reconnection..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                    continue
                
                info = self._mqtt_client.publish(topic, payload, qos=MqttQoS.EXACTLY_ONCE)
                
                # Wait for publish to complete with timeout
                # Use async sleep to allow event loop to breathe
                timeout_counter = 0
                max_wait_iterations = 100  # 100 seconds max wait
                
                while not info.is_published():
                    if timeout_counter >= max_wait_iterations:
                        raise TimeoutError(f"Publish timeout after {max_wait_iterations} seconds")
                    
                    logger.debug(f"waiting for publish completion: rc = {info.rc}")
                    retval = self._mqtt_client.loop(MQTT_LOOP_CHECK_PERIOD)
                    logger.debug(f"retval from loop = {retval}")
                    
                    # CRITICAL: Yield to event loop to allow misc_loop() to send keepalives
                    # This prevents broker disconnections during long send operations
                    await asyncio.sleep(0)
                    timeout_counter += 1
                
                logger.debug(f"sending chunk {seqno} to {topic} is done")
                return  # Success, exit retry loop
                
            except RuntimeError as e:
                if "not currently connected" in str(e):
                    logger.warning(
                        f"Publish failed due to disconnection (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to send chunk {seqno} to {topic} after {max_retries} attempts"
                        )
                        raise
                else:
                    # Different RuntimeError, re-raise immediately
                    raise
            except TimeoutError as e:
                logger.warning(
                    f"Publish timeout (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(
                        f"Failed to send chunk {seqno} to {topic} after {max_retries} attempts due to timeout"
                    )
                    raise
            except Exception as e:
                logger.error(f"Unexpected error sending chunk {seqno} to {topic}: {e}")
                raise
        
        # If we exit the loop without returning, all retries failed
        raise RuntimeError(f"Failed to send chunk {seqno} to {topic} after {max_retries} retries")

    async def cleanup(self):
        """Clean up resources in backend."""
        # NOTE: DG attempted implementation to fix issue Stop MQTT client loop
        if self._mqtt_client is not None:
            self._mqtt_client.loop_stop()

        # Disconnect MQTT client
        if self._mqtt_client is not None:
            self._mqtt_client.disconnect()

        # Clear channels
        self._channels.clear()

        # Cancel asyncio tasks
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

        await self._loop.shutdown_asyncgens()
        self._loop.stop()

    def set_cleanup_ready_async(self, end_id: str) -> None:
        """Set cleanup ready event for a given end id."""
        # NOTE: DG attempted implementation to fix issue
        logger.debug(f"Setting cleanup ready for {end_id} in async loop")
        self._cleanup_ready.add(end_id)

    def set_cleanup_ready(self, end_id: str) -> None:
        """Set cleanup ready event for a given end id."""
        # NOTE: DG attempted implementation to fix issue
        logger.debug(f"Setting cleanup ready for {end_id} from non-loop thread")
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self.set_cleanup_ready_async, end_id)
        else:
            self.set_cleanup_ready_async(end_id)


class AsyncioHelper:
    """Asyncio helper class.

    Asyncio MQTT client example from
    https://github.com/eclipse/paho.mqtt.python/blob/master/examples/loop_asyncio.py
    """

    def __init__(self, loop, client):
        """Initialize AsyncioHelper instance."""
        self.loop = loop
        self.client = client
        self.client.on_socket_open = self.on_socket_open
        self.client.on_socket_close = self.on_socket_close
        self.client.on_socket_register_write = self.on_socket_register_write
        self.client.on_socket_unregister_write = self.on_socket_unregister_write

    def on_socket_open(self, client, userdata, sock):
        """Call a callback function when socket opens."""
        logger.debug("Socket opened")

        def cb():
            logger.debug("Socket is readable, calling loop_read")
            client.loop_read()

        self.loop.add_reader(sock, cb)
        self.misc = self.loop.create_task(self.misc_loop())

    def on_socket_close(self, client, userdata, sock):
        """Call a callback function when socket closes."""
        logger.debug("Socket closed")
        self.loop.remove_reader(sock)
        self.misc.cancel()

    def on_socket_register_write(self, client, userdata, sock):
        """Watch socket's writability."""
        logger.debug("Watching socket for writability.")

        def cb():
            logger.debug("Socket is writable, calling loop_write")
            client.loop_write()

        self.loop.add_writer(sock, cb)

    def on_socket_unregister_write(self, client, userdata, sock):
        """Stop watching socket's writability."""
        logger.debug("Stop watching socket for writability.")
        self.loop.remove_writer(sock)

    async def misc_loop(self):
        """Start misc loop."""
        logger.debug("misc_loop started")
        while self.client.loop_misc() == mqtt.MQTT_ERR_SUCCESS:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
        logger.debug("misc_loop finished")