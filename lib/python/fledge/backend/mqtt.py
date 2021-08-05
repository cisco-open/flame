import asyncio
import hashlib
import logging
import time
import uuid
from enum import IntEnum
from typing import Tuple

import paho.mqtt.client as mqtt
from google.protobuf.any_pb2 import Any
from paho.mqtt.client import MQTTv5

from ..channel import RXQ, TXQ, Channel
from ..common.constants import MQTT_TOPIC_PREFIX, BackendEvent
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

# message's signature is valid for 1 sec
MSG_SIG_PERIOD = 1  # 1 sec

logger = logging.getLogger(__name__)


# info for mqtt qos is found at
# https://www.hivemq.com/blog/mqtt-essentials-part-6-mqtt-quality-of-service-levels/
class MqttQoS(IntEnum):
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class MqttBackend(AbstractBackend):
    '''
    MqttBackend only allows a singleton instance
    '''
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
        if cls._instance is None:
            logger.info('creating an MqttBackend instance')
            cls._instance = super(MqttBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._channels = {}
        self._cleanup_waits = {}
        self._id = str(uuid.uuid1())

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

            coro = self._monitor_end_termination()
            _ = asyncio.create_task(coro)

        coro = _init_loop_stuff()
        _ = run_async(coro, self._loop, 1)

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

    def configure(self, broker: str, job_id: str):
        self._broker = broker
        self._job_id = job_id
        self._health_check_topic = f'{MQTT_TOPIC_PREFIX}/{self._job_id}'

        client = mqtt.Client(self._id, protocol=MQTTv5)
        client.on_connect = self.on_connect
        client.will_set(
            self._health_check_topic,
            payload=f'{self._id}:{END_STATUS_OFF}',
            qos=MqttQoS.EXACTLY_ONCE
        )
        client.connect(self._broker)
        client.subscribe(self._health_check_topic)
        client.loop_start()

        self._mqtt_client = client
        self._last_payload_sig = {}
        self._msg_chunks = {}

    def uid(self):
        '''
        return backend id
        '''
        return self._id

    def eventq(self):
        return self._eventq

    def on_connect(self, client, userdata, flags, rc, properties=None):
        # publish health data; format: <end_id>:<status>
        # status is either END_STATUS_ON or END_STATUS_OFF
        client.publish(
            self._health_check_topic,
            payload=f'{self._id}:{END_STATUS_ON}',
            qos=MqttQoS.EXACTLY_ONCE
        )

    def on_message(self, client, userdata, message):
        if message.topic == self._health_check_topic:
            health_data = str(message.payload.decode("utf-8"))
            # the correct format of health data: <end_id>:<status>
            (end_id, status) = health_data.split(':')[0:2]
            if end_id == self._id or status == END_STATUS_ON:
                # nothing to do
                return

            async def _add_cleanup_waits(end_id):
                expiry = time.time() + MQTT_TIME_WAIT
                self._cleanup_waits[end_id] = expiry

            # add end_id to _cleanup_waits dictionary
            _ = run_async(_add_cleanup_waits(end_id), self._loop)

            return

        logger.debug(
            f'topic: {message.topic}; paylod length: {len(message.payload)}'
        )
        any_msg = Any().FromString(message.payload)

        if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
            msg = msg_pb2.Notify()
            any_msg.Unpack(msg)

            if msg.channel_name not in self._channels:
                logger.debug('channel not found')
                return

            channel = self._channels[msg.channel_name]

            async def _has():
                return channel.has(msg.end_id)

            has_end, _ = run_async(_has(), self._loop)
            if not has_end:
                # this is the first time to see this end,
                # so let's notify my presence to the end
                self.notify(msg.channel_name)

            async def _add():
                await channel.add(msg.end_id)

            _ = run_async(_add(), self._loop)

        elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
            msg = msg_pb2.Data()
            any_msg.Unpack(msg)

            async def _update_cleanup_waits(end_id):
                # update is needed only if end_id's termination is detected
                if end_id in self._cleanup_waits:
                    expiry = time.time() + MQTT_TIME_WAIT
                    self._cleanup_waits[end_id] = expiry

            # update end_id to _cleanup_waits dictionary
            _ = run_async(_update_cleanup_waits(msg.end_id), self._loop)

            payload, fully_assembled = self.assemble_chunks(msg)

            channel = self._channels[msg.channel_name]

            async def _put():
                rxq = channel.get_q(msg.end_id, RXQ)
                await rxq.put(payload)

            if fully_assembled:
                logger.debug(f'fully assembled data size = {len(payload)}')
                _ = run_async(_put(), self._loop)
        else:
            logger.debug('unknown message type')

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
        self._mqtt_client.subscribe(topic, qos=MqttQoS.EXACTLY_ONCE)
        self._mqtt_client.on_message = self.on_message

    def notify(self, channel_name):
        if channel_name not in self._channels:
            return False

        channel = self._channels[channel_name]

        topic = self.topic_for_pub(channel)

        msg = msg_pb2.Notify()
        msg.end_id = self._id
        msg.channel_name = channel_name

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        self._mqtt_client.publish(topic, payload, qos=MqttQoS.EXACTLY_ONCE)

        return True

    def loop(self):
        return self._loop

    def add_channel(self, channel):
        self._channels[channel.name()] = channel

    def create_tx_task(self, channel_name, end_id):
        if (
            channel_name not in self._channels or
            not self._channels[channel_name].has(end_id)
        ):
            return False

        channel = self._channels[channel_name]

        coro = self._tx_task(channel, end_id)
        _ = asyncio.create_task(coro)

        return True

    async def _tx_task(self, channel, end_id):
        '''
        _tx_task() must be created per tx queue right after end_id is added to
        channel (e.g., channel.add(end_id))
        '''
        txq = channel.get_q(end_id, TXQ)

        while True:
            data = await txq.get()

            self.send_chunks(channel, data)

            txq.task_done()

    def send_chunks(self, channel: Channel, data: bytes) -> None:
        # TODO: This is hack. Need to revisit it later.
        #       The issue is that a topic is a broadcast medium by nature
        #       in mqtt. But a channel manintains a list of ends and
        #       can try to send a same message to each end.
        #       This behavior is essentailly multiple, repeated transmissions
        #       of the same message to all ends.
        #       This check can be in channel, but the code/interface becomes
        #       ugly as channel needs to know what is the underlying backend
        #       and handle situations differently.
        name = channel.name()
        digest = hashlib.md5(data).hexdigest()
        if (
            name in self._last_payload_sig and
            self._last_payload_sig[name][0] == digest and
            self._last_payload_sig[name][1] + MSG_SIG_PERIOD >= time.time()
        ):
            logger.info('Last seen payload; skipping tx')
            return

        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            self.send_chunk(channel, chunk, seqno, eom)

        # update payload signature
        self._last_payload_sig[name] = [digest, time.time()]

    def send_chunk(
        self, channel: Channel, data: bytes, seqno: int, eom: bool
    ) -> None:
        msg = msg_pb2.Data()
        msg.end_id = self._id
        msg.channel_name = channel.name()
        msg.seqno = seqno
        msg.eom = eom
        msg.payload = data

        any = Any()
        any.Pack(msg)
        payload = any.SerializeToString()

        topic = self.topic_for_pub(channel)
        info = self._mqtt_client.publish(
            topic, payload, qos=MqttQoS.EXACTLY_ONCE
        )
        info.wait_for_publish()

        logger.debug(f'sending chunk {seqno} to {topic} is done')

    def topic_for_pub(self, ch: Channel):
        sep = '/'
        topic = sep.join(
            [
                f'{MQTT_TOPIC_PREFIX}',
                f'{ch.job_id()}',
                f'{ch.name()}',
                f'{ch.groupby()}',
                f'{ch.my_role()}',
                f'{self._id}',
            ]
        )

        return topic
