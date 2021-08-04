import asyncio
import hashlib
import logging
import uuid
from typing import Tuple

import paho.mqtt.client as mqtt
from google.protobuf.any_pb2 import Any
from paho.mqtt.client import MQTTv5

from ..channel import RXQ, TXQ, Channel
from ..common.constants import MQTT_TOPIC_PREFIX
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from .abstract import AbstractBackend
from .chunk_store import ChunkStore

END_STATUS_ON = 'online'
END_STATUS_OFF = 'offline'

logger = logging.getLogger(__name__)


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

    def __new__(cls):
        if cls._instance is None:
            logger.info('creating an MqttBackend instance')
            cls._instance = super(MqttBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._channels = {}
        self._id = str(uuid.uuid1())

        with background_thread_loop() as loop:
            self._loop = loop

        async def _init_loop_stuff():
            self._eventq = asyncio.Queue()

        coro = _init_loop_stuff()
        _ = run_async(coro, self._loop, 1)

        self._initialized = True

    def configure(self, broker: str, job_id: str):
        self._broker = broker
        self._job_id = job_id
        self._health_check_topic = f'{MQTT_TOPIC_PREFIX}/{self._job_id}'

        client = mqtt.Client(self._id, protocol=MQTTv5)
        client.on_connect = self.on_connect
        client.will_set(
            self._health_check_topic,
            payload=f'{self._id}:{END_STATUS_OFF}',
            qos=0
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
            qos=0
        )

    def on_message(self, client, userdata, message):
        if message.topic == self._health_check_topic:
            health_data = str(message.payload.decode("utf-8"))
            # the correct format of health data: <end_id>:<status>
            (end_id, status) = health_data.split(':')[0:2]
            if end_id == self._id or status == END_STATUS_ON:
                # nothing to do
                return

            logger.info(f'end ({end_id}) status = {status}')

            async def _remove(channel, end_id):
                await channel.remove(end_id)

            # remove queues allocated for the end from all channels
            for channel in self._channels.values():
                _ = run_async(_remove(channel, end_id), self._loop)

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
                logger.debug(f'end ({msg.end_id}) seen first time')
                # this is the first time to see this end,
                # so let's notify my presence to the end
                self.notify(msg.channel_name)

            async def _add():
                await channel.add(msg.end_id)

            _ = run_async(_add(), self._loop)

        elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
            msg = msg_pb2.Data()
            any_msg.Unpack(msg)

            logger.debug(f'data message incoming from {msg.end_id}')
            channel = self._channels[msg.channel_name]

            payload, fully_assembled = self.assemble_chunks(msg)

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
        self._mqtt_client.subscribe(topic)
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

        self._mqtt_client.publish(topic, payload)

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
        digest = hashlib.md5(data).hexdigest()
        if (
            channel.name() in self._last_payload_sig and
            self._last_payload_sig[channel.name()] == digest
        ):
            logger.info('Last seen payload; skipping tx')
            return

        chunk_store = ChunkStore()
        chunk_store.set_data(data)

        logger.debug(f'sending data size = {len(data)}')
        while True:
            chunk, seqno, eom = chunk_store.get_chunk()
            if chunk is None:
                break

            self.send_chunk(channel, chunk, seqno, eom)

        # update payload signature
        self._last_payload_sig[channel.name()] = digest

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
        self._mqtt_client.publish(topic, payload)

        logger.debug(f'sending chunk {seqno} to {topic} is done')

    def topic_for_pub(self, ch: Channel):
        return f'{MQTT_TOPIC_PREFIX}/{ch.job_id()}/{ch.name()}/{ch.my_role()}/{self._id}'
