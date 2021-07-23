import asyncio
import hashlib
import uuid

import paho.mqtt.client as mqtt
from google.protobuf.any_pb2 import Any

from ..channel import RXQ, TXQ, Channel
from ..common.constants import MQTT_TOPIC_PREFIX
from ..common.util import background_thread_loop, run_async
from ..proto import backend_msg_pb2 as msg_pb2
from .abstract import AbstractBackend

END_STATUS_ON = 'online'
END_STATUS_OFF = 'offline'


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

    def __new__(cls):
        if cls._instance is None:
            print('creating a MqttBackend instance')
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

        client = mqtt.Client(self._id)
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

    def uid(self):
        '''
        return backend id
        '''
        return self._id

    def eventq(self):
        return self._eventq

    def on_connect(self, client, userdata, flags, rc):
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

            async def _remove(channel, end_id):
                await channel.remove(end_id)

            # remove queues allocated for the end from all channels
            for channel in self._channels.values():
                _ = run_async(_remove(channel, end_id), self._loop)

            return

        any_msg = Any().FromString(message.payload)

        if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
            msg = msg_pb2.Notify()
            any_msg.Unpack(msg)

            if msg.channel_name not in self._channels:
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

            channel = self._channels[msg.channel_name]

            async def _put():
                rxq = channel.get_q(msg.end_id, RXQ)
                await rxq.put(msg.payload)

            _ = run_async(_put(), self._loop)

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
        name = channel.name()
        txq = channel.get_q(end_id, TXQ)

        while True:
            data = await txq.get()
            msg = msg_pb2.Data()
            msg.end_id = self._id
            msg.channel_name = name
            msg.payload = data

            any = Any()
            any.Pack(msg)
            payload = any.SerializeToString()

            # TODO: This is hack. Need to revisit it later.
            #       The issue is that a topic is a broadcast medium by nature
            #       in mqtt. But a channel manintains a list of ends and
            #       can try to send a same message to each end.
            #       This behavior is essentailly multiple, repeated transmissions
            #       of the same message to all ends.
            #       This check can be in channel, but the code/interface becomes
            #       ugly as channel needs to know what is the underlying backend
            #       and handle situations differently.
            digest = hashlib.md5(payload).hexdigest()
            if (
                name not in self._last_payload_sig or
                self._last_payload_sig[name] != digest
            ):
                topic = self.topic_for_pub(channel)
                self._mqtt_client.publish(topic, payload)
                self._last_payload_sig[name] = digest
            else:
                print('Last seen payload; skipping tx')

            txq.task_done()

    def topic_for_pub(self, ch: Channel):
        return f'{MQTT_TOPIC_PREFIX}/{ch.job_id()}/{ch.name()}/{ch.my_role()}/{self._id}'
