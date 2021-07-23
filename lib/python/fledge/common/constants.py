from enum import Enum

MSG_LEN_FIELD_SIZE = 4
SOCK_OP_WAIT_TIME = 10  # 10 seconds

# backend related constants
MQTT_TOPIC_PREFIX = '/fledge'
UNIX_SOCKET_PATH = '/tmp/local_registry.socket'


class BackendEvent(Enum):
    DISCONNECT = 1
