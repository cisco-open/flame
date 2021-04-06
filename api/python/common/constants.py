from enum import Enum

MSG_LEN_FIELD_SIZE = 4
SOCK_OP_WAIT_TIME = 10  # 10 seconds

UNIX_SOCKET_PATH = '/tmp/local_registry.socket'


class SockType(Enum):
    CLIENT = 1
    SERVER = 2
