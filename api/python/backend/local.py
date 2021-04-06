import asyncio
import concurrent.futures
import struct
import uuid
from threading import Thread

from common.comm import _recv_msg, _send_msg
from common.constants import SOCK_OP_WAIT_TIME, SockType
from proto import backend_msg_pb2 as msg_pb2

from .abstract import AbstractBackend

UNIX_SOCKET_PATH_TEMPLATE = '/tmp/fledge-{}.socket'


class LocalBackend(AbstractBackend):
    '''
    LocalBackend only allows a singleton instance
    '''
    _instance = None

    _initialized = False
    _endpoints = None
    _queue = None

    _loop = None
    _thread = None
    _server = None

    _id = None
    _backend = None

    def __new__(cls):
        if cls._instance is None:
            print('creating a LocalBackend object')
            cls._instance = super(LocalBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        self._endpoints = {}
        self._id = str(uuid.uuid1())

        self._loop = asyncio.get_event_loop()

        self._thread = Thread(
            target=self._background_thread_loop,
            args=(self._loop, ),
            daemon=True
        )
        self._thread.start()

        _ = asyncio.run_coroutine_threadsafe(self._setup_server(), self._loop)

    def _background_thread_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _register_end(self, reader, writer):
        any_msg = await _recv_msg(self.reader)

        if not any_msg.Is(msg_pb2.Connect.DESCRIPTOR):
            # not expected message
            writer.close()
            await writer.wait_closed()
            return

        conn_msg = msg_pb2.Connect()
        any_msg.Unpack(conn_msg)

        print(f'registering end: {conn_msg.end_id}')
        self._endpoints[conn_msg.end_id] = (reader, writer, SockType.SERVER)

    async def _handle_msg(self, reader):
        any_msg = await _recv_msg(reader)

        if any_msg is None:
            return

        if any_msg.Is(msg_pb2.Notify.DESCRIPTOR):
            msg = msg_pb2.Notify()
            any_msg.Unpack(msg)
            print('>>> Notify', msg)
            # TODO: update channel associated with end_id

        elif any_msg.Is(msg_pb2.Data.DESCRIPTOR):
            msg = msg_pb2.Data()
            any_msg.Unpack(msg)
            print('>>> Data', msg)
            # TODO: put data into an asyncio queue

        else:
            print('unknown message')

    async def _handle(self, reader, writer):
        await self._register_end(reader, writer)

        while not reader.at_eof():
            await self._handle_msg(reader)

        writer.close()
        await writer.wait_closed()

    async def _setup_server(self):
        # create a queue with size 1 in a thread where _loop is running
        self._queue = asyncio.Queue(1)

        self._backend = UNIX_SOCKET_PATH_TEMPLATE.format(self._id)
        self._server = await asyncio.start_unix_server(
            self._handle, path=self._backend
        )

        name = self._server.sockets[0].getsockname()
        print(f'serving on {name}')

        async with self._server:
            await self._server.serve_forever()

    async def _connect(self, path):
        reader, writer = await asyncio.open_unix_connection(f'{path}')

        # register backend id to end
        msg = msg_pb2.Connect()
        msg.end_id = self._id

        await _send_msg(writer, msg)

        return reader, writer

    async def _notify(self, writer, channel_name):
        msg = msg_pb2.Notify()
        msg.end_id = self._id
        msg.channel_name = channel_name

        await _send_msg(writer, msg)

    async def _close(self, end_id):
        if end_id is not self._endpoints:
            return

        _, writer, _ = self._endpoints[end_id]
        if writer.is_closing():
            return

        writer.close()
        await writer.wait_closed()

    def _run_async(self, coro, timeout=None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.result(timeout)

    def uid(self):
        '''
        return backend id
        '''
        return self._id

    def endpoint(self):
        return self._backend

    def connect(self, end_id, endpoint):
        if end_id in self._endpoints:
            return True

        coro = self._connect(endpoint)
        try:
            reader, writer = self._run_async(coro, SOCK_OP_WAIT_TIME)
        except concurrent.futures.TimeoutError:
            return False

        self._endpoints[end_id] = (reader, writer, SockType.CLIENT)

        return True

    def nofity(self, end_id, channel_name):
        if end_id not in self._endpoints:
            return False

        _, writer, _ = self._endpoints[end_id]

        coro = self._notify(writer, channel_name)
        try:
            success = self._run_async(coro, SOCK_OP_WAIT_TIME)
        except concurrent.futures.TimeoutError:
            raise False

        return success

    def close(self):
        for end_id in list(self._endpoints):
            coro = self._close(end_id)
            try:
                _ = self._run_async(coro, SOCK_OP_WAIT_TIME)
            except concurrent.futures.TimeoutError:
                # ignore exception and move onto next endpoint
                pass

            del self._endpoints[end_id]

            print(f'closed end: {end_id}')

    def send(self, end_id, msg):
        pass

    def recv(self, end_id):
        pass


if __name__ == "__main__":
    import time

    local_backend = LocalBackend()
    while True:
        time.sleep(5)
