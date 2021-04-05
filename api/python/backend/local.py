import asyncio
import concurrent.futures
import struct
import uuid
from threading import Thread

from .abstract import AbstractBackend

MSG_LEN_FIELD_SIZE = 4
SOCK_OP_WAIT_TIME = 10
UNIX_SOCKET_PATH_TEMPLATE = '/tmp/fledge-{}.socket'


class LocalBackend(AbstractBackend):
    '''
    LocalBackend only allows a singleton instance
    '''
    _instance = None

    _initialized = False
    _endpoints = None

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
            target=self._loop_thread, args=(self._loop, ), daemon=True
        )
        self._thread.start()

        _ = asyncio.run_coroutine_threadsafe(self._setup_server(), self._loop)

    def _loop_thread(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _register_peer(self, reader, writer):
        data = await reader.read(MSG_LEN_FIELD_SIZE)
        msglen = struct.unpack('>I', data)[0]

        peer = await reader.read(msglen)
        peer = peer.decode()

        print(f'registering peer: {peer}')
        self._endpoints[peer] = (reader, writer)

    async def _setup_server(self):
        self._backend = UNIX_SOCKET_PATH_TEMPLATE.format(self._id)
        self._server = await asyncio.start_unix_server(
            self._register_peer, path=self._backend
        )

        name = self._server.sockets[0].getsockname()
        print(f'serving on {name}')

        async with self._server:
            await self._server.serve_forever()

    async def _connect(self, path):
        reader, writer = await asyncio.open_unix_connection(f'{path}')

        # register backend id to peer
        id = self._id.encode()
        data = struct.pack('>I', len(id)) + id

        writer.write(data)
        await writer.drain()

        return reader, writer

    async def _close(self, peer):
        if peer is not self._endpoints:
            return

        _, writer = self._endpoints[peer]
        if writer.is_closing():
            return

        writer.close()
        await writer.wait_closed()

    def uid(self):
        '''
        return backend id
        '''
        return self._id

    def endpoint(self):
        return self._backend

    def connect(self, peer, path):
        fut = asyncio.run_coroutine_threadsafe(self._connect(path), self._loop)
        try:
            reader, writer = fut.result(SOCK_OP_WAIT_TIME)
        except concurrent.futures.TimeoutError:
            # TODO: set more appropriate error type
            raise RuntimeError

        self._endpoints[peer] = (reader, writer)

    def close(self):
        for peer in list(self._endpoints):
            fut = asyncio.run_coroutine_threadsafe(
                self._close(peer), self._loop
            )

            try:
                _ = fut.result(SOCK_OP_WAIT_TIME)
            except concurrent.futures.TimeoutError:
                return

            del self._endpoints[peer]

            print(f'closed peer: {peer}')

    def send(self, peer, msg):
        pass

    def recv(self, peer):
        pass


if __name__ == "__main__":
    import time

    local_backend = LocalBackend()
    while True:
        time.sleep(5)
