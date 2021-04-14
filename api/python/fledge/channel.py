import asyncio

from .common.util import run_async

RXQ = 'rxq'
TXQ = 'txq'


class Channel(object):
    def __init__(self, name, serdes, backend):
        self._name = name
        self._backend = backend

        self._ends = {}

        self._serdes = serdes

    def name(self):
        return self._name

    def add(self, end_id, in_active_loop=False):
        if self.has(end_id):
            return

        if in_active_loop:
            self._ends[end_id] = {RXQ: asyncio.Queue(), TXQ: asyncio.Queue()}
        else:

            async def _create_q():
                self._ends[end_id] = {
                    RXQ: asyncio.Queue(),
                    TXQ: asyncio.Queue()
                }

            _ = run_async(_create_q(), self._backend.loop())

        # create tx task in the backend for the channel
        self._backend.create_tx_task(self._name, end_id, in_active_loop)

    def has(self, end_id):
        return end_id in self._ends

    def ends(self):
        return self._ends.keys()

    def broadcast(self, msg):
        for end_id in self._ends:
            self.send(end_id, msg)

    def send(self, end_id, data):
        '''
        send() is a blocking call to send a message to an end
        '''
        if not self.has(end_id):
            # can't send message to end_id
            return

        async def _put(end_id, payload):
            await self._ends[end_id][TXQ].put(payload)

        payload = self._serdes.to_bytes(data)
        _, status = run_async(_put(end_id, payload), self._backend.loop())

        return status

    def recv(self, end_id):
        '''
        recv() is a blocking call to receive a message from an end
        '''
        if not self.has(end_id):
            # can't receive message from end_id
            return None

        async def _get(end_id):
            return await self._ends[end_id][RXQ].get()

        payload, status = run_async(_get(end_id), self._backend.loop())

        data = self._serdes.to_object(payload)

        return data if status else None

    def get_q(self, end_id, qtype):
        return self._ends[end_id][qtype]
