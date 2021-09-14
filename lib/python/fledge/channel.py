# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import cloudpickle

from .common.util import run_async
from .config import GROUPBY_DEFAULT_GROUP

RXQ = 'rxq'
TXQ = 'txq'


class Channel(object):
    def __init__(
        self,
        backend,
        job_id: str,
        name: str,
        me='',
        other='',
        groupby=GROUPBY_DEFAULT_GROUP
    ):
        self._backend = backend
        self._job_id = job_id
        self._name = name
        self._my_role = me
        self._other_role = other
        self._groupby = groupby

        # _ends must be accessed within backend's loop
        self._ends = {}

    def job_id(self):
        return self._job_id

    def name(self):
        return self._name

    def my_role(self):
        return self._my_role

    def other_role(self):
        return self._other_role

    def groupby(self):
        return self._groupby

    '''
    ### The following are not asyncio methods
    ### But access to _ends variable should take place in the backend loop
    ### Therefore, when necessary, coroutine is defined inside each method
    ### and the coroutine is executed via run_async()
    '''

    def ends(self):
        async def inner():
            return list(self._ends.keys())

        result, _ = run_async(inner(), self._backend.loop())
        return result

    def broadcast(self, msg):
        for end_id in self.ends():
            self.send(end_id, msg)

    def send(self, end_id, message):
        '''
        send() is a blocking call to send a message to an end
        '''
        async def _put():
            if not self.has(end_id):
                # can't send message to end_id
                return

            payload = cloudpickle.dumps(message)
            await self._ends[end_id][TXQ].put(payload)

        _, status = run_async(_put(), self._backend.loop())

        return status

    def recv(self, end_id):
        '''
        recv() is a blocking call to receive a message from an end
        '''
        async def _get():
            if not self.has(end_id):
                # can't receive message from end_id
                return None

            rxq = self._ends[end_id][RXQ]
            payload = await rxq.get()
            rxq.task_done()
            return payload

        payload, status = run_async(_get(), self._backend.loop())

        return cloudpickle.loads(payload) if payload and status else None

    '''
    ### The following are asyncio methods of backend loop
    ### Therefore, they must be called in the backend loop
    '''

    async def add(self, end_id):
        if self.has(end_id):
            return

        self._ends[end_id] = {RXQ: asyncio.Queue(), TXQ: asyncio.Queue()}

        # create tx task in the backend for the channel
        self._backend.create_tx_task(self._name, end_id)

    async def remove(self, end_id):
        if not self.has(end_id):
            return

        rxq = self._ends[end_id][RXQ]
        del self._ends[end_id]

        # put bogus data to unblock a get() call
        await rxq.put(b'')

    def has(self, end_id):
        return end_id in self._ends

    def get_q(self, end_id, qtype):
        return self._ends[end_id][qtype]
