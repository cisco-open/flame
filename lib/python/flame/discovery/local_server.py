# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import asyncio

from ..common.comm import _recv_msg, _send_msg
from ..common.constants import UNIX_SOCKET_PATH
from ..proto import registry_msg_pb2 as msg_pb2


class JobState(object):
    def __init__(self, job):
        self.job = job
        self.state = {}

    def update(self, channel, role, uid, endpoint):
        if channel not in self.state:
            self.state[channel] = {}
        if role not in self.state[channel]:
            self.state[channel][role] = {}

        self.state[channel][role][uid] = endpoint

    def get(self, channel):
        if channel not in self.state:
            return None

        return self.state[channel]


class LocalRegistry(object):
    def __init__(self):
        self._server = None
        self.state = {}

    def _handle_get(self, msg):
        result = None
        for rec in msg.record:
            if rec.job in self.state:
                result = self.state[rec.job].get(rec.channel)

        resp = msg_pb2.Response()

        resp.req_seq = msg.req_seq
        resp.status = msg_pb2.Response.OK

        for role, uidep in result.items():
            uideps = msg_pb2.UidEndpoints()
            for uid, ep in uidep.items():
                uideps.uid.append(uid)
                uideps.endpoint.append(ep)

            resp.role_to_uidep[role].CopyFrom(uideps)

        print(resp)

        return resp

    def _handle_set(self, msg):
        for rec in msg.record:
            if rec.job not in self.state:
                self.state[rec.job] = JobState(rec.job)

            job_state = self.state[rec.job]

            job_state.update(rec.channel, rec.role, msg.uid, msg.endpoint)

        resp = msg_pb2.Response()

        resp.req_seq = msg.req_seq
        resp.status = msg_pb2.Response.OK

        return resp

    async def _handle(self, reader, writer):
        while not reader.at_eof():
            any_msg = await _recv_msg(reader)
            if any_msg is None:
                continue

            if any_msg.Is(msg_pb2.Get.DESCRIPTOR):
                msg = msg_pb2.Get()
                any_msg.Unpack(msg)
                print('>>> Get', msg)

                resp = self._handle_get(msg)
                await _send_msg(writer, resp)

            elif any_msg.Is(msg_pb2.Set.DESCRIPTOR):
                msg = msg_pb2.Set()
                any_msg.Unpack(msg)
                print('>>> Set', msg)

                resp = self._handle_set(msg)
                print('>>> response', resp)
                await _send_msg(writer, resp)

            else:
                print('unknown message')

        writer.close()
        await writer.wait_closed()

        print('connection closed')

    async def start(self):
        self._server = await asyncio.start_unix_server(
            self._handle, path=UNIX_SOCKET_PATH
        )

        addr = self._server.sockets[0].getsockname()
        print(f'Serving on {addr}')

        async with self._server:
            await self._server.serve_forever()


async def main():
    registry = LocalRegistry()
    await registry.start()


if __name__ == "__main__":
    asyncio.run(main())
