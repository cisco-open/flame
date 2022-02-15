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
"""Channel manager."""

import asyncio
import atexit
import logging
from typing import Optional

from .backends import backend_provider
from .channel import Channel
from .common.constants import DEFAULT_RUN_ASYNC_WAIT_TIME, BackendEvent
from .common.util import background_thread_loop, run_async
from .config import Config
from .discovery_clients import discovery_client_provider
from .selectors import selector_provider

logger = logging.getLogger(__name__)


class ChannelManager(object):
    """ChannelManager manages channels and creates a singleton instance."""

    _instance = None

    _config = None
    _job_id = None
    _role = None

    _channels = None

    _loop = None

    _backend = None
    _discovery_client = None
    _selector = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            logger.info('creating a ChannelManager instance')
            cls._instance = super(ChannelManager, cls).__new__(cls)
        return cls._instance

    def __call__(self, config: Config):
        """Initialize instance variables."""
        self._config = config
        self._job_id = self._config.job.job_id
        self._role = self._config.role
        self._agent_id = self._config.agent_id

        self._channels = {}

        with background_thread_loop() as loop:
            self._loop = loop

        self._selector = selector_provider.get(self._config.selector.sort,
                                               **self._config.selector.kwargs)

        self._backend = backend_provider.get(self._config.backend)
        broker = self._config.brokers.sort_to_host[self._config.backend]
        self._backend.configure(broker, self._job_id, self._agent_id)
        self._discovery_client = discovery_client_provider.get(
            self._config.agent)

        async def inner():
            # create a coroutine task
            coro = self._backend_eventq_task(self._backend.eventq())
            _ = asyncio.create_task(coro)

        _ = run_async(inner(), self._backend.loop())

        atexit.register(self.cleanup)

    async def _backend_eventq_task(self, eventq):
        while True:
            (event_type, info) = await eventq.get()

            if event_type == BackendEvent.DISCONNECT:
                for _, channel in self._channels.items():
                    await channel.remove(info)

    def join_all(self) -> None:
        """join_all ensures that a role joins all of its channels."""
        for ch_name in self._config.channels.keys():
            self.join(ch_name)

    def join(self, name: str) -> bool:
        """Join a channel."""
        if self.is_joined(name):
            return True

        channel_config = self._config.channels[name]

        if self._role == channel_config.pair[0]:
            me = channel_config.pair[0]
            other = channel_config.pair[1]
        else:
            me = channel_config.pair[1]
            other = channel_config.pair[0]

        groupby = channel_config.groupby.groupable_value(self._config.realm)

        self._channels[name] = Channel(self._backend, self._selector,
                                       self._job_id, name, me, other, groupby)
        self._channels[name].join()

    # TODO: groupby feature with non-mqtt backend should be implemented
    # TODO: _join_non_mqtt() is non-functioning deprecated code;
    #       now it's not called; remove or refactor it
    def _join_non_mqtt(self, name):
        """Join a channel when backend is not mqtt."""
        coro = self._discovery_client.connect()
        _, status = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not status:
            return False

        coro = self._discovery_client.register(self._job_id, name, self._role,
                                               self._backend.uid(),
                                               self._backend.endpoint())
        _, status = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if status:
            self._channels[name] = Channel(self._backend, self._selector,
                                           self._job_id, name)
            self._backend.attach_channel(self._channels[name])
        else:
            return False

        coro = self._discovery_client.get(self._job_id, name)
        channel_info, status = run_async(coro, self._loop,
                                         DEFAULT_RUN_ASYNC_WAIT_TIME)
        if not status:
            return False

        # connect to other ends to complete join to channels
        for role, end_id, endpoint in channel_info:
            # the same backend id; skip
            if end_id is self._backend.uid():
                continue

            channel_config = self._config.channels[name]
            one = channel_config.pair[0]
            other = channel_config.pair[1]

            # doesn't match channel config; skip connection
            if ((one != self._role or other != role)
                    and (one != role or other != self._role)):
                continue

            # connect to endpoint
            self._backend.connect(end_id, endpoint)

            # notify end_id of the channel handled by the backend
            self._backend.notify(end_id, name)

            # update channel
            coro = self._channels[name].add(end_id)
            _ = run_async(coro, self._backend.loop())

        coro = self._discovery_client.close()
        _ = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)

        return True

    def leave(self, name):
        """Leave a channel."""
        if not self.is_joined(name):
            return

        # TODO: reset_channel isn't implemented; the whole discovery module
        #       needs to be revisited.
        coro = self._discovery_client.reset_channel(self._job_id, name,
                                                    self._role,
                                                    self._backend.uid())

        _, status = run_async(coro, self._loop, DEFAULT_RUN_ASYNC_WAIT_TIME)
        if status:
            del self._channels[name]

        return status

    def get_by_tag(self, tag: str) -> Optional[Channel]:
        """Return a channel object that matches a given function tag."""
        if tag not in self._config.func_tag_map:
            return None

        channel_name = self._config.func_tag_map[tag]
        return self.get(channel_name)

    def get(self, name: str) -> Optional[Channel]:
        """Return a channel object in a given channel name."""
        if not self.is_joined(name):
            # didn't join the channel yet
            return None

        return self._channels[name]

    def is_joined(self, name):
        """Check if node joined a channel or not."""
        if name in self._channels:
            return True
        else:
            return False

    def cleanup(self):
        """Clean up pending asyncio tasks."""
        for task in asyncio.all_tasks(self._backend.loop()):
            task.cancel()
