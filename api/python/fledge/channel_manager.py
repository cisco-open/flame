from .backends import backend_provider
from .channel import Channel
from .common.constants import SOCK_OP_WAIT_TIME
from .common.util import background_thread_loop, run_async
from .registry_agents import registry_agent_provider
from .serdeses import serdes_provider


class ChannelManager(object):
    def __init__(
        self, frontend, backend, registry_agent, job, role, channels_roles
    ):
        self.job = job
        self.role = role
        self.channels_roles = channels_roles

        self.channels = {}

        with background_thread_loop() as loop:
            self._loop = loop

        self._serdes = serdes_provider.get(frontend)
        self._backend = backend_provider.get(backend)
        self._registry_agent = registry_agent_provider.get(registry_agent)

    def join(self, name):
        '''
        joins a channel
        '''
        if self.is_joined(name):
            return True

        coro = self._registry_agent.connect()
        _, status = run_async(coro, self._loop, SOCK_OP_WAIT_TIME)
        if not status:
            return False

        coro = self._registry_agent.register(
            self.job, name, self.role, self._backend.uid(),
            self._backend.endpoint()
        )
        _, status = run_async(coro, self._loop, SOCK_OP_WAIT_TIME)
        if status:
            self.channels[name] = Channel(name, self._serdes, self._backend)
            self._backend.add_channel(self.channels[name])
        else:
            return False

        # role_tuples should have at most two entries
        role_tuples = self.channels_roles[name]

        coro = self._registry_agent.get(self.job, name)
        channel_info, status = run_async(coro, self._loop, SOCK_OP_WAIT_TIME)
        if not status:
            return False

        for role, end_id, endpoint in channel_info:
            # the same backend id; skip
            if end_id is self._backend.uid():
                continue

            proceed = False
            for from_role, to_role in role_tuples:
                proceed = self.role == from_role and role == to_role
                if proceed:
                    break
            if not proceed:
                continue

            # connect to endpoint
            self._backend.connect(end_id, endpoint)

            # notify end_id of the channel handled by the backend
            self._backend.notify(end_id, name)

            # update channel
            self.channels[name].add(end_id)

        coro = self._registry_agent.close()
        _ = run_async(coro, self._loop, SOCK_OP_WAIT_TIME)

        return True

    def leave(self, name):
        '''
        leave a channel
        '''
        if not self.is_joined(name):
            return

        coro = self._registry_agent.reset_channel(
            self.job, name, self.role, self._backend.uid()
        )

        _, status = run_async(coro, self._loop, SOCK_OP_WAIT_TIME)
        if status:
            del self.channels[name]

        return status

    def get(self, name):
        '''
        returns a channel object in the given channel
        '''
        if not self.is_joined(name):
            # didn't join the channel yet
            return None

        return self.channels[name]

    def is_joined(self, name):
        '''
        check if node joined a channel or not
        '''
        if name in self.channels:
            return True
        else:
            return False
