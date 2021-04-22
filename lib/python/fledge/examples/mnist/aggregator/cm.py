from ....channel_manager import ChannelManager

FRONTEND = 'keras'
BACKEND = 'local'
REGISTRY_AGENT = 'local'
CHANNEL_NAME = 'param-channel'
JOB_NAME = 'keras-mnist-job'
MY_ROLE = 'aggregator'
OTHER_ROLE = 'trainer'
CHANNELS_ROLES = {CHANNEL_NAME: ((MY_ROLE, OTHER_ROLE), (OTHER_ROLE, MY_ROLE))}


class CM(ChannelManager):
    def __init__(self):
        super().__init__()
        self.cm = ChannelManager()
        super().__call__(
            FRONTEND, BACKEND, REGISTRY_AGENT, JOB_NAME, MY_ROLE, CHANNELS_ROLES
        )
        super().join(CHANNEL_NAME)
