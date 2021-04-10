from ....channel_manager import ChannelManager

REGISTRY_AGENT = 'local'
BACKEND = 'local'
CHANNEL_NAME = 'param-channel'
JOB_NAME = 'fedavg'
MY_ROLE = 'aggregator'
CHANNELS_ROLES = {CHANNEL_NAME: ((MY_ROLE, 'trainer'), ('trainer', MY_ROLE))}


class Aggregator(object):
    def __init__(self):
        self.cm = ChannelManager(
            REGISTRY_AGENT, BACKEND, JOB_NAME, MY_ROLE, CHANNELS_ROLES
        )
        self.cm.join(CHANNEL_NAME)

    def run(self):
        pass


if __name__ == "__main__":
    aggregator = Aggregator()
    aggregator.run()
