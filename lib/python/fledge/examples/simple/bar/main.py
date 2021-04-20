import time

from ....channel_manager import ChannelManager

FRONTEND = 'basic'
BACKEND = 'local'
REGISTRY_AGENT = 'local'
CHANNEL_NAME = 'simple-channel'
JOB_NAME = 'simple-job'
MY_ROLE = 'bar'
OTHER_ROLE = 'foo'
CHANNELS_ROLES = {CHANNEL_NAME: ((MY_ROLE, OTHER_ROLE), (OTHER_ROLE, MY_ROLE))}


class Bar(object):
    def __init__(self):
        self.cm = ChannelManager()
        self.cm(
            FRONTEND, BACKEND, REGISTRY_AGENT, JOB_NAME, MY_ROLE, CHANNELS_ROLES
        )
        self.cm.join(CHANNEL_NAME)

    def run(self):
        channel = self.cm.get(CHANNEL_NAME)
        while True:
            for end in channel.ends():
                msg = channel.recv(end)
                if not msg:
                    print('no data received')
                    continue

                data = msg.get_state()
                print(f'type = {type(msg)}, msg = {msg}, data = {data}')

                data[:] = [i + 1 for i in data]
                msg.set_state(data)
                channel.send(end, msg)
            time.sleep(1)


if __name__ == "__main__":
    bar = Bar()
    bar.run()
