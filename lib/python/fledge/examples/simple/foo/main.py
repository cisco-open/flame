import time

from ....channel_manager import ChannelManager

FRONTEND = 'basic'
BACKEND = 'local'
REGISTRY_AGENT = 'local'
CHANNEL_NAME = 'simple-channel'
JOB_NAME = 'simple-job'
MY_ROLE = 'foo'
OTHER_ROLE = 'bar'
CHANNELS_ROLES = {CHANNEL_NAME: ((MY_ROLE, OTHER_ROLE), (OTHER_ROLE, MY_ROLE))}


class Foo(object):
    def __init__(self):
        self.cm = ChannelManager()
        self.cm(
            FRONTEND, BACKEND, REGISTRY_AGENT, JOB_NAME, MY_ROLE, CHANNELS_ROLES
        )
        self.cm.join(CHANNEL_NAME)

    def run(self):
        msg = [0, 1, 2, 3, 4, 5]
        channel = self.cm.get(CHANNEL_NAME)
        while True:
            for end in channel.ends():
                channel.send(end, msg)
                msg = channel.recv(end)
                if not msg:
                    print('no data received')
                    continue

                print(f'type = {type(msg)}, msg = {msg}')
                msg[:] = [i + 1 for i in msg]
            time.sleep(1)


if __name__ == "__main__":
    foo = Foo()
    foo.run()
