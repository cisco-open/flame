import time
from fledge.channel_manager import ChannelManager


class Foo(object):
    def __init__(self, config_file: str):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('simple-channel')

    def run(self):
        data = [0, 1, 2, 3, 4, 5]
        channel = self.cm.get('simple-channel')

        while True:
            for end in channel.ends():
                channel.send(end, data)
                data = channel.recv(end)
                if not data:
                    print('no data received')
                    continue

                print(f'type = {type(data)}, data = {data}')
                data[:] = [i + 1 for i in data]
            time.sleep(1)


# On fledge/lib/python, run python3 -m fledge.examples.simple.foo.main
if __name__ == "__main__":
    foo = Foo('/fledge/example/simple/foo/config.json')
    foo.run()
