import time

from fledge.channel_manager import ChannelManager


class Bar(object):
    def __init__(self, config_file: str):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('simple-channel')

    def run(self):
        channel = self.cm.get('simple-channel')
        while True:
            for end in channel.ends():
                data = channel.recv(end)
                if not data:
                    print('no data received')
                    continue

                # data = msg.get_state()
                print(f'type = {type(data)}, data = {data}')

                data[:] = [i + 1 for i in data]
                channel.send(end, data)
            time.sleep(1)


# On fledge/lib/python, run python3 -m fledge.examples.simple.bar.main
if __name__ == "__main__":
    bar = Bar('/fledge/example/simple/bar/config.json')
    bar.run()
