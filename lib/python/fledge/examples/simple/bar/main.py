import time

from .cm import CHANNEL_NAME, CM


class Bar(object):
    def __init__(self):
        self.cm = CM()

    def run(self):
        channel = self.cm.get(CHANNEL_NAME)
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


if __name__ == "__main__":
    bar = Bar()
    bar.run()
