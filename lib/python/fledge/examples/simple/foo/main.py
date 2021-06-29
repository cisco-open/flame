import time

from .cm import CHANNEL_NAME, CM


class Foo(object):
    def __init__(self):
        self.cm = CM()

    def run(self):
        data = [0, 1, 2, 3, 4, 5]
        channel = self.cm.get(CHANNEL_NAME)

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


if __name__ == "__main__":
    foo = Foo()
    foo.run()
