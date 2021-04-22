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
                msg = channel.message_object()
                msg.set_state(data)
                channel.send(end, msg)
                msg = channel.recv(end)
                if not msg:
                    print('no data received')
                    continue

                data = msg.get_state()
                print(f'type = {type(msg)}, msg = {msg}, data = {data}')
                data[:] = [i + 1 for i in data]
            time.sleep(1)


if __name__ == "__main__":
    foo = Foo()
    foo.run()
