# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from ....channel_manager import ChannelManager


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


# On flame/lib/python, run python3 -m flame.examples.simple.foo.main
if __name__ == "__main__":
    foo = Foo('flame/examples/simple/foo/config.json')
    foo.run()
