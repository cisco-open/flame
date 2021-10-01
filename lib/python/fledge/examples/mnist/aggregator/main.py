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

import numpy as np
from fledge.channel_manager import ChannelManager
from fledge.config import Config
from tensorflow import keras
from tensorflow.keras import layers

# keras mnist example from https://keras.io/examples/vision/mnist_convnet/


class Aggregator(object):
    def __init__(self, config: Config):
        self.config = config
        self.cm = ChannelManager()
        self.cm(config)
        self.cm.join('param-channel')

        self._rounds = 5
        if 'rounds' in self.config.hyperparameters:
            self._rounds = self.config.hyperparameters['rounds']

    def prepare(self):
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_test = np.expand_dims(x_test, -1)
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # model.summary()

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        self._model = model
        self._x_test = x_test
        self._y_test = y_test

    def test(self):
        score = self._model.evaluate(self._x_test, self._y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def run(self):
        self.prepare()
        channel = self.cm.get('param-channel')

        i = 0
        while i < self._rounds:
            print(f'>>> round {i+1}')
            # send out global model parameters to trainers
            for end in channel.ends():
                channel.send(end, self._model.get_weights())

            # TODO: lines below need to be abstracted for different
            # frontends (e.g., keras, pytorch, etc)
            total = 0
            weights_array = []
            # receive local model parameters from trainers
            for end in channel.ends():
                msg = channel.recv(end)
                if not msg:
                    print('no data received')
                    continue

                weights = msg[0]
                count = msg[1]
                total += count
                weights_array.append((weights, count))
                print(f'got {end}\'s parameters trained with {count} samples')

            if len(weights_array) == 0 or total == 0:
                print('no local model parameters are obtained')
                time.sleep(1)
                continue

            count = weights_array[0][1]
            rate = count / total
            global_weights = [weight * rate for weight in weights_array[0][0]]

            for weights, count in weights_array[1:]:
                rate = count / total

                for idx in range(len(weights)):
                    global_weights[idx] += weights[idx] * rate

            self._model.set_weights(global_weights)
            self.test()
            i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)
    aggregator = Aggregator(config)

    print("Starting aggregator...")
    aggregator.run()
