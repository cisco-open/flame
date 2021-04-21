import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from ....channel_manager import ChannelManager

FRONTEND = 'keras'
BACKEND = 'local'
REGISTRY_AGENT = 'local'
CHANNEL_NAME = 'param-channel'
JOB_NAME = 'keras-mnist-job'
MY_ROLE = 'trainer'
OTHER_ROLE = 'aggregator'
CHANNELS_ROLES = {CHANNEL_NAME: ((MY_ROLE, OTHER_ROLE), (OTHER_ROLE, MY_ROLE))}

# keras mnist example from https://keras.io/examples/vision/mnist_convnet/


class Trainer(object):
    def __init__(self, n_split=1, split_idx=0, rounds=1):
        self.cm = ChannelManager()
        self.cm(
            FRONTEND, BACKEND, REGISTRY_AGENT, JOB_NAME, MY_ROLE, CHANNELS_ROLES
        )
        self.cm.join(CHANNEL_NAME)

        self._n_split = n_split
        self._split_idx = split_idx
        self._rounds = rounds

        self._split_weight = 0
        for i in range(self._n_split):
            self._split_weight += 2**i
        self._split_weight = 1 / self._split_weight

    def prepare(self):
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

        def get_split_array(arr):
            split_array = []
            arr_idx = 0
            for idx in range(self._n_split):
                arr_idx += int((len(arr)) * self._split_weight * (idx + 1))
                split_array.append(arr_idx)

            return split_array

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        split_array = get_split_array(x_train)
        x_train = np.split(x_train, split_array)[self._split_idx]

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        split_array = get_split_array(y_train)
        y_train = np.split(y_train, split_array)[self._split_idx]

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print("y_train shape:", y_train.shape)
        print(y_train.shape[0], "train samples")

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

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        self._model = model
        self._x_train = x_train
        self._y_train = y_train

    def train(self):
        batch_size = 128
        epochs = 1

        self._model.fit(
            self._x_train,
            self._y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1
        )

    def run(self):
        self.prepare()
        channel = self.cm.get(CHANNEL_NAME)
        msg = channel.message_object()

        i = 0
        while i < self._rounds:
            ends = channel.ends()
            if len(ends) == 0:
                time.sleep(1)
                continue

            print(f'>>> round {i+1}')

            # one aggregator is sufficient
            end = ends[0]
            msg = channel.recv(end)

            weights = msg.get_state()
            self._model.set_weights(weights)
            self.train()

            # craft a message to inform aggregator
            msg.set_state(self._model.get_weights())
            msg.set_attr('count', len(self._x_train))
            channel.send(end, msg)

            # increase round
            i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--n_split',
        type=int,
        default=1,
        help='number of splits of a training dataset'
    )
    parser.add_argument(
        '--split_idx',
        type=int,
        default=0,
        help='index of split between 0 and (n_split-1)'
    )
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    trainer = Trainer(args.n_split, args.split_idx, args.rounds)
    trainer.run()
