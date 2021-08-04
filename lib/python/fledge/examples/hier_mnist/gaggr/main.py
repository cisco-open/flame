import logging
import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from ....channel_manager import ChannelManager

logger = logging.getLogger(__name__)

# keras mnist example from https://keras.io/examples/vision/mnist_convnet/


class GlobalAggregator(object):
    def __init__(self, config_file: str, rounds=1):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('global-channel')

        self._rounds = rounds

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
        (x_test.shape[0], "test samples")

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
        logger.info(f'Test loss: {score[0]}')
        logger.info(f'Test accuracy: {score[1]}')

    def run(self):
        self.prepare()
        channel = self.cm.get('global-channel')

        i = 0
        while i < self._rounds:
            logger.info(f'>>> round {i+1}')
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
                    logger.info('no data received')
                    continue

                weights = msg[0]
                count = msg[1]
                if count == 0:
                    continue

                total += count
                weights_array.append((weights, count))
                logger.info(
                    f'got {end}\'s parameters trained with {count} samples'
                )

            if len(weights_array) == 0 or total == 0:
                logger.info('no local model parameters are obtained')
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


# example cmd in the following:
# python3 -m fledge.examples.hier_mnist.gaggr.main --config fledge/examples/hier_mnist/gaggr/config.json --rounds 3
# run the above command in fledge/lib/python folder
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    gaggr = GlobalAggregator(args.config, args.rounds)
    gaggr.run()
