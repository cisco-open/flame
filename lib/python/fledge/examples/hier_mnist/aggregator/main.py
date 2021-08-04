import logging
import time

from ....channel_manager import ChannelManager

logger = logging.getLogger(__name__)


class Aggregator(object):
    def __init__(self, config_file: str, rounds=1):
        self.cm = ChannelManager()
        self.cm(config_file)
        self.cm.join('global-channel')
        self.cm.join('param-channel')

        self._rounds = rounds

    def get_global_model_weights(self):
        ends = []
        while len(ends) == 0:
            time.sleep(1)
            ends = self.global_channel.ends()
            continue

        # one aggregator is sufficient
        end = ends[0]
        weights = self.global_channel.recv(end)

        return weights

    def distribute_model_weights(self, weights):
        # send out global model parameters to trainers
        for end in self.param_channel.ends():
            self.param_channel.send(end, weights)

    def aggregate_model_weights(self):
        total = 0
        weights_array = []

        # receive local model parameters from trainers
        for end in self.param_channel.ends():
            msg = self.param_channel.recv(end)
            if not msg:
                logger.info('no data received')
                continue

            weights = msg[0]
            count = msg[1]
            total += count
            weights_array.append((weights, count))
            logger.info(f'got {end}\'s parameters trained with {count} samples')

        if len(weights_array) == 0 or total == 0:
            logger.info('no local model parameters are obtained')
            time.sleep(1)
            return None, 0

        count = weights_array[0][1]
        rate = count / total
        agg_weights = [weight * rate for weight in weights_array[0][0]]

        for weights, count in weights_array[1:]:
            rate = count / total

            for idx in range(len(weights)):
                agg_weights[idx] += weights[idx] * rate

        return agg_weights, total

    def send_weights_to_global_aggregator(self, weights, count):
        ends = self.global_channel.ends()
        # one global aggregator is sufficient
        end = ends[0]

        data = (weights, count)
        self.global_channel.send(end, data)

    def run(self):
        self.global_channel = self.cm.get('global-channel')
        self.param_channel = self.cm.get('param-channel')

        i = 0
        while i < self._rounds:
            weights = self.get_global_model_weights()
            if weights is None:
                continue

            logger.info(f'>>> round {i+1}')

            self.distribute_model_weights(weights)

            agg_weights, count = self.aggregate_model_weights()
            if agg_weights is None:
                continue

            self.send_weights_to_global_aggregator(agg_weights, count)

            i += 1


# example cmd: python3 -m fledge.examples.hier_mnist.aggregator.main --rounds 3
# run the above command in fledge/lib/python folder
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    aggregator = Aggregator(
        'fledge/examples/hier_mnist/aggregator/config.json',
        args.rounds,
    )
    aggregator.run()

    # There is a bug in mqtt backend implemtnation where a subscriber
    # fails to receive a message from a publisher when the publisher terminates.
    # This is due to the fact that mqtt last will message is used to signal
    # the termination of a node, which is an out-of-band mechanism.
    # The following is a simple hack used temporarily until a proper fix is
    # implemented.
    #
    # TODO: remove the following after the fix is implemented.
    while True:
        time.sleep(1)
