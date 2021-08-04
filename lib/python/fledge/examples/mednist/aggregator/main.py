from ....common.util import install_packages

install_packages(['monai', 'sklearn', 'tqdm'])

# example cmd: python3 -m fledge.examples.mednist.aggregator.main --rounds 3
# run the above command in fledge/lib/python folder
if __name__ == "__main__":
    import argparse

    from .role import Aggregator

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--rounds', type=int, default=1, help='number of training rounds'
    )

    args = parser.parse_args()

    aggregator = Aggregator(
        'fledge/examples/mednist/aggregator/config.json',
        args.rounds,
    )
    aggregator.run()
