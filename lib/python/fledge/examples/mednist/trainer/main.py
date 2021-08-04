from ....common.util import install_packages

install_packages(['monai', 'sklearn', 'tqdm'])

# example cmd: python3 -m fledge.examples.mednist.trainer.main --n_split 2 --rounds 3 --split_idx 0
# run the above command in fledge/lib/python folder
if __name__ == "__main__":
    import argparse

    from .role import Trainer

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

    trainer = Trainer(
        'fledge/examples/mednist/trainer/config.json',
        args.n_split,
        args.split_idx,
        args.rounds,
    )
    trainer.run()
