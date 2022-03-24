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

from ....common.util import install_packages

install_packages(['monai', 'sklearn', 'tqdm'])

# example cmd: python3 -m flame.examples.mednist.trainer.main --n_split 2 --rounds 3 --split_idx 0
# run the above command in flame/lib/python folder
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
        'flame/examples/mednist/trainer/config.json',
        args.n_split,
        args.split_idx,
        args.rounds,
    )
    trainer.run()
