#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Generate a Dirichlet (non-IID) CIFAR-10 train split for N trainers.

Writes ``_metadata/dataset_splits/<dataset>_alpha<alpha>_n<N>.yaml`` in the schema
the launcher expects (``spawner.MetadataLoader.get_dataset_split``):

    dataset_name, dirichlet_alpha, num_trainers, total_samples,
    trainer_data_splits: { trainer_001: [idx, ...], ... }

Per-class Dirichlet(alpha) over trainers (smaller alpha => more heterogeneous).
Deterministic given --seed. Example:

    python gen_dirichlet_split.py --alpha 0.1 --num-trainers 50
"""
import argparse
import os

import numpy as np
import yaml
from torchvision.datasets import CIFAR10

_HERE = os.path.dirname(os.path.abspath(__file__))
_META = os.path.abspath(os.path.join(_HERE, "..", "..", "_metadata"))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "data"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--num-trainers", type=int, required=True)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=_DATA)
    ap.add_argument("--out-dir", default=os.path.join(_META, "dataset_splits"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    targets = np.array(CIFAR10(args.data_root, train=True, download=True).targets)
    n_classes = int(targets.max()) + 1
    n = args.num_trainers

    splits = {i: [] for i in range(n)}
    for c in range(n_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.full(n, args.alpha))
        # cut points (drop the trailing 1.0 boundary)
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for t, chunk in enumerate(np.split(idx, cuts)):
            splits[t].extend(int(x) for x in chunk)

    # Guarantee no empty trainer: steal one sample from the largest if needed.
    for t in range(n):
        if not splits[t]:
            donor = max(range(n), key=lambda k: len(splits[k]))
            splits[t].append(splits[donor].pop())

    trainer_data_splits = {
        f"trainer_{t + 1:03d}": sorted(splits[t]) for t in range(n)
    }
    doc = {
        "dataset_name": args.dataset,
        "dirichlet_alpha": args.alpha,
        "num_trainers": n,
        "total_samples": int(len(targets)),
        "trainer_data_splits": trainer_data_splits,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.dataset}_alpha{args.alpha}_n{n}.yaml")
    with open(out, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)

    sizes = [len(v) for v in trainer_data_splits.values()]
    print(f"wrote {out}")
    print(f"  trainers={n} total={sum(sizes)} "
          f"min={min(sizes)} max={max(sizes)} mean={sum(sizes)/n:.0f}")


if __name__ == "__main__":
    main()
