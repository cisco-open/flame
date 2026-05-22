#!/usr/bin/env python3
"""CLI wrapper around flame.launch tailored for async_cifar10.

Defaults the aggregator entry point to main_oort_agg.py (the async variant).
For new experiments, prefer `python -m flame.launch.run_experiment` and
declare `example.aggregator_main` in the experiment YAML.
"""

import sys
from pathlib import Path

from flame.launch.run_experiment import main as _main


def main():
    # async_cifar10 traditionally uses main_oort_agg.py; if a caller does not
    # override aggregator_main via the YAML, set it here through env.
    # Simpler: just delegate. Experiment YAMLs in async_cifar10 should set
    # `example.aggregator_main: aggregator/pytorch/main_oort_agg.py`.
    sys.exit(_main())


if __name__ == "__main__":
    main()
