# DEPRECATED (2026-05-21)

The shell scripts in this directory directly invoke `pytorch/main*.py` and are
no longer maintained. The aggregator entrypoints they reference have been
renamed (`main.py` → `main_asyncfl_agg.py`, `main_oort_agg.py` →
`main_oort_sync_agg.py`), so these scripts will not run as-is.

Use the YAML-based launcher instead. It selects the correct aggregator stack
from the baseline and validates selector/stack compatibility:

    python -m flame.launch.run_experiment \
        examples/async_cifar10/expt_scripts_2026/<experiment>.yaml

Baselines (selector + optimizer + stack) live in
`examples/_metadata/baselines.yaml`; experiments declare `baseline: <name>` and
override only what they need.
