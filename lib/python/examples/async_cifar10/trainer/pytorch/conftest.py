# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Two sys.path entries main.py needs that pytest's default package-import
(rootdir=lib/python, so this module is examples.async_cifar10.trainer.pytorch)
doesn't provide on its own:

- this directory itself, so main.py's own sibling-style imports
  (`from memory_profiler import ...`) resolve, matching how it's actually
  launched in production (as a script, run from within this directory);
- `examples/async_cifar10/`, so tests here can import main.py via its full
  package path (`from trainer.pytorch.main import ...`) without needing
  `examples.async_cifar10.` prefixed on every import.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
