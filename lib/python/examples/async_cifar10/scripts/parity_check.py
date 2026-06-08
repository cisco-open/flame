"""Stable CLI entry point — thin shim over scripts.parity.cli.

Usage:
    python scripts/parity_check.py --real <dir> --sim <dir> [options]

See ``scripts/parity/cli.py`` for the full option list, or run with --help.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parity.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
