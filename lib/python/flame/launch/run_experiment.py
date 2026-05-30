# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CLI: python -m flame.launch.run_experiment <experiment.yaml>"""

import argparse
import sys
from pathlib import Path

from flame.launch.runner import ExperimentRunner


def _autodetect_example_dir(yaml_path: Path) -> Path:
    """Walk up parents until parent.name == 'examples'; else yaml's parent."""
    p = yaml_path.resolve()
    for parent in p.parents:
        if parent.parent.name == "examples":
            return parent
    return p.parent


def _force_utf8_stdio() -> None:
    """Make console output locale-independent.

    The launcher prints status glyphs (checkmarks, warnings). On hosts whose
    locale is latin-1 (LANG=en_US without .UTF-8), printing those crashes with
    UnicodeEncodeError. Reconfigure our streams to UTF-8 (replace on failure)
    rather than scrubbing every glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run federated experiments from YAML.")
    parser.add_argument("config", type=Path, help="experiment YAML file")
    parser.add_argument(
        "--example-dir", type=Path, default=None,
        help="example root (autodetected from config path if omitted)",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="shared metadata directory (default: <example>/metadata)",
    )
    args = parser.parse_args(argv)

    example_dir = args.example_dir or _autodetect_example_dir(args.config)
    runner = ExperimentRunner(example_dir, metadata_dir=args.metadata_dir)
    runner.run_experiment_batch(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
