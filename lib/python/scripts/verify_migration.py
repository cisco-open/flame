# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Strict byte-equivalence verifier for trainer-config migration.

Reads `_migration_plan.yaml` (produced by migrate_trainer_configs.py), the
trainer_registry file, and each dataset_splits file. For every source JSON
referenced by the plan, reconstructs the JSON dict and compares the
`json.dumps(..., indent=4)` serialization byte-for-byte against the on-disk
original. Exits non-zero on any mismatch.

Run:
  python -m scripts.verify_migration \\
      --plan lib/python/examples/cifar10/_migration_plan.yaml \\
      [--report path/to/report.json]
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


# Must mirror migrate_trainer_configs.py.
PER_TRAINER_TOP_KEYS_HANDLED_ELSEWHERE = {"taskid"}
PER_TRAINER_HP_KEYS_HANDLED_ELSEWHERE = {"trainer_indices_list"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_repo_root() / path)


def _trainer_key(trainer_id: int) -> str:
    return f"trainer_{trainer_id:03d}"


def _deep_apply_overrides(base: dict, overrides: dict) -> dict:
    """Apply sparse overrides into base in-place; nested dicts merged recursively."""
    for k, v in overrides.items():
        if (
            isinstance(v, dict)
            and isinstance(base.get(k), dict)
        ):
            _deep_apply_overrides(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _reconstruct(
    template: dict,
    task_id: str,
    indices: list[int],
    overrides: dict | None,
) -> dict:
    """Build a dict whose key order matches the source JSON exactly.

    Source format places `taskid` first, then the rest of the keys in
    template's order. `trainer_indices_list` is the last entry of
    `hyperparameters`.
    """
    body = deepcopy(template)
    if overrides:
        _deep_apply_overrides(body, overrides)

    # taskid must come first
    out: dict[str, Any] = {"taskid": task_id}
    for k, v in body.items():
        if k == "hyperparameters" and isinstance(v, dict):
            hp_ordered: dict[str, Any] = {kk: deepcopy(vv) for kk, vv in v.items()}
            hp_ordered["trainer_indices_list"] = indices
            out[k] = hp_ordered
        else:
            out[k] = v
    return out


def _serialize(obj: Any) -> bytes:
    """Use json.dumps(indent=4) — matches source format byte-for-byte."""
    return json.dumps(obj, indent=4).encode()


def _short_diff(expected: bytes, actual: bytes, max_lines: int = 30) -> str:
    exp_lines = expected.decode(errors="replace").splitlines(keepends=True)
    act_lines = actual.decode(errors="replace").splitlines(keepends=True)
    diff = list(difflib.unified_diff(exp_lines, act_lines, "expected", "actual", n=2))
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more lines)\n"]
    return "".join(diff)


def verify(plan_path: Path) -> tuple[int, int, list[dict]]:
    """Return (passed, failed, failures)."""
    plan = yaml.safe_load(plan_path.read_text())
    registry = yaml.safe_load(
        _resolve(plan["trainer_registry_file"]).read_text()
    )["trainers"]

    passed = 0
    failed = 0
    failures: list[dict] = []

    for d in plan["directories"]:
        split = yaml.safe_load(
            _resolve(d["dataset_split_file"]).read_text()
        )["trainer_data_splits"]
        template = d["template"]
        overrides_all = d.get("per_trainer_overrides") or {}
        src_dir = _resolve(d["source"])

        for trainer_id in d["trainer_ids"]:
            tk = _trainer_key(trainer_id)
            task_id = registry[tk]["task_id"]
            indices = split[tk]
            overrides = overrides_all.get(tk)
            reconstructed = _reconstruct(template, task_id, indices, overrides)

            src_path = src_dir / f"trainer_{trainer_id}.json"
            expected = src_path.read_bytes()
            actual = _serialize(reconstructed)

            if expected == actual:
                passed += 1
            else:
                failed += 1
                failures.append({
                    "source": str(src_path),
                    "dir": d["name"],
                    "trainer_id": trainer_id,
                    "expected_size": len(expected),
                    "actual_size": len(actual),
                    "diff": _short_diff(expected, actual),
                })

    return passed, failed, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    passed, failed, failures = verify(args.plan.resolve())
    total = passed + failed

    print(f"verified: {passed}/{total} pass")
    if failed:
        print(f"  FAILED: {failed}")
        for f in failures[:5]:
            print(f"\n  --- {f['source']} ---")
            print(f"  sizes: expected={f['expected_size']} actual={f['actual_size']}")
            print(f["diff"])
        if failed > 5:
            print(f"\n  ...({failed - 5} more failures; see --report)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(
                {
                    "plan": str(args.plan),
                    "passed": passed,
                    "failed": failed,
                    "failures": failures,
                },
                f,
                indent=2,
            )
        print(f"\nreport: {args.report}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
