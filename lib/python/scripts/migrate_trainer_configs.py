# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Migrate per-trainer JSON configs into shared YAML metadata.

For each `<example>/trainer/configN/trainer_*.json`, capture:
  - taskid                              -> trainer_registry_<namespace>.yaml
  - hyperparameters.trainer_indices_list -> dataset_splits/<dataset>_<ns>_<configN>.yaml
  - any other per-trainer divergence (e.g. realm) -> plan.directories[].per_trainer_overrides
  - everything else (the dir's constant fields, in original key order)
                                        -> plan.directories[].template

Output together is lossless: see verify_migration.py.

Run:
  python -m scripts.migrate_trainer_configs \\
      --example lib/python/examples/cifar10 \\
      --metadata-out lib/python/examples/_metadata \\
      --plan-out lib/python/examples/cifar10/_migration_plan.yaml \\
      --namespace sync --dataset-name cifar10
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PER_TRAINER_TOP_KEYS_HANDLED_ELSEWHERE = {"taskid"}
PER_TRAINER_HP_KEYS_HANDLED_ELSEWHERE = {"trainer_indices_list"}


def _load_trainer_jsons(config_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(config_dir.glob("trainer_*.json")):
        tid = int(p.stem.split("_", 1)[1])
        out[tid] = json.loads(p.read_text())
    return out


def _trainer_key(trainer_id: int) -> str:
    return f"trainer_{trainer_id:03d}"


def _strip_template_fields(d: dict) -> dict:
    """Drop taskid + hp.trainer_indices_list, preserving key order otherwise."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in PER_TRAINER_TOP_KEYS_HANDLED_ELSEWHERE:
            continue
        if k == "hyperparameters" and isinstance(v, dict):
            out[k] = {
                hk: deepcopy(hv) for hk, hv in v.items()
                if hk not in PER_TRAINER_HP_KEYS_HANDLED_ELSEWHERE
            }
        else:
            out[k] = deepcopy(v)
    return out


def _equal_preserving_order(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a.keys()) != list(b.keys()):
            return False
        return all(_equal_preserving_order(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _equal_preserving_order(x, y) for x, y in zip(a, b)
        )
    return a == b


def _compute_overrides(stripped: dict, template: dict) -> dict:
    """Return a dict of keys in `stripped` whose values differ from `template`.

    The result is a *sparse* dict mirroring stripped's structure: only keys
    whose values differ are included, and nested dicts are likewise sparse.
    Lists are treated atomically (entire list included if any element differs).
    """
    out: dict[str, Any] = {}
    for k, v in stripped.items():
        if k not in template:
            out[k] = deepcopy(v)
            continue
        tv = template[k]
        if isinstance(v, dict) and isinstance(tv, dict):
            sub = _compute_overrides(v, tv)
            if sub:
                out[k] = sub
        elif not _equal_preserving_order(v, tv):
            out[k] = deepcopy(v)
    return out


def migrate_example(
    example_dir: Path,
    metadata_out: Path,
    plan_out: Path,
    namespace: str,
    dataset_name: str,
) -> dict:
    trainer_dir = example_dir / "trainer"
    config_dirs = sorted(
        d for d in trainer_dir.iterdir()
        if d.is_dir() and d.name.startswith("config")
    )
    if not config_dirs:
        raise ValueError(f"no config* dirs under {trainer_dir}")

    plan: dict[str, Any] = {
        "schema_version": 1,
        "example_dir": _rel_or_abs(example_dir),
        "namespace": namespace,
        "dataset_name": dataset_name,
        "trainer_registry_file": None,
        "directories": [],
    }

    # task_id must be stable per trainer across all config dirs.
    task_ids_per_trainer: dict[int, set[str]] = {}
    for cd in config_dirs:
        for tid, data in _load_trainer_jsons(cd).items():
            task_ids_per_trainer.setdefault(tid, set()).add(data["taskid"])
    for tid, tids in task_ids_per_trainer.items():
        if len(tids) > 1:
            raise ValueError(
                f"trainer {tid} has multiple task_ids across configs: {sorted(tids)}"
            )

    registry = {
        _trainer_key(tid): {
            "task_id": next(iter(task_ids_per_trainer[tid])),
            "trainer_id": tid,
        }
        for tid in sorted(task_ids_per_trainer)
    }

    metadata_out.mkdir(parents=True, exist_ok=True)
    splits_dir = metadata_out / "dataset_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    registry_file = metadata_out / f"trainer_registry_{namespace}.yaml"
    _write_yaml(registry_file, {"trainers": registry})
    plan["trainer_registry_file"] = _rel_or_abs(registry_file)

    for cd in config_dirs:
        cd_jsons = _load_trainer_jsons(cd)
        trainers = sorted(cd_jsons)
        template = _strip_template_fields(cd_jsons[trainers[0]])

        per_trainer_overrides: dict[str, dict] = {}
        indices_map: dict[str, list[int]] = {}

        for tid in trainers:
            tk = _trainer_key(tid)
            data = cd_jsons[tid]

            hp = data.get("hyperparameters", {})
            if "trainer_indices_list" not in hp:
                raise ValueError(
                    f"{cd.name}/trainer_{tid}.json missing hyperparameters.trainer_indices_list"
                )
            indices_map[tk] = hp["trainer_indices_list"]

            stripped = _strip_template_fields(data)
            overrides = _compute_overrides(stripped, template)
            if overrides:
                per_trainer_overrides[tk] = overrides

        split_file_name = f"{dataset_name}_{namespace}_{cd.name}.yaml"
        split_path = splits_dir / split_file_name
        _write_yaml(
            split_path,
            {
                "dataset": dataset_name,
                "namespace": namespace,
                "source": _rel_or_abs(cd),
                "trainer_data_splits": indices_map,
            },
        )

        plan["directories"].append({
            "name": cd.name,
            "source": _rel_or_abs(cd),
            "trainer_ids": trainers,
            "dataset_split_file": _rel_or_abs(split_path),
            "template": template,
            "per_trainer_overrides": per_trainer_overrides,
        })

    plan_out.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(plan_out, plan)
    return plan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rel_or_abs(path: Path) -> str:
    root = _repo_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--namespace", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="cifar10")
    args = parser.parse_args()

    example_dir = args.example.resolve()
    if not example_dir.is_dir():
        print(f"error: --example {example_dir} is not a directory", file=sys.stderr)
        return 2

    namespace = args.namespace or example_dir.name
    plan = migrate_example(
        example_dir=example_dir,
        metadata_out=args.metadata_out.resolve(),
        plan_out=args.plan_out.resolve(),
        namespace=namespace,
        dataset_name=args.dataset_name,
    )

    n_dirs = len(plan["directories"])
    n_trainers = sum(len(d["trainer_ids"]) for d in plan["directories"])
    n_with_overrides = sum(
        len(d["per_trainer_overrides"]) for d in plan["directories"]
    )
    print(f"migrated {n_dirs} directories ({n_trainers} trainer JSONs)")
    print(f"  trainer registry:        {plan['trainer_registry_file']}")
    print(f"  plan:                    {args.plan_out}")
    print(f"  trainers w/ overrides:   {n_with_overrides}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
