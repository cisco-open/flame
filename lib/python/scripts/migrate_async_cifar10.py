# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Migrate async_cifar10 trainer JSON configs into the shared hierarchical metadata.

Scope: the 4 canonical `_oort` dirs:
    config_dir<a>_num300_traceFail_6d_3state_oort  for a in {0.1, 1, 10, 100}

User decisions (2026-05-20):
  - alpha=0.1 is ground truth for all availability states.
  - Drop alpha=100's distinct 2_state data (legacy; alpha=0.1's wins).
  - Drop syn_40 (legacy; not in alpha=0.1).
  - Verification: strict byte-exact for alpha=0.1 (300 files); lightweight
    indices-only for alpha=1/10/100 (900 files, check trainer_indices_list
    matches shared dataset_splits).
  - After verification passes, delete all 4 alpha dirs.

Hierarchical _metadata schema (n=300):
    trainer_registry.yaml                         # task_id, training_delay_s, mobiperf_device_id, speed_class
    availability_traces/mobiperf_traces.yaml      # device_NNN.states_2st/3st_50/3st_75 (unchanged)
    availability_traces/synthetic_traces.yaml     # syn_<name>:
                                                  #   pattern (uniform, e.g. syn_0)
                                                  #   per_trainer.n300.<tk> (per-trainer, e.g. syn_20, syn_50)
    dataset_splits/cifar10_alpha<a>_n300.yaml     # trainer_data_splits (unchanged)

Run:
  python -m scripts.migrate_async_cifar10               # dry-run; verify only
  python -m scripts.migrate_async_cifar10 --write-plan  # verify + write plan
  python -m scripts.migrate_async_cifar10 --apply       # full apply + delete
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "lib/python/examples"
META = EXAMPLES / "_metadata"
TRAINER_REGISTRY = META / "trainer_registry.yaml"
MOBIPERF_TRACES = META / "availability_traces/mobiperf_traces.yaml"
SYNTHETIC_TRACES = META / "availability_traces/synthetic_traces.yaml"

SRC_BASE = EXAMPLES / "async_cifar10/trainer"
ALPHA_DIRS = {
    "0.1": SRC_BASE / "config_dir0.1_num300_traceFail_6d_3state_oort",
    "1":   SRC_BASE / "config_dir1_num300_traceFail_6d_3state_oort",
    "10":  SRC_BASE / "config_dir10_num300_traceFail_6d_3state_oort",
    "100": SRC_BASE / "config_dir100_num300_traceFail_6d_3state_oort",
}
ALPHA_SPLIT_FILE = {
    "0.1": META / "dataset_splits/cifar10_alpha0.1_n300.yaml",
    "1":   META / "dataset_splits/cifar10_alpha1.0_n300.yaml",
    "10":  META / "dataset_splits/cifar10_alpha10.0_n300.yaml",
    "100": META / "dataset_splits/cifar10_alpha100.0_n300.yaml",
}
PLAN_OUT = META / "migration_plan_async_cifar10.yaml"
POPULATION = "n300"
CANONICAL_ALPHA = "0.1"

# Reconstruction map for alpha=0.1 only (the canonical source).
# Maps `hp.<json_key>` -> trace lookup spec.
ALPHA01_TRACE_FIELDS: dict[str, tuple[str, str]] = {
    "avl_events_mobiperf_2st":    ("mobiperf", "states_2st"),
    "avl_events_mobiperf_3st_50": ("mobiperf", "states_3st_50"),
    "avl_events_mobiperf_3st_75": ("mobiperf", "states_3st_75"),
    "avl_events_syn_0":  ("synthetic", "syn_0"),
    "avl_events_syn_20": ("synthetic", "syn_20"),
    "avl_events_syn_50": ("synthetic", "syn_50"),
}

PER_TRAINER_HP_VIA_REGISTRY = {"training_delay_s"}
PER_TRAINER_HP_VIA_DATASET_SPLIT = {"trainer_indices_list"}


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _stringify_trace(trace_lol: list[list]) -> str:
    return str([tuple(x) for x in trace_lol])


def _trainer_key(tid: int) -> str:
    return f"trainer_{tid:03d}"


# ──────────────────────────────────────────────────────────────────────────────
# Trace lookups (operate on in-memory patched metadata)
# ──────────────────────────────────────────────────────────────────────────────


def _lookup_mobiperf(mobiperf: dict, tid: int, sub: str) -> list[list]:
    return mobiperf["traces"][f"device_{tid:03d}"][sub]


def _lookup_synthetic(synthetic: dict, name: str, tid: int) -> list[list]:
    entry = synthetic["traces"][name]
    per_t = entry.get("per_trainer", {}).get(POPULATION)
    if per_t and _trainer_key(tid) in per_t:
        return per_t[_trainer_key(tid)]
    return entry["pattern"]


# ──────────────────────────────────────────────────────────────────────────────
# Survey
# ──────────────────────────────────────────────────────────────────────────────


def survey() -> dict:
    """Read all 4 alpha dirs and derive: registry patches, synthetic per-trainer
    entries (from canonical alpha=0.1 only), and alpha=0.1 reconstruction info."""
    registry = _load_yaml(TRAINER_REGISTRY)["trainers"]

    # 1. Registry patches: training_delay_s typing. Source of truth = alpha=0.1.
    registry_patches: dict[str, dict] = {}
    for tid in range(1, 301):
        tk = _trainer_key(tid)
        j = json.loads((ALPHA_DIRS[CANONICAL_ALPHA] / f"trainer_{tid}.json").read_text())
        json_task = j["taskid"]
        json_delay = j["hyperparameters"]["training_delay_s"]
        if registry[tk]["task_id"] != json_task:
            raise ValueError(f"{tk}: taskid mismatch with registry")
        if registry[tk].get("training_delay_s") != json_delay:
            registry_patches[tk] = {"training_delay_s": json_delay}

    # 2. Synthetic per-trainer entries (from alpha=0.1 canonical).
    #    syn_0 is uniform across trainers -> keep as `pattern` only.
    #    syn_20, syn_50 are per-trainer -> extract for all 300.
    synthetic_per_trainer: dict[str, dict[str, list[list]]] = {
        "syn_20": {},
        "syn_50": {},
    }
    for tid in range(1, 301):
        tk = _trainer_key(tid)
        hp = json.loads(
            (ALPHA_DIRS[CANONICAL_ALPHA] / f"trainer_{tid}.json").read_text()
        )["hyperparameters"]
        for name in ("syn_20", "syn_50"):
            raw = ast.literal_eval(hp[f"avl_events_{name}"])
            synthetic_per_trainer[name][tk] = [list(t) for t in raw]

    # 3. alpha=0.1 reconstruction template + per-trainer overrides.
    trainers01 = sorted(
        int(p.stem.split("_", 1)[1])
        for p in ALPHA_DIRS[CANONICAL_ALPHA].glob("trainer_*.json")
        if "_test" not in p.stem
    )
    first = json.loads(
        (ALPHA_DIRS[CANONICAL_ALPHA] / f"trainer_{trainers01[0]}.json").read_text()
    )
    template01 = _build_template(first)

    per_trainer_overrides: dict[str, dict] = {}
    for tid in trainers01:
        j = json.loads((ALPHA_DIRS[CANONICAL_ALPHA] / f"trainer_{tid}.json").read_text())
        ov = _diff(_strip_for_diff(j), _strip_for_diff(template01))
        if ov:
            per_trainer_overrides[_trainer_key(tid)] = ov

    # 4. For alpha=1/10/100 we only collect trainer ids and a lightweight
    #    indices-verification plan against the existing dataset_splits.
    legacy_dirs: dict[str, dict] = {}
    for alpha in ("1", "10", "100"):
        d = ALPHA_DIRS[alpha]
        tids = sorted(
            int(p.stem.split("_", 1)[1])
            for p in d.glob("trainer_*.json")
            if "_test" not in p.stem
        )
        legacy_dirs[alpha] = {
            "source": str(d.relative_to(REPO_ROOT)),
            "trainers": tids,
            "dataset_split_file": str(ALPHA_SPLIT_FILE[alpha].relative_to(REPO_ROOT)),
        }

    return {
        "registry_patches": registry_patches,
        "synthetic_per_trainer": synthetic_per_trainer,
        "alpha01": {
            "source": str(ALPHA_DIRS[CANONICAL_ALPHA].relative_to(REPO_ROOT)),
            "trainers": trainers01,
            "template": template01,
            "trace_field_map": {k: list(v) for k, v in ALPHA01_TRACE_FIELDS.items()},
            "dataset_split_file": str(ALPHA_SPLIT_FILE[CANONICAL_ALPHA].relative_to(REPO_ROOT)),
            "per_trainer_overrides": per_trainer_overrides,
        },
        "legacy_dirs": legacy_dirs,
    }


def _build_template(sample_json: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in sample_json.items():
        if k == "taskid":
            out[k] = "__TASKID__"
        elif k == "hyperparameters" and isinstance(v, dict):
            hp_out: dict[str, Any] = {}
            for hk, hv in v.items():
                if (
                    hk in PER_TRAINER_HP_VIA_REGISTRY
                    or hk in PER_TRAINER_HP_VIA_DATASET_SPLIT
                    or hk.startswith("avl_events_")
                ):
                    hp_out[hk] = None
                else:
                    hp_out[hk] = deepcopy(hv)
            out[k] = hp_out
        else:
            out[k] = deepcopy(v)
    return out


def _strip_for_diff(d: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k == "taskid":
            continue
        if k == "hyperparameters" and isinstance(v, dict):
            out[k] = {
                hk: deepcopy(hv) for hk, hv in v.items()
                if hk not in PER_TRAINER_HP_VIA_REGISTRY
                and hk not in PER_TRAINER_HP_VIA_DATASET_SPLIT
                and not hk.startswith("avl_events_")
            }
        else:
            out[k] = deepcopy(v)
    return out


def _diff(a: dict, b: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in a.items():
        if k not in b:
            out[k] = deepcopy(v); continue
        bv = b[k]
        if isinstance(v, dict) and isinstance(bv, dict):
            sub = _diff(v, bv)
            if sub: out[k] = sub
        elif v != bv:
            out[k] = deepcopy(v)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction (alpha=0.1 only)
# ──────────────────────────────────────────────────────────────────────────────


def reconstruct_alpha01(
    trainer_id: int, info: dict, registry: dict,
    mobiperf: dict, synthetic: dict, splits: dict,
) -> dict:
    template = info["template"]
    tk = _trainer_key(trainer_id)
    task_id = registry[tk]["task_id"]
    training_delay_s = registry[tk]["training_delay_s"]
    indices = splits[tk]
    overrides = info["per_trainer_overrides"].get(tk, {})

    out: dict[str, Any] = {}
    for k, v in template.items():
        if k == "taskid":
            out[k] = task_id
        elif k == "hyperparameters":
            hp_out: dict[str, Any] = {}
            for hk, hv in v.items():
                if hk == "trainer_indices_list":
                    hp_out[hk] = indices
                elif hk == "training_delay_s":
                    hp_out[hk] = training_delay_s
                elif hk in ALPHA01_TRACE_FIELDS:
                    kind, sub = ALPHA01_TRACE_FIELDS[hk]
                    if kind == "mobiperf":
                        trace = _lookup_mobiperf(mobiperf, trainer_id, sub)
                    else:
                        trace = _lookup_synthetic(synthetic, sub, trainer_id)
                    hp_out[hk] = _stringify_trace(trace)
                else:
                    hp_out[hk] = deepcopy(hv)
            if "hyperparameters" in overrides:
                for ok, ov in overrides["hyperparameters"].items():
                    hp_out[ok] = ov
            out[k] = hp_out
        else:
            out[k] = deepcopy(v)

    for ok, ov in overrides.items():
        if ok != "hyperparameters":
            out[ok] = ov

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────


def verify_strict_alpha01(
    info: dict, registry: dict, mobiperf: dict, synthetic: dict,
) -> tuple[int, int, list[dict]]:
    """Byte-exact reconstruction match against every alpha=0.1 source JSON."""
    splits = _load_yaml(ALPHA_SPLIT_FILE[CANONICAL_ALPHA])["trainer_data_splits"]
    src_dir = ALPHA_DIRS[CANONICAL_ALPHA]

    passed = failed = 0
    failures: list[dict] = []
    for tid in info["trainers"]:
        recon = reconstruct_alpha01(tid, info, registry, mobiperf, synthetic, splits)
        actual = json.dumps(recon, indent=4).encode()
        expected = (src_dir / f"trainer_{tid}.json").read_bytes()
        if actual == expected:
            passed += 1
        else:
            failed += 1
            failures.append({
                "alpha": CANONICAL_ALPHA, "trainer_id": tid,
                "src": str(src_dir / f"trainer_{tid}.json"),
                "diff": _short_diff(expected, actual),
            })
    return passed, failed, failures


def verify_indices_legacy(legacy_dirs: dict) -> tuple[int, int, list[dict]]:
    """For alpha=1/10/100: each trainer's JSON indices must match shared dataset_split."""
    passed = failed = 0
    failures: list[dict] = []
    for alpha, info in legacy_dirs.items():
        splits = _load_yaml(REPO_ROOT / info["dataset_split_file"])["trainer_data_splits"]
        src_dir = REPO_ROOT / info["source"]
        for tid in info["trainers"]:
            tk = _trainer_key(tid)
            j = json.loads((src_dir / f"trainer_{tid}.json").read_text())
            json_idx = j["hyperparameters"]["trainer_indices_list"]
            yaml_idx = splits.get(tk)
            if json_idx == yaml_idx:
                passed += 1
            else:
                failed += 1
                failures.append({
                    "alpha": alpha, "trainer_id": tid,
                    "src": str(src_dir / f"trainer_{tid}.json"),
                    "issue": (
                        f"indices mismatch: json_len={len(json_idx)} "
                        f"yaml_len={len(yaml_idx) if yaml_idx else 'MISSING'}"
                    ),
                })
    return passed, failed, failures


def _short_diff(expected: bytes, actual: bytes, max_lines: int = 40) -> str:
    import difflib
    exp = expected.decode(errors="replace").splitlines(keepends=True)
    act = actual.decode(errors="replace").splitlines(keepends=True)
    diff = list(difflib.unified_diff(exp, act, "expected", "actual", n=2))
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more lines)\n"]
    return "".join(diff)


# ──────────────────────────────────────────────────────────────────────────────
# Mutations
# ──────────────────────────────────────────────────────────────────────────────


def apply_registry_patches(patches: dict) -> dict:
    reg = _load_yaml(TRAINER_REGISTRY)
    for tk, fields in patches.items():
        reg["trainers"][tk].update(fields)
    return reg


def apply_synthetic_per_trainer(per_trainer: dict[str, dict[str, list]]) -> dict:
    """Inject per-trainer realizations into synthetic_traces.yaml in-memory."""
    syn = _load_yaml(SYNTHETIC_TRACES)
    for name, by_tk in per_trainer.items():
        entry = syn["traces"].setdefault(
            name, {"description": f"Imported synthetic trace {name}"}
        )
        entry.setdefault("per_trainer", {})[POPULATION] = by_tk
    return syn


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--write-plan", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    print("loading shared metadata...")
    mobiperf_full = _load_yaml(MOBIPERF_TRACES)

    print(f"surveying {len(ALPHA_DIRS)} alpha dirs...")
    survey_data = survey()

    print(f"  registry patches needed: {len(survey_data['registry_patches'])} trainers")
    syn_summary = {n: len(d) for n, d in survey_data["synthetic_per_trainer"].items()}
    print(f"  synthetic per-trainer entries to add: {syn_summary}")
    print(f"  alpha=0.1: {len(survey_data['alpha01']['trainers'])} trainers, "
          f"{len(survey_data['alpha01']['per_trainer_overrides'])} per-trainer overrides")
    for alpha, info in survey_data["legacy_dirs"].items():
        print(f"  alpha={alpha} (legacy, indices-only verify): {len(info['trainers'])} trainers")

    patched_registry = apply_registry_patches(survey_data["registry_patches"])
    patched_synthetic = apply_synthetic_per_trainer(survey_data["synthetic_per_trainer"])

    print("\nverify strict (alpha=0.1)...")
    p1, f1, failures1 = verify_strict_alpha01(
        survey_data["alpha01"], patched_registry["trainers"],
        mobiperf_full, patched_synthetic,
    )
    print(f"  {p1}/{p1 + f1} pass (byte-exact)")
    if f1:
        for f in failures1[:3]:
            print(f"\n  --- alpha={f['alpha']} trainer_{f['trainer_id']} ---")
            print(f["diff"])
        if f1 > 3:
            print(f"\n  ...({f1 - 3} more failures)")
        return 1

    print("\nverify indices-only (alpha=1/10/100)...")
    p2, f2, failures2 = verify_indices_legacy(survey_data["legacy_dirs"])
    print(f"  {p2}/{p2 + f2} pass (indices match shared dataset_split)")
    if f2:
        for f in failures2[:5]:
            print(f"  FAIL alpha={f['alpha']} trainer_{f['trainer_id']}: {f['issue']}")
        return 1

    print(f"\nALL CHECKS PASS: {p1} strict + {p2} indices = {p1 + p2} total")

    if args.apply or args.write_plan:
        plan_dict = {
            "schema_version": 1,
            "scope": "async_cifar10 _oort 6d_3state",
            "canonical_alpha": CANONICAL_ALPHA,
            "population": POPULATION,
            "source_dirs": {a: str(d.relative_to(REPO_ROOT)) for a, d in ALPHA_DIRS.items()},
            "trainer_registry": str(TRAINER_REGISTRY.relative_to(REPO_ROOT)),
            "mobiperf_traces": str(MOBIPERF_TRACES.relative_to(REPO_ROOT)),
            "synthetic_traces": str(SYNTHETIC_TRACES.relative_to(REPO_ROOT)),
            "registry_patches_applied": list(survey_data["registry_patches"]),
            "synthetic_per_trainer_added": {
                n: list(by_tk) for n, by_tk in survey_data["synthetic_per_trainer"].items()
            },
            "alpha01_strict": {
                "source": survey_data["alpha01"]["source"],
                "trainers": survey_data["alpha01"]["trainers"],
                "dataset_split_file": survey_data["alpha01"]["dataset_split_file"],
                "template": survey_data["alpha01"]["template"],
                "trace_field_map": survey_data["alpha01"]["trace_field_map"],
                "per_trainer_overrides": survey_data["alpha01"]["per_trainer_overrides"],
            },
            "legacy_indices_only": survey_data["legacy_dirs"],
            "dropped_legacy_data": [
                "alpha=100 distinct avl_events_2_state per-trainer realizations",
                "alpha=1/10/100 avl_events_syn_40 per-trainer realizations",
                "alpha=1/10/100 older key names (2_state, 3_state_*) — superseded by mobiperf_*",
            ],
        }
        _dump_yaml(PLAN_OUT, plan_dict)
        print(f"\nplan written: {PLAN_OUT}")

    if args.apply:
        _dump_yaml(TRAINER_REGISTRY, patched_registry)
        _dump_yaml(SYNTHETIC_TRACES, patched_synthetic)
        print(f"patched: {TRAINER_REGISTRY}")
        print(f"patched: {SYNTHETIC_TRACES}")
        for alpha, d in ALPHA_DIRS.items():
            shutil.rmtree(d)
            print(f"deleted: {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
