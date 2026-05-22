# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Baseline catalog loader + deep-merge with provenance tracking."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_baselines(metadata_dir: Path) -> dict:
    """Read `<metadata_dir>/baselines.yaml`. Returns the `baselines` mapping."""
    path = metadata_dir / "baselines.yaml"
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("baselines", {})


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge. Overlay wins for non-dict values; dicts merge."""
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_with_provenance(
    layers: list[tuple[str, dict]],
) -> tuple[dict, dict]:
    """Merge layers in order. Returns (merged_dict, provenance_map).

    provenance_map maps each leaf dotted-key path -> the layer name that
    contributed the final value.
    """
    merged: dict = {}
    provenance: dict[str, str] = {}
    for layer_name, layer in layers:
        if not layer:
            continue
        _merge_into(merged, layer, layer_name, provenance, prefix="")
    return merged, provenance


def _merge_into(
    target: dict, source: dict, layer_name: str, provenance: dict, prefix: str
) -> None:
    for k, v in source.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if not isinstance(target.get(k), dict):
                target[k] = {}
            _merge_into(target[k], v, layer_name, provenance, path)
        else:
            target[k] = deepcopy(v)
            provenance[path] = layer_name


def format_provenance(
    name: str, provenance: dict, max_paths: int = 40
) -> str:
    """One-shot summary of which layer set which path. For run-time logging."""
    if not provenance:
        return f"  [{name}] (no merged fields)"
    by_layer: dict[str, list[str]] = {}
    for path, layer in sorted(provenance.items()):
        by_layer.setdefault(layer, []).append(path)
    lines = [f"  [{name}] field provenance:"]
    for layer in sorted(by_layer):
        paths = by_layer[layer]
        shown = paths if len(paths) <= max_paths else paths[:max_paths] + [
            f"... ({len(paths) - max_paths} more)"
        ]
        lines.append(f"    from {layer}:")
        for p in shown:
            lines.append(f"      - {p}")
    return "\n".join(lines)
