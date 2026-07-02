# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for analyze_run.py's per-example telemetry manifest
(P5.4): _find_manifest_path(), load_manifest(), configure_from_manifest().

A manifest is optional -- an example with no telemetry_manifest.yaml (e.g.
async_cifar10) must fall back to the pre-manifest defaults (hardcoded
MODEL_PARAM_COUNT, fwdllm-shaped _DEFAULT_PROGRESS_HIERARCHY) unchanged. A
malformed manifest must not crash analysis. --model-params on the CLI must
never be clobbered by a manifest's model_param_count.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                 "scripts", "analysis"),
)

import analyze_run as ar  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_module_globals():
    """configure_from_manifest() mutates module-level globals that other
    tests (e.g. test_progress_key.py) rely on being at their defaults --
    snapshot and restore around every test in this file."""
    saved = (ar.MODEL_PARAM_COUNT, ar.MODEL_MB, list(ar._PROGRESS_HIERARCHY),
              ar._MODEL_PARAMS_CLI_OVERRIDDEN)
    yield
    (ar.MODEL_PARAM_COUNT, ar.MODEL_MB, ar._PROGRESS_HIERARCHY,
     ar._MODEL_PARAMS_CLI_OVERRIDDEN) = saved


def _make_example(tmp_path, name, manifest_yaml=None):
    """Build a .../examples/<name>/experiments/<run>/telemetry dir, optionally
    with a telemetry_manifest.yaml next to `experiments/` (matching fwdllm's
    real on-disk layout)."""
    example_dir = tmp_path / "examples" / name
    telemetry_dir = example_dir / "experiments" / "run_x" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    if manifest_yaml is not None:
        (example_dir / "telemetry_manifest.yaml").write_text(manifest_yaml)
    return str(telemetry_dir)


FWDLLM_MANIFEST = """
model_param_count: 450340
progress_hierarchy:
  - field: data_id
    bound: 150
  - field: iteration_per_data_id
    bound: 15
"""


def test_find_manifest_path_locates_manifest_next_to_examples_root(tmp_path):
    telemetry_dir = _make_example(tmp_path, "fwdllm", manifest_yaml=FWDLLM_MANIFEST)
    found = ar._find_manifest_path(telemetry_dir)
    assert found == os.path.join(tmp_path, "examples", "fwdllm", "telemetry_manifest.yaml")


def test_find_manifest_path_returns_none_when_absent(tmp_path):
    """An example with no manifest (e.g. async_cifar10) -- absence is not an
    error, just "use defaults"."""
    telemetry_dir = _make_example(tmp_path, "async_cifar10", manifest_yaml=None)
    assert ar._find_manifest_path(telemetry_dir) is None


def test_find_manifest_path_returns_none_outside_any_examples_root(tmp_path):
    stray_dir = tmp_path / "not_an_example" / "telemetry"
    stray_dir.mkdir(parents=True)
    assert ar._find_manifest_path(str(stray_dir)) is None


def test_load_manifest_parses_declared_fields(tmp_path):
    telemetry_dir = _make_example(tmp_path, "fwdllm", manifest_yaml=FWDLLM_MANIFEST)
    manifest = ar.load_manifest(telemetry_dir)
    assert manifest["model_param_count"] == 450340
    assert manifest["progress_hierarchy"] == [
        {"field": "data_id", "bound": 150},
        {"field": "iteration_per_data_id", "bound": 15},
    ]


def test_load_manifest_returns_none_when_absent(tmp_path):
    telemetry_dir = _make_example(tmp_path, "async_cifar10", manifest_yaml=None)
    assert ar.load_manifest(telemetry_dir) is None


def test_load_manifest_malformed_yaml_does_not_crash(tmp_path, capsys):
    telemetry_dir = _make_example(tmp_path, "broken", manifest_yaml="{not: valid: yaml: [")
    assert ar.load_manifest(telemetry_dir) is None
    assert "warning" in capsys.readouterr().out


def test_configure_from_manifest_sets_globals_from_fwdllm_manifest(tmp_path):
    telemetry_dir = _make_example(tmp_path, "fwdllm", manifest_yaml=FWDLLM_MANIFEST)
    ar._MODEL_PARAMS_CLI_OVERRIDDEN = False
    manifest = ar.configure_from_manifest(telemetry_dir)
    assert manifest is not None
    assert ar.MODEL_PARAM_COUNT == 450340
    assert ar.MODEL_MB == pytest.approx(450340 * 4 / 1e6)
    assert ar._PROGRESS_HIERARCHY == [
        {"field": "data_id", "bound": 150},
        {"field": "iteration_per_data_id", "bound": 15},
    ]


def test_configure_from_manifest_falls_back_to_defaults_when_absent(tmp_path):
    telemetry_dir = _make_example(tmp_path, "async_cifar10", manifest_yaml=None)
    ar.MODEL_PARAM_COUNT = 999  # simulate a leftover value from a prior run
    ar._MODEL_PARAMS_CLI_OVERRIDDEN = False
    manifest = ar.configure_from_manifest(telemetry_dir)
    assert manifest is None
    assert ar.MODEL_PARAM_COUNT == 999  # untouched -- no manifest to apply
    assert ar._PROGRESS_HIERARCHY == ar._DEFAULT_PROGRESS_HIERARCHY


def test_configure_from_manifest_respects_cli_override_guard(tmp_path):
    """--model-params on the CLI must win over a manifest's model_param_count
    (main() sets _MODEL_PARAMS_CLI_OVERRIDDEN=True before analyze() runs)."""
    telemetry_dir = _make_example(tmp_path, "fwdllm", manifest_yaml=FWDLLM_MANIFEST)
    ar.MODEL_PARAM_COUNT = 123456
    ar.MODEL_MB = 123456 * 4 / 1e6
    ar._MODEL_PARAMS_CLI_OVERRIDDEN = True
    manifest = ar.configure_from_manifest(telemetry_dir)
    assert manifest is not None  # still loaded, for progress_hierarchy/event_categories
    assert ar.MODEL_PARAM_COUNT == 123456  # not clobbered
    assert ar._PROGRESS_HIERARCHY == [
        {"field": "data_id", "bound": 150},
        {"field": "iteration_per_data_id", "bound": 15},
    ]  # progress_hierarchy still applies -- the CLI flag only guards model size
