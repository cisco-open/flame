# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for debug_run.sh's `--trace` substitution.

`make_debug_yaml`'s generator is Python embedded in a bash heredoc (no
importable module exists for it), so these tests extract and exec the actual
heredoc source out of debug_run.sh itself -- never a hand-copied
reimplementation that could silently drift from what actually runs.

Regression target: the generator used to patch the *aggregator's* trace
config (`h["trackTrainerAvail"]["trace"]` / `h["client_notify"]["trace"]` /
`h["availability_trace"]`) on `--trace` override, but never touched the
*trainer's* `hyperparameters.client_notify.trace` -- so every trainer, real
and sim, silently ran against the trainer_base.yaml default (`syn_0`,
always-available) regardless of the requested trace. See
UNAVAILABILITY_DESIGN.md Batch 3 T3.1.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_DEBUG_RUN_SH = (
    Path(__file__).parents[2]
    / "examples/async_cifar10/scripts/debug_run.sh"
)
_SCR_DIR = (
    Path(__file__).parents[2]
    / "examples/async_cifar10/expt_scripts_2026"
)


def _extract_make_debug_yaml_source() -> str:
    """Pull the Python heredoc body of make_debug_yaml() out of debug_run.sh."""
    text = _DEBUG_RUN_SH.read_text(encoding="utf-8")
    m = re.search(
        r"make_debug_yaml\(\) \{.*?<<'PY'\n(.*?)\nPY\n", text, re.DOTALL
    )
    assert m, "could not locate make_debug_yaml()'s PY heredoc in debug_run.sh"
    return m.group(1)


@pytest.fixture(scope="module")
def generator_source() -> str:
    return _extract_make_debug_yaml_source()


def _run_generator(
    source: str,
    tmp_path: Path,
    baselines: str,
    trace: str,
    mode: str = "both",
    runtime_s: int = 100,
) -> list[dict]:
    """Exec the extracted generator with a controlled argv; return experiments."""
    outpath = tmp_path / "out.yaml"
    argv = [
        "make_debug_yaml",
        str(_SCR_DIR),
        baselines,
        str(runtime_s),
        str(outpath),
        "0",  # smoke
        "",  # sim_wall_ceiling_s
        mode,
        trace,
        "",  # num_trainers override
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        exec(compile(source, "make_debug_yaml<heredoc>", "exec"), {})
    finally:
        sys.argv = old_argv
    assert outpath.is_file(), "generator produced no output (no baseline matched?)"
    with open(outpath) as f:
        return yaml.safe_load(f)["experiments"]


class TestTrainerTraceSubstitution:
    """The bug: trainer.config_overrides.hyperparameters.client_notify.trace
    must track --trace, independent of which aggregator-side branch (legacy
    trackTrainerAvail / HP-level client_notify / availability_trace) a given
    baseline takes."""

    @pytest.mark.parametrize(
        "baseline",
        ["felix", "oort", "oort_star", "refl", "feddance", "fedbuff"],
    )
    def test_trainer_client_notify_trace_matches_override(
        self, generator_source, tmp_path, baseline
    ):
        exps = _run_generator(generator_source, tmp_path, baseline, trace="syn_20")
        assert exps, f"no experiments generated for baseline={baseline}"
        for exp in exps:
            trace = (
                exp["trainer"]["config_overrides"]["hyperparameters"]
                ["client_notify"]["trace"]
            )
            assert trace == "syn_20", (
                f"{exp['name']}: trainer client_notify.trace={trace!r}, "
                f"expected 'syn_20' (--trace override was not propagated to "
                f"the trainer)"
            )

    def test_trainer_client_notify_trace_tracks_different_overrides(
        self, generator_source, tmp_path
    ):
        for trace in ("syn_0", "syn_20", "syn_50"):
            exps = _run_generator(
                generator_source, tmp_path, "feddance", trace=trace
            )
            for exp in exps:
                got = (
                    exp["trainer"]["config_overrides"]["hyperparameters"]
                    ["client_notify"]["trace"]
                )
                assert got == trace, f"{exp['name']}: expected {trace!r}, got {got!r}"

    def test_no_trace_override_leaves_trainer_config_untouched(
        self, generator_source, tmp_path
    ):
        """No --trace: don't inject a client_notify block that wasn't there."""
        exps = _run_generator(generator_source, tmp_path, "feddance", trace="")
        for exp in exps:
            t_hp = exp["trainer"].get("config_overrides", {}).get("hyperparameters", {})
            assert "client_notify" not in t_hp

    def test_aggregator_trace_still_substituted(self, generator_source, tmp_path):
        """Regression guard: fixing the trainer side must not break the
        pre-existing aggregator-side substitution paths."""
        exps = _run_generator(generator_source, tmp_path, "feddance", trace="syn_50")
        for exp in exps:
            h = exp["aggregator"]["config_overrides"]["hyperparameters"]
            assert h.get("availability_trace") == "syn_50"
            assert h.get("simUnavailability") is True

        exps = _run_generator(generator_source, tmp_path, "oort", trace="syn_50")
        for exp in exps:
            h = exp["aggregator"]["config_overrides"]["hyperparameters"]
            assert h["trackTrainerAvail"]["trace"] == "syn_50"


class TestMultiTraceSubstitution:
    """--trace accepts a space-separated list: one full experiment set per
    trace, queued in a single generated YAML/run."""

    def test_two_traces_double_the_experiment_count(self, generator_source, tmp_path):
        single = _run_generator(generator_source, tmp_path, "felix", trace="syn_20")
        both = _run_generator(generator_source, tmp_path, "felix", trace="syn_20 syn_50")
        assert len(both) == 2 * len(single)

    def test_each_trace_propagates_to_its_own_experiments(
        self, generator_source, tmp_path
    ):
        exps = _run_generator(generator_source, tmp_path, "felix", trace="syn_20 syn_50")
        by_trace = {
            trace: [
                e for e in exps
                if e["trainer"]["config_overrides"]["hyperparameters"]
                ["client_notify"]["trace"] == trace
            ]
            for trace in ("syn_20", "syn_50")
        }
        assert len(by_trace["syn_20"]) == len(by_trace["syn_50"]) == len(exps) // 2

    def test_experiment_names_disambiguate_by_trace(self, generator_source, tmp_path):
        exps = _run_generator(generator_source, tmp_path, "felix", trace="syn_20 syn_50")
        names = [e["name"] for e in exps]
        assert len(names) == len(set(names)), f"duplicate names: {names}"
