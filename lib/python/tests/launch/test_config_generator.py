# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfigGenerator with the real shared metadata bundle."""

from pathlib import Path

import pytest
import yaml


SHARED_METADATA = Path(__file__).resolve().parents[2] / "examples" / "_metadata"
TRAINER_BASE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "feddance_cifar10"
    / "configs"
    / "trainer_base.yaml"
)


pytestmark = pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not TRAINER_BASE.is_file(),
    reason="shared metadata or feddance_cifar10 trainer_base not present",
)


@pytest.fixture
def loader():
    from flame.launch.spawner import MetadataLoader

    return MetadataLoader(SHARED_METADATA)


@pytest.fixture
def gen(loader):
    from flame.launch.spawner import ConfigGenerator

    return ConfigGenerator(loader, TRAINER_BASE)


class TestMetadataLoader:
    def test_registry_loaded(self, loader):
        assert len(loader.trainer_registry) > 0

    def test_dataset_splits_loaded(self, loader):
        assert len(loader.dataset_splits) > 0


class TestConfigGenerator:
    def test_generate_basic(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=1, alpha=0.1, availability_mode="syn_0"
        )
        assert "taskid" in cfg
        assert isinstance(cfg["taskid"], str) and len(cfg["taskid"]) > 0
        assert cfg["hyperparameters"]["trainer_indices_list"]
        assert cfg["selector"]["sort"] == "feddance"

    def test_overrides_applied(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=2,
            alpha=0.1,
            availability_mode="syn_0",
            **{"job.id": "experiment-xyz", "hyperparameters.batchSize": 64},
        )
        assert cfg["job"]["id"] == "experiment-xyz"
        assert cfg["hyperparameters"]["batchSize"] == 64


FWDLLM_TRAINER_BASE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "fwdllm"
    / "configs"
    / "trainer_base.yaml"
)
FWDLLM_AGGREGATOR_MAIN = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "fwdllm"
    / "aggregator"
    / "main_fedfwd_agg.py"
)


@pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not FWDLLM_TRAINER_BASE.is_file(),
    reason="shared metadata or fwdllm trainer_base not present",
)
class TestFwdllmEndToEndConfigGeneration:
    """For each of the four fwdllm baselines, generate the aggregator + a
    trainer config end-to-end (extends the single retired
    fedfwd_async_random_dynkc coverage to all four)."""

    @pytest.fixture
    def baselines(self):
        from flame.launch.baselines import load_baselines

        return load_baselines(SHARED_METADATA)

    @pytest.fixture
    def gen(self, loader):
        from flame.launch.spawner import ConfigGenerator

        return ConfigGenerator(loader, FWDLLM_TRAINER_BASE)

    @pytest.mark.parametrize(
        "baseline_name", ["fwdllm", "fwdllm_plus", "fluxtune", "fluxtune_dynkc"]
    )
    def test_trainer_config_generates_without_keyerror(
        self, gen, baselines, baseline_name
    ):
        gen.set_baseline_overrides(baselines[baseline_name].get("trainer", {}))
        cfg = gen.generate_trainer_config(
            trainer_id=1,
            alpha=0.1,
            availability_mode="syn_0",
            dataset_name="agnews",
            num_trainers=10,
            skip_index_splits=True,
            **{"hyperparameters.client_idx": 0},
        )
        assert cfg["hyperparameters"]["client_idx"] == 0
        assert "trainer_indices_list" not in cfg["hyperparameters"]

    @pytest.mark.parametrize(
        "baseline_name,expected",
        [
            ("fwdllm", {"client_notify.enabled": "False"}),
            ("fwdllm_plus", {"client_notify.enabled": "False"}),
            (
                "fluxtune",
                {
                    "select_perturbation_using_jvp": True,
                    "client_notify.enabled": "True",
                },
            ),
            ("fluxtune_dynkc", {"client_notify.enabled": "False"}),
        ],
    )
    def test_trainer_config_matches_matrix(
        self, gen, baselines, baseline_name, expected
    ):
        gen.set_baseline_overrides(baselines[baseline_name].get("trainer", {}))
        cfg = gen.generate_trainer_config(
            trainer_id=1,
            alpha=0.1,
            availability_mode="syn_0",
            dataset_name="agnews",
            num_trainers=10,
            skip_index_splits=True,
        )
        hp = cfg["hyperparameters"]
        for dotted_key, want in expected.items():
            cur = hp
            for part in dotted_key.split(".")[:-1]:
                cur = cur[part]
            assert cur[dotted_key.split(".")[-1]] == want

    @pytest.mark.parametrize(
        "baseline_name,expected",
        [
            (
                "fwdllm",
                {
                    "selector.sort": "random",
                    "optimizer.sort": "fedavg",
                    "hyperparameters.reselect_each_iteration": False,
                },
            ),
            (
                "fwdllm_plus",
                {
                    "selector.sort": "random",
                    "optimizer.sort": "fedavg",
                    "hyperparameters.reselect_each_iteration": True,
                },
            ),
            (
                "fluxtune",
                {"selector.sort": "async_oort", "optimizer.sort": "fedbuff"},
            ),
            (
                "fluxtune_dynkc",
                {"selector.sort": "async_random", "optimizer.sort": "fedbuff"},
            ),
        ],
    )
    def test_aggregator_config_matches_matrix_and_validates(
        self, baselines, baseline_name, expected
    ):
        import json

        from flame.launch.baselines import deep_merge
        from flame.launch.runner import ExperimentRunner

        tmpl = json.load(open(SHARED_METADATA / "aggregator_base.json"))
        merged = deep_merge(tmpl, baselines[baseline_name]["aggregator"])
        for dotted_key, want in expected.items():
            cur = merged
            for part in dotted_key.split(".")[:-1]:
                cur = cur[part]
            assert cur[dotted_key.split(".")[-1]] == want

        # Real entrypoint, not a synthetic fixture -- locks in the
        # main_fedfwd_agg.py marker import that makes the stack-detection
        # regex find the fwdllm stack on this file.
        runner = ExperimentRunner(FWDLLM_AGGREGATOR_MAIN.parents[1])
        runner._validate_stack(FWDLLM_AGGREGATOR_MAIN, merged)


FWDLLM_EXPT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "examples" / "fwdllm" / "expt_scripts"
)


@pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not FWDLLM_TRAINER_BASE.is_file(),
    reason="shared metadata or fwdllm trainer_base not present",
)
class TestFwdllmSmokeYamlsResolve:
    """The n10_smoke.yaml for each of the three owner-spec baselines (fwdllm,
    fwdllm_plus, fluxtune) must load, resolve its baseline, generate a
    trainer config, and pass _validate_stack against the real entrypoint --
    this is the dry-run Smoke Test D does live, kept as a permanent
    regression test for all three so real/sim parity runs stay covered."""

    @pytest.mark.parametrize(
        "yaml_name,expected_baseline",
        [
            ("fwdllm_n10_smoke.yaml", "fwdllm"),
            ("fwdllm_plus_n10_smoke.yaml", "fwdllm_plus"),
            ("fluxtune_n10_smoke.yaml", "fluxtune"),
        ],
    )
    def test_smoke_yaml_end_to_end(self, yaml_name, expected_baseline):
        import json

        from flame.launch.baselines import deep_merge, load_baselines
        from flame.launch.experiment_config import load_experiment_config
        from flame.launch.runner import ExperimentRunner
        from flame.launch.spawner import ConfigGenerator, MetadataLoader

        path = FWDLLM_EXPT_SCRIPTS / yaml_name
        if not path.is_file():
            pytest.skip(f"{yaml_name} not present in this checkout")

        exp = load_experiment_config(path).experiments[0]
        assert exp.baseline == expected_baseline
        assert exp.trainer.num_trainers == 10

        baselines = load_baselines(SHARED_METADATA)
        baseline = baselines[exp.baseline]

        meta = MetadataLoader(SHARED_METADATA)
        cg = ConfigGenerator(meta, FWDLLM_TRAINER_BASE)
        cg.set_baseline_overrides(baseline.get("trainer", {}))
        cfg = cg.generate_trainer_config(
            1,
            alpha=0.1,
            availability_mode=exp.trainer.availability.mode,
            dataset_name=exp.trainer.dataset.name,
            num_trainers=10,
            skip_index_splits=exp.trainer.dataset.path_style,
            **{"hyperparameters.client_idx": 0},
        )
        assert cfg["hyperparameters"]["client_idx"] == 0
        assert "trainer_indices_list" not in cfg["hyperparameters"]

        tmpl = json.load(open(SHARED_METADATA / "aggregator_base.json"))
        merged_agg = deep_merge(tmpl, baseline["aggregator"])
        merged_agg = deep_merge(merged_agg, exp.aggregator.config_overrides or {})

        runner = ExperimentRunner(FWDLLM_AGGREGATOR_MAIN.parents[1])
        runner._validate_stack(FWDLLM_AGGREGATOR_MAIN, merged_agg)


class TestTrainerSpawnerForwardsDatasetIdentity:
    """Regression guard for the agg_goal-class bug found in TrainerSpawner:
    spawn_trainer()/spawn_all() used to silently drop dataset_name/
    num_trainers on the floor instead of forwarding them to
    ConfigGenerator.generate_trainer_config(), which then fell back to its
    own defaults ("cifar10", 300) regardless of the real experiment -- e.g.
    a 48-trainer experiment would load the cifar10_alpha<a>_n300.yaml split
    file instead of cifar10_alpha<a>_n48.yaml whenever both exist, silently
    handing every trainer the wrong (but structurally valid, non-crashing)
    partition. See MIGRATING_TO_LAUNCHER.md's "real config vs. dead fields"
    section."""

    class _RecordingConfigGenerator:
        def __init__(self):
            self.calls = []

        def generate_trainer_config(self, trainer_id, alpha, availability_mode,
                                     dataset_name="cifar10", num_trainers=300,
                                     skip_index_splits=False, **overrides):
            self.calls.append(
                {"trainer_id": trainer_id, "dataset_name": dataset_name,
                 "num_trainers": num_trainers}
            )
            return {"taskid": "t", "hyperparameters": {}, "job": {}}

    def _spawner(self, recording_gen):
        from flame.launch.spawner import TrainerSpawner

        return TrainerSpawner(recording_gen, num_gpus=1, cpu_pinning=False)

    def test_spawn_trainer_forwards_dataset_name_and_num_trainers(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        monkeypatch.setattr(
            subprocess, "Popen",
            lambda *a, **k: type("P", (), {"pid": 1})(),
        )
        gen = self._RecordingConfigGenerator()
        spawner = self._spawner(gen)
        spawner.spawn_trainer(
            trainer_id=5, alpha=0.1, availability_mode="syn_0",
            trainer_main_path=tmp_path / "main.py",
            dataset_name="agnews", num_trainers=48,
        )
        assert gen.calls == [
            {"trainer_id": 5, "dataset_name": "agnews", "num_trainers": 48}
        ]

    def test_spawn_all_forwards_dataset_name_and_num_trainers(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        monkeypatch.setattr(
            subprocess, "Popen",
            lambda *a, **k: type("P", (), {"pid": 1})(),
        )
        gen = self._RecordingConfigGenerator()
        spawner = self._spawner(gen)
        spawner.spawn_all(
            trainer_ids=[1, 2, 3], alpha=0.1, availability_mode="syn_0",
            trainer_main_path=tmp_path / "main.py",
            dataset_name="agnews", num_trainers=48,
        )
        assert all(
            c["dataset_name"] == "agnews" and c["num_trainers"] == 48
            for c in gen.calls
        )
        assert len(gen.calls) == 3

    def test_spawn_all_defaults_match_generate_trainer_config_defaults(
        self, tmp_path, monkeypatch
    ):
        """Callers that don't pass dataset_name/num_trainers (e.g. direct,
        non-launcher use) must fall back to the same defaults
        generate_trainer_config itself documents -- not silently diverge."""
        import subprocess

        monkeypatch.setattr(
            subprocess, "Popen",
            lambda *a, **k: type("P", (), {"pid": 1})(),
        )
        gen = self._RecordingConfigGenerator()
        spawner = self._spawner(gen)
        spawner.spawn_trainer(
            trainer_id=1, alpha=0.1, availability_mode="syn_0",
            trainer_main_path=tmp_path / "main.py",
        )
        assert gen.calls == [
            {"trainer_id": 1, "dataset_name": "cifar10", "num_trainers": 300}
        ]
