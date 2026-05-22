# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for migrate_trainer_configs + verify_migration."""

import json
from pathlib import Path

import pytest


def _make_trainer_json(taskid: str, indices: list[int], realm: str = "default/us") -> dict:
    """Mirrors the cifar10 source schema (and its key order)."""
    return {
        "taskid": taskid,
        "backend": "mqtt",
        "brokers": [{"host": "localhost", "sort": "mqtt"}],
        "groupAssociation": {"param-channel": "default"},
        "channels": [],
        "dataset": "ds",
        "dependencies": ["numpy >= 1.2.0"],
        "hyperparameters": {
            "batchSize": 32,
            "learningRate": 0.01,
            "rounds": 5,
            "trainer_indices_list": indices,
        },
        "baseModel": {"name": "", "version": 1},
        "job": {"id": "abc", "name": "test"},
        "registry": {"sort": "dummy", "uri": ""},
        "selector": {"sort": "default", "kwargs": {}},
        "optimizer": {"sort": "fedavg", "kwargs": {}},
        "maxRunTime": 300,
        "realm": realm,
        "role": "trainer",
    }


@pytest.fixture
def fake_example(tmp_path):
    ex = tmp_path / "example"
    trainer = ex / "trainer"
    trainer.mkdir(parents=True)

    # Two config dirs, three trainers. config_a: all same realm. config_b: per-trainer realm.
    for cfg_name, realms in [
        ("config_a", {1: "default/us", 2: "default/us", 3: "default/us"}),
        ("config_b", {1: "default/us", 2: "default/uk", 3: "default/india"}),
    ]:
        d = trainer / cfg_name
        d.mkdir()
        for tid in (1, 2, 3):
            taskid = f"task_{tid:03d}"
            indices = list(range(tid * 100, tid * 100 + 50))
            data = _make_trainer_json(taskid, indices, realms[tid])
            (d / f"trainer_{tid}.json").write_text(
                json.dumps(data, indent=4)
            )
    return ex


class TestMigrate:
    def test_produces_registry_and_splits(self, fake_example, tmp_path):
        from scripts.migrate_trainer_configs import migrate_example

        metadata = tmp_path / "_meta"
        plan_path = tmp_path / "plan.yaml"
        plan = migrate_example(
            example_dir=fake_example,
            metadata_out=metadata,
            plan_out=plan_path,
            namespace="test",
            dataset_name="ds",
        )

        assert plan_path.is_file()
        assert (metadata / "trainer_registry_test.yaml").is_file()
        assert (metadata / "dataset_splits" / "ds_test_config_a.yaml").is_file()
        assert (metadata / "dataset_splits" / "ds_test_config_b.yaml").is_file()
        assert len(plan["directories"]) == 2
        # config_b should have per-trainer overrides for trainers 2 and 3 (realm).
        b = next(d for d in plan["directories"] if d["name"] == "config_b")
        assert "trainer_002" in b["per_trainer_overrides"]
        assert "trainer_003" in b["per_trainer_overrides"]
        assert "trainer_001" not in b["per_trainer_overrides"]
        # config_a has no overrides.
        a = next(d for d in plan["directories"] if d["name"] == "config_a")
        assert a["per_trainer_overrides"] == {}


class TestVerify:
    def test_strict_byte_equivalence(self, fake_example, tmp_path):
        from scripts.migrate_trainer_configs import migrate_example
        from scripts.verify_migration import verify

        metadata = tmp_path / "_meta"
        plan_path = tmp_path / "plan.yaml"
        migrate_example(
            example_dir=fake_example,
            metadata_out=metadata,
            plan_out=plan_path,
            namespace="test",
            dataset_name="ds",
        )

        passed, failed, failures = verify(plan_path)
        assert failed == 0, failures[:2]
        assert passed == 6  # 2 dirs × 3 trainers

    def test_detects_tampering(self, fake_example, tmp_path):
        from scripts.migrate_trainer_configs import migrate_example
        from scripts.verify_migration import verify

        metadata = tmp_path / "_meta"
        plan_path = tmp_path / "plan.yaml"
        migrate_example(
            example_dir=fake_example,
            metadata_out=metadata,
            plan_out=plan_path,
            namespace="test",
            dataset_name="ds",
        )
        # Corrupt one source JSON post-migration.
        tampered = fake_example / "trainer" / "config_a" / "trainer_1.json"
        data = json.loads(tampered.read_text())
        data["hyperparameters"]["trainer_indices_list"][0] = 999999
        tampered.write_text(json.dumps(data, indent=4))

        passed, failed, failures = verify(plan_path)
        assert failed == 1
        assert "trainer_1.json" in failures[0]["source"]


class TestTaskIdStability:
    def test_rejects_inconsistent_task_ids(self, fake_example, tmp_path):
        from scripts.migrate_trainer_configs import migrate_example

        # Modify config_b/trainer_1.json to have a different taskid.
        f = fake_example / "trainer" / "config_b" / "trainer_1.json"
        d = json.loads(f.read_text())
        d["taskid"] = "different_task_id"
        f.write_text(json.dumps(d, indent=4))

        with pytest.raises(ValueError, match="multiple task_ids"):
            migrate_example(
                example_dir=fake_example,
                metadata_out=tmp_path / "_meta",
                plan_out=tmp_path / "plan.yaml",
                namespace="test",
                dataset_name="ds",
            )
