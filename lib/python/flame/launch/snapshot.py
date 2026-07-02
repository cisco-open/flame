"""
Configuration snapshot system for experiment reproducibility.

Saves complete experiment state including:
- Experiment configuration
- Spawner commands
- Aggregator config copy
- Metadata checksums
- Git commit info
"""

import yaml
import hashlib
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from flame.launch.experiment_config import ExperimentConfig


class ExperimentSnapshot:
    """Manages experiment snapshots for reproducibility."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.snapshot_file = experiment_dir / "snapshot.yaml"

    def create_snapshot(
        self,
        exp_config: ExperimentConfig,
        metadata_dir: Path,
        aggregator_config_path: Path,
        trainer_spawn_command: List[str],
        aggregator_spawn_command: List[str],
        agg_cfg: Dict = None,
    ):
        """
        Create complete experiment snapshot.

        Args:
            exp_config: Experiment configuration
            metadata_dir: Path to metadata directory
            aggregator_config_path: Path to aggregator JSON config
            trainer_spawn_command: Command used to spawn trainers
            aggregator_spawn_command: Command used to spawn aggregator
            agg_cfg: The real, merged aggregator config (if available) -- used
                to record the effective `agg_goal` instead of the
                possibly-unset `exp_config.aggregator.agg_goal` typed field.
        """
        snapshot_data = {
            "snapshot_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "hostname": subprocess.check_output(["hostname"]).decode().strip(),
            "experiment": self._serialize_experiment_config(exp_config, agg_cfg),
            "spawn_commands": {
                "aggregator": aggregator_spawn_command,
                "trainers": trainer_spawn_command,
            },
            "git_info": self._get_git_info(),
            "metadata_location": str(metadata_dir.absolute()),
            "metadata_checksums": self._compute_metadata_checksums(metadata_dir),
            "aggregator_config": str(aggregator_config_path),
        }

        # Save snapshot
        with open(self.snapshot_file, "w") as f:
            yaml.dump(snapshot_data, f, default_flow_style=False, sort_keys=False)

        # Copy aggregator config only if it's outside the experiment dir
        # (the runner now writes the merged config directly into self.experiment_dir).
        agg_copy = self.experiment_dir / "aggregator_config.json"
        if Path(aggregator_config_path).resolve() != agg_copy.resolve():
            shutil.copy(aggregator_config_path, agg_copy)
            print(f"  ✓ Aggregator config copied: {agg_copy}")

        print(f"  ✓ Snapshot saved: {self.snapshot_file}")
        print(f"  ✓ Metadata location recorded: {metadata_dir}")

    def _serialize_experiment_config(
        self, exp_config: ExperimentConfig, agg_cfg: Dict = None
    ) -> Dict:
        """Serialize experiment config to dict."""
        effective_agg_goal = (
            (agg_cfg or {}).get("hyperparameters", {}).get("aggGoal")
            if agg_cfg is not None
            else (exp_config.aggregator.agg_goal if exp_config.aggregator else None)
        )
        return {
            "name": exp_config.name,
            "description": exp_config.description,
            "trainer": {
                "num_trainers": exp_config.trainer.num_trainers,
                "start_id": exp_config.trainer.start_id,
                "dataset": {
                    "name": exp_config.trainer.dataset.name,
                    "dirichlet_alpha": exp_config.trainer.dataset.dirichlet_alpha,
                },
                "availability": {
                    "mode": exp_config.trainer.availability.mode,
                },
                "battery_threshold": exp_config.trainer.battery_threshold,
                "time_mode": exp_config.trainer.time_mode,
            },
            "aggregator": (
                {
                    "config_template": exp_config.aggregator.config_template,
                    "selector": exp_config.aggregator.selector,
                    "tracking_mode": exp_config.aggregator.tracking_mode,
                    "agg_goal": effective_agg_goal,
                }
                if exp_config.aggregator
                else None
            ),
            "execution": {
                "num_gpus": exp_config.execution.num_gpus,
                "sleep_between_spawns": exp_config.execution.sleep_between_spawns,
                "aggregator_warmup_time": exp_config.execution.aggregator_warmup_time,
            },
        }

    def _get_git_info(self) -> Dict:
        """Get current git commit information."""
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            # Check for uncommitted changes
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            return {
                "commit": commit,
                "branch": branch,
                "clean": len(status) == 0,
                "uncommitted_changes": bool(status),
            }
        except Exception:
            # Git may not be available or not a git repository
            return {"error": "Git information not available"}

    def _compute_metadata_checksums(self, metadata_dir: Path) -> Dict:
        """Compute checksums for metadata files."""
        checksums = {}

        # Trainer registry
        registry_file = metadata_dir / "trainer_registry.yaml"
        if registry_file.exists():
            checksums["trainer_registry"] = self._file_checksum(registry_file)

        # Dataset splits
        splits_dir = metadata_dir / "dataset_splits"
        if splits_dir.exists():
            checksums["dataset_splits"] = {}
            for split_file in sorted(splits_dir.glob("*.yaml")):
                checksums["dataset_splits"][split_file.name] = self._file_checksum(
                    split_file
                )

        # Availability traces
        traces_dir = metadata_dir / "availability_traces"
        if traces_dir.exists():
            checksums["availability_traces"] = {}
            for trace_file in sorted(traces_dir.glob("*.yaml")):
                checksums["availability_traces"][trace_file.name] = self._file_checksum(
                    trace_file
                )

        return checksums

    def _file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()  # Use full hash for collision resistance

    @classmethod
    def load_snapshot(cls, snapshot_file: Path) -> Dict:
        """Load experiment snapshot for reproduction."""
        with open(snapshot_file) as f:
            return yaml.safe_load(f)
