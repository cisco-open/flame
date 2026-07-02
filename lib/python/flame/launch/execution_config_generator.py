"""Generate compact execution_config.yaml records pointing at shared metadata."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flame.launch.experiment_config import ExperimentConfig


def get_git_info() -> Dict[str, str]:
    """Get current git state."""
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
                ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        dirty = len(status_output.strip()) > 0

        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as e:
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False,
            "error": str(e),
        }


def create_execution_config(
    exp_config: ExperimentConfig,
    aggregator_config_path: Path,
    spawn_commands: Optional[Dict[str, List[str]]] = None,
    execution_timestamp: Optional[datetime] = None,
    agg_cfg: Optional[Dict] = None,
) -> Dict:
    """
    Generate compact execution config with metadata keys.

    Args:
        exp_config: Experiment configuration
        aggregator_config_path: Path to aggregator JSON config
        spawn_commands: Actual spawn commands used (optional)
        execution_timestamp: Timestamp of execution (default: now)
        agg_cfg: The real, merged aggregator config (if available) -- used to
            record the effective `agg_goal` instead of the possibly-unset
            `exp_config.aggregator.agg_goal` typed field.

    Returns:
        Dictionary with compact execution config
    """
    if execution_timestamp is None:
        execution_timestamp = datetime.now()

    # Get git state
    git_info = get_git_info()

    # Build metadata keys. Must match the real lookup key computed in
    # flame/launch/spawner.py:ConfigGenerator.get_dataset_split() -- this is
    # a record of what was actually loaded, not an independent guess, so it
    # has to use the same (dataset_name, alpha, num_trainers) inputs the
    # runner passes to spawn_all() (see runner.py:run_experiment).
    alpha = exp_config.trainer.dataset.dirichlet_alpha
    dataset_split_key = (
        f"{exp_config.trainer.dataset.name}_alpha{alpha}_n{exp_config.trainer.num_trainers}"
    )
    availability_trace_key = exp_config.trainer.availability.mode

    # Build compact config
    config = {
        "execution_timestamp": execution_timestamp.isoformat(),
        "git_info": git_info,
        # Metadata references by KEY (not embedded data)
        "metadata_refs": {
            "trainer_registry": "metadata/trainer_registry.yaml",
            "dataset_split_key": dataset_split_key,
            "availability_trace_key": availability_trace_key,
        },
        # Aggregator config by PATH (not embedded)
        "aggregator": {
            "config_file": str(aggregator_config_path),
            "selector": (
                exp_config.aggregator.selector if exp_config.aggregator else "default"
            ),
            "tracking_mode": (
                exp_config.aggregator.tracking_mode
                if exp_config.aggregator
                else "default"
            ),
            "agg_goal": (
                (agg_cfg or {}).get("hyperparameters", {}).get("aggGoal")
                if agg_cfg is not None
                else (exp_config.aggregator.agg_goal if exp_config.aggregator else None)
            ),
        },
        # Experiment parameters (what changes between runs)
        "experiment": {
            "name": exp_config.name,
            "description": exp_config.description,
            "trainer": {
                "num_trainers": exp_config.trainer.num_trainers,
                "start_id": exp_config.trainer.start_id,
                "trainer_id_range": [
                    exp_config.trainer.start_id,
                    exp_config.trainer.start_id + exp_config.trainer.num_trainers - 1,
                ],
                "dataset": {
                    "name": exp_config.trainer.dataset.name,
                    "alpha": alpha,
                    "split_key": dataset_split_key,
                },
                "availability": {
                    "mode": exp_config.trainer.availability.mode,
                    "trace_key": availability_trace_key,
                },
                "battery_threshold": exp_config.trainer.battery_threshold,
                "time_mode": exp_config.trainer.time_mode,
            },
            "execution": {
                "num_gpus": exp_config.execution.num_gpus,
                "sleep_between_spawns": exp_config.execution.sleep_between_spawns,
                "aggregator_warmup_time": exp_config.execution.aggregator_warmup_time,
            },
        },
    }

    # Add spawn commands if provided
    if spawn_commands:
        config["spawn_commands"] = spawn_commands

    return config


def save_execution_config(config: Dict, output_path: Path):
    """
    Save execution config to YAML file.

    Args:
        config: Execution config dictionary
        output_path: Path to save YAML file
    """
    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print(f"  ✓ Saved execution config: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(
        f"    Git: {config['git_info']['commit'][:7]} ({config['git_info']['branch']})"
    )
    print(f"    Metadata keys:")
    print(f"      - Dataset: {config['metadata_refs']['dataset_split_key']}")
    print(f"      - Availability: {config['metadata_refs']['availability_trace_key']}")


