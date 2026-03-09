"""
Experiment configuration schema for Phase 3.

Defines data structures for experiment configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "cifar10"
    dirichlet_alpha: float = 0.1


@dataclass
class AvailabilityConfig:
    """Availability trace configuration."""

    mode: str = (
        "mobiperf_2st"  # mobiperf_2st, mobiperf_3st_50, mobiperf_3st_75, syn_0, syn_20, syn_50
    )


@dataclass
class TrainerConfig:
    """Trainer configuration."""

    num_trainers: int = 300
    start_id: int = 1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    battery_threshold: int = 50  # For 3-state modes
    speedup_factor: float = 1.0
    enable_training_delays: bool = True  # Enable per-trainer training delays
    hyperparameters: Optional[dict] = None  # Trainer-specific hyperparameters (e.g., batchSize, learningRate)


@dataclass
class AggregatorConfig:
    """Aggregator configuration."""

    config_template: str  # Path to aggregator JSON config
    selector: str = "oort"
    tracking_mode: str = "oracular"  # oracular, default
    agg_goal: int = 10
    log_to_wandb: bool = False  # Enable wandb logging
    wandb_run_name: Optional[str] = None  # Custom wandb run name


@dataclass
class MonitoringConfig:
    """Resource monitoring configuration."""

    enabled: bool = True
    check_interval_seconds: int = 30
    ram_warning_percent: float = 80.0
    ram_critical_percent: float = 90.0
    gpu_warning_percent: float = 80.0
    gpu_critical_percent: float = 90.0


@dataclass
class ExecutionConfig:
    """Execution configuration."""

    num_gpus: int = 8
    sleep_between_spawns: float = 1.0
    aggregator_warmup_time: int = 10
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    name: str
    description: Optional[str] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    aggregator: AggregatorConfig = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def get_log_prefix(self, timestamp_str: str = None) -> str:
        """
        Generate descriptive log filename prefix.

        Format: DD_MM_YY_HH_MM_<selector>_n<num>_<tracking>_alpha<alpha>_<avail>
        Example: 03_02_26_14_30_oort_n300_oracular_alpha0p1_mobiperf2st

        Args:
            timestamp_str: Optional timestamp string in DD_MM_YY_HH_MM format.
                         If not provided, uses current time.
        """
        from datetime import datetime

        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%d_%m_%y_%H_%M")

        alpha = self.trainer.dataset.dirichlet_alpha
        alpha_str = f"alpha{alpha}".replace(".", "p")

        avail = self.trainer.availability.mode.replace("_", "")

        selector = self.aggregator.selector if self.aggregator else "default"
        tracking = self.aggregator.tracking_mode if self.aggregator else "default"

        prefix = (
            f"{timestamp_str}_{selector}_n{self.trainer.num_trainers}_"
            f"{tracking}_{alpha_str}_{avail}"
        )

        return prefix


@dataclass
class ExperimentBatch:
    """Batch of experiments to run sequentially."""

    experiments: List[ExperimentConfig]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentBatch":
        """Load experiment batch from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        experiments = []
        for exp_data in data.get("experiments", []):
            # Parse nested configs
            trainer_data = exp_data.get("trainer", {})
            dataset_data = trainer_data.get("dataset", {})
            avail_data = trainer_data.get("availability", {})
            agg_data = exp_data.get("aggregator", {})
            exec_data = exp_data.get("execution", {})
            monitor_data = exec_data.get("monitoring", {}) if exec_data else {}

            exp = ExperimentConfig(
                name=exp_data["name"],
                description=exp_data.get("description"),
                trainer=TrainerConfig(
                    num_trainers=trainer_data.get("num_trainers", 300),
                    start_id=trainer_data.get("start_id", 1),
                    dataset=(
                        DatasetConfig(**dataset_data)
                        if dataset_data
                        else DatasetConfig()
                    ),
                    availability=(
                        AvailabilityConfig(**avail_data)
                        if avail_data
                        else AvailabilityConfig()
                    ),
                    battery_threshold=trainer_data.get("battery_threshold", 50),
                    speedup_factor=trainer_data.get("speedup_factor", 1.0),
                    enable_training_delays=trainer_data.get("enable_training_delays", True),
                    hyperparameters=trainer_data.get("hyperparameters"),
                ),
                aggregator=AggregatorConfig(**agg_data) if agg_data else None,
                execution=(
                    ExecutionConfig(
                        num_gpus=exec_data.get("num_gpus", 8),
                        sleep_between_spawns=exec_data.get("sleep_between_spawns", 1.0),
                        aggregator_warmup_time=exec_data.get("aggregator_warmup_time", 10),
                        monitoring=(
                            MonitoringConfig(**monitor_data)
                            if monitor_data
                            else MonitoringConfig()
                        ),
                    )
                    if exec_data
                    else ExecutionConfig()
                ),
            )
            experiments.append(exp)

        return cls(experiments=experiments)


def load_experiment_config(yaml_path: Path) -> ExperimentBatch:
    """Load experiment configuration from YAML file."""
    return ExperimentBatch.from_yaml(yaml_path)
