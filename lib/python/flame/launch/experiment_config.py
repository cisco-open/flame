"""
Experiment configuration schema.

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
    # True for path-style datasets (e.g. H5 file paths) that have no
    # _metadata/dataset_splits/<name>_alpha<a>_n<N>.yaml index-list file.
    # Skips the index-split lookup in ConfigGenerator.generate_trainer_config().
    path_style: bool = False


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
    # Which <name>_alpha<a>_n<N>.yaml split file to read, i.e. the N the
    # dataset was partitioned for. Defaults to num_trainers. Set it larger
    # than num_trainers to spawn a subset cohort (the first num_trainers
    # trainers) against an existing wider partition -- e.g. num_trainers=10
    # + split_num_trainers=300 runs a 10-trainer smoke off the n300 split
    # without needing a dedicated n10 split file. See spawn_all's docstring:
    # this N is a property of the partition, not of how many are spawned.
    split_num_trainers: Optional[int] = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    battery_threshold: int = 50  # For 3-state modes
    # Simulation time mode: "real" (trainer sleeps modeled delays at true pace)
    # or "simulated" (no sleep; aggregator orders by a virtual clock). Replaces
    # the removed speedup_factor.
    time_mode: str = "simulated"
    enable_training_delays: bool = True  # Enable per-trainer training delays
    hyperparameters: Optional[dict] = None  # Trainer-specific hyperparameters (e.g., batchSize, learningRate)
    config_overrides: Optional[dict] = None  # Deep-merged into per-trainer config last (wins over baseline)
    # When set, the runner injects hyperparameters.client_idx =
    # (trainer_id - 1) % client_idx_modulo per trainer, for path-style
    # datasets (e.g. H5 partitions) that need N trainers wrapped onto M
    # data partitions. None = no per-trainer client_idx injection.
    client_idx_modulo: Optional[int] = None


@dataclass
class AggregatorConfig:
    """Aggregator configuration.

    `selector`/`tracking_mode` do NOT configure the aggregator -- they are
    descriptive labels (log filename, snapshot/execution_config records)
    only. The real selector/optimizer/hyperparameters come exclusively from
    `config_template` -> baseline -> `config_overrides` (see
    runner.py:_build_aggregator_config). The runner validates `selector`
    against the real merged value and raises on mismatch, so a stale label
    is caught rather than silently ignored.
    """

    config_template: Optional[str] = None  # Path to aggregator JSON config (optional when using baseline)
    selector: str = "oort"
    tracking_mode: str = "oracular"  # oracular, default
    # Single source of truth for the aggregation-goal count. When set, the
    # runner fans this value into every real runtime consumer as the final
    # merge layer, so they can never drift apart:
    #   - hyperparameters.aggGoal: aggregator's wait-for-N-contributions
    #     threshold (config.hyperparameters.aggregation_goal).
    #   - selector.kwargs.aggGoal: read by fedbuff/async_random/async_oort/
    #     oracle selectors.
    #   - selector.kwargs.aggr_num: read by oort/refl_oort/feddance
    #     selectors (same concept, different kwarg name).
    # None = leave whatever config_template/baseline/config_overrides
    # already produced untouched.
    agg_goal: Optional[int] = None
    log_to_wandb: bool = False
    wandb_run_name: Optional[str] = None
    config_overrides: Optional[dict] = None  # Deep-merged into aggregator JSON last (wins over baseline)


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
class ExampleConfig:
    """Per-example layout. Paths are relative to example_dir unless absolute."""

    dir: Optional[str] = None
    trainer_main: str = "trainer/pytorch/main.py"
    aggregator_main: Optional[str] = None  # owned by baseline; falls back to default in runner
    trainer_base: str = "configs/trainer_base.yaml"


@dataclass
class MetadataPaths:
    """Shared-metadata locations. Paths may be absolute or repo-relative.

    Only `dir` and `registry` are real -- `MetadataLoader` (spawner.py)
    takes a single root directory and reads fixed `dataset_splits/` and
    `availability_traces/` subdirectories under it; there is no per-
    component override hook to plug a separate location into. Don't re-add
    `dataset_splits_dir`/`traces_dir`-shaped fields here unless
    `MetadataLoader` is actually changed to accept them -- that pairing
    (declared-but-never-wired field) is the exact bug class this file's
    `AggregatorConfig.agg_goal` history is a cautionary tale for.
    """

    dir: Optional[str] = None
    registry: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    name: str
    description: Optional[str] = None
    baseline: Optional[str] = None  # Key into _metadata/baselines.yaml
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    aggregator: AggregatorConfig = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    example: ExampleConfig = field(default_factory=ExampleConfig)
    metadata: MetadataPaths = field(default_factory=MetadataPaths)

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

            example_data = exp_data.get("example", {})
            metadata_data = exp_data.get("metadata", {})

            exp = ExperimentConfig(
                name=exp_data["name"],
                description=exp_data.get("description"),
                baseline=exp_data.get("baseline"),
                example=ExampleConfig(**example_data) if example_data else ExampleConfig(),
                metadata=MetadataPaths(**metadata_data) if metadata_data else MetadataPaths(),
                trainer=TrainerConfig(
                    num_trainers=trainer_data.get("num_trainers", 300),
                    start_id=trainer_data.get("start_id", 1),
                    split_num_trainers=trainer_data.get("split_num_trainers"),
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
                    time_mode=trainer_data.get("time_mode", "simulated"),
                    enable_training_delays=trainer_data.get("enable_training_delays", True),
                    hyperparameters=trainer_data.get("hyperparameters"),
                    config_overrides=trainer_data.get("config_overrides"),
                    client_idx_modulo=trainer_data.get("client_idx_modulo"),
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
