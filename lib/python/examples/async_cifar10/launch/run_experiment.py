#!/usr/bin/env python3
"""
Experiment Runner for Phase 3.

Orchestrates complete experiments:
- Spawns aggregator
- Spawns trainers
- Captures logs to files
- Creates reproducible snapshots
- Handles cleanup on Ctrl+C
"""
import sys
import signal
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from launch.experiment_config import load_experiment_config, ExperimentConfig
from launch.spawner import MetadataLoader, ConfigGenerator, TrainerSpawner
from launch.aggregator_spawner import AggregatorSpawner
from launch.snapshot import ExperimentSnapshot
from launch.execution_config_generator import (
    create_execution_config,
    save_execution_config,
)
from launch.resource_monitor import create_monitor_from_config


class ExperimentRunner:
    """Orchestrates complete federated learning experiments."""

    def __init__(self, example_dir: Path):
        self.example_dir = example_dir
        self.metadata_dir = example_dir / "metadata"
        self.experiments_dir = example_dir / "experiments"

        self.current_exp_dir = None
        self.aggregator_spawner = None
        self.trainer_spawner = None
        self.resource_monitor = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run_experiment(self, exp_config: ExperimentConfig):
        """Run a single experiment."""
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT: {exp_config.name}")
        print("=" * 70)

        # Initialize paths early for exception handling
        aggregator_config_path = None
        aggregator_main_path = None
        agg_log_file = None
        trainers_log_file = None

        try:
            # Step 1: Setup experiment directory
            print("\n[1/6] Setting up experiment directory...")
            self.current_exp_dir = self._create_experiment_directory(exp_config)
            print(f"  ✓ Created: {self.current_exp_dir}")

            # Step 2: Initialize spawners
            print("\n[2/6] Initializing spawners...")

            # Load aggregator config to extract job ID
            aggregator_config_path = (
                self.example_dir / exp_config.aggregator.config_template
            )
            if not aggregator_config_path.exists():
                raise FileNotFoundError(
                    f"Aggregator config not found: {aggregator_config_path}"
                )

            with open(aggregator_config_path) as f:
                agg_config = json.load(f)
                agg_job_id = agg_config.get("job", {}).get("id")
                agg_job_name = agg_config.get("job", {}).get("name")

            if not agg_job_id:
                raise ValueError(
                    f"Aggregator config missing job.id: {aggregator_config_path}"
                )

            print(f"  Aggregator job ID: {agg_job_id}")

            metadata_loader = MetadataLoader(self.metadata_dir)
            config_gen = ConfigGenerator(
                metadata_loader, self.example_dir / "configs" / "trainer_base.yaml"
            )

            # Get log file paths with descriptive names
            log_prefix = exp_config.get_log_prefix()

            agg_log_file = self.current_exp_dir / f"{log_prefix}_aggregator.log"
            trainers_log_file = self.current_exp_dir / f"{log_prefix}_trainers.log"
            monitor_log_file = self.current_exp_dir / f"{log_prefix}_resources.log"

            self.aggregator_spawner = AggregatorSpawner(log_file=agg_log_file)
            self.trainer_spawner = TrainerSpawner(
                config_gen,
                num_gpus=exp_config.execution.num_gpus,
                sleep_between_spawns=exp_config.execution.sleep_between_spawns,
                log_file=trainers_log_file,
            )

            # Initialize resource monitor if enabled
            if exp_config.execution.monitoring.enabled:
                monitor_config = {
                    "check_interval_seconds": exp_config.execution.monitoring.check_interval_seconds,
                    "ram_warning_percent": exp_config.execution.monitoring.ram_warning_percent,
                    "ram_critical_percent": exp_config.execution.monitoring.ram_critical_percent,
                    "gpu_warning_percent": exp_config.execution.monitoring.gpu_warning_percent,
                    "gpu_critical_percent": exp_config.execution.monitoring.gpu_critical_percent,
                }
                self.resource_monitor = create_monitor_from_config(
                    monitor_log_file, monitor_config
                )
                print(f"  ✓ Resource monitoring enabled (interval: {exp_config.execution.monitoring.check_interval_seconds}s)")
                print(f"    RAM thresholds: {exp_config.execution.monitoring.ram_warning_percent}% warning, {exp_config.execution.monitoring.ram_critical_percent}% critical")
                print(f"    GPU thresholds: {exp_config.execution.monitoring.gpu_warning_percent}% warning, {exp_config.execution.monitoring.gpu_critical_percent}% critical")

            print("  ✓ Spawners initialized")

            # Step 3: Start aggregator
            print("\n[3/6] Starting aggregator...")
            aggregator_main_path = (
                self.example_dir / "aggregator" / "pytorch" / "main_oort_agg.py"
            )

            print(f"  Spawning aggregator with config: {aggregator_config_path}")
            if exp_config.aggregator.log_to_wandb:
                print(f"  Wandb logging enabled")
                if exp_config.aggregator.wandb_run_name:
                    print(f"    Run name: {exp_config.aggregator.wandb_run_name}")
            self.aggregator_spawner.spawn(
                aggregator_main_path,
                aggregator_config_path,
                log_to_wandb=exp_config.aggregator.log_to_wandb,
                wandb_run_name=exp_config.aggregator.wandb_run_name,
            )

            # Wait for aggregator to be ready
            if not self.aggregator_spawner.wait_until_ready(
                exp_config.execution.aggregator_warmup_time
            ):
                raise RuntimeError("Aggregator failed to start")

            # Start resource monitoring after aggregator is up
            if self.resource_monitor:
                self.resource_monitor.start()

            # Step 4: Create execution config and snapshot
            print("\n[4/6] Creating execution config and snapshot...")

            # Build spawn commands for record
            trainer_spawn_cmd = self._build_trainer_spawn_command(exp_config)
            agg_spawn_cmd = [
                sys.executable,
                str(aggregator_main_path),
                str(aggregator_config_path),  # Positional argument
            ]

            # Create compact execution config (primary reproducibility record)
            exec_config = create_execution_config(
                exp_config,
                aggregator_config_path.relative_to(self.example_dir),
                spawn_commands={
                    "aggregator": [str(c) for c in agg_spawn_cmd],
                    "trainers": [str(c) for c in trainer_spawn_cmd],
                },
            )
            exec_config_path = self.current_exp_dir / "execution_config.yaml"
            save_execution_config(exec_config, exec_config_path)

            # Also create legacy snapshot for compatibility
            snapshot = ExperimentSnapshot(self.current_exp_dir)
            snapshot.create_snapshot(
                exp_config,
                self.metadata_dir,
                aggregator_config_path,
                trainer_spawn_cmd,
                agg_spawn_cmd,
            )

            # Step 5: Spawn trainers
            print("\n[5/6] Spawning trainers...")
            trainer_ids = list(
                range(
                    exp_config.trainer.start_id,
                    exp_config.trainer.start_id + exp_config.trainer.num_trainers,
                )
            )

            # Pass aggregator job ID to trainers for MQTT communication
            # Prepare config overrides
            config_overrides = {
                "job.id": agg_job_id,
                "job.name": agg_job_name,
                "hyperparameters.training_delay_enabled": str(exp_config.trainer.enable_training_delays),
            }
            
            # Apply trainer-specific hyperparameters from experiment config (if specified)
            if exp_config.trainer.hyperparameters:
                print(f"  Applying trainer hyperparameters: {exp_config.trainer.hyperparameters}")
                for key, value in exp_config.trainer.hyperparameters.items():
                    config_overrides[f"hyperparameters.{key}"] = value
            
            self.trainer_spawner.spawn_all(
                trainer_ids,
                alpha=exp_config.trainer.dataset.dirichlet_alpha,
                availability_mode=exp_config.trainer.availability.mode,
                trainer_main_path=self.example_dir / "trainer" / "pytorch" / "main.py",
                **config_overrides,
            )

            # Step 6: Monitor
            print("\n[6/6] Monitoring experiment...")
            print("\n" + "─" * 70)
            print("EXPERIMENT RUNNING")
            print("─" * 70)
            print(f"\nExperiment directory: {self.current_exp_dir}")
            print(f"Aggregator log:       tail -f {agg_log_file}")
            print(f"Trainers log:         tail -f {trainers_log_file}")
            if self.resource_monitor:
                print(f"Resource monitor:     tail -f {monitor_log_file}")
            print(f"\nPress Ctrl+C to terminate experiment...")
            print("─" * 70 + "\n")

            # Wait for processes
            self.trainer_spawner.wait_all()

            print("\n✓ Experiment completed successfully")

        except Exception as e:
            import traceback

            print(f"\n✗ Experiment failed: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise
        finally:
            self._cleanup()

    def run_experiment_batch(self, config_file: Path):
        """Run batch of experiments sequentially."""
        print("=" * 70)
        print("EXPERIMENT BATCH RUNNER")
        print("=" * 70)

        # Load experiment batch
        batch = load_experiment_config(config_file)

        print(f"\nLoaded {len(batch.experiments)} experiments from {config_file.name}")
        for i, exp in enumerate(batch.experiments, 1):
            print(f"  {i}. {exp.name}")

        # Run each experiment
        for i, exp_config in enumerate(batch.experiments, 1):
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {i}/{len(batch.experiments)}")
            print(f"{'='*70}")

            try:
                self.run_experiment(exp_config)
            except Exception as e:
                print(f"\n✗ Experiment {exp_config.name} failed: {e}")
                # Ask user if they want to continue
                response = input("\nContinue with remaining experiments? (y/n): ")
                if response.lower() != "y":
                    break

        print("\n" + "=" * 70)
        print("BATCH COMPLETE")
        print("=" * 70)

    def _create_experiment_directory(self, exp_config: ExperimentConfig) -> Path:
        """Create timestamped experiment directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.experiments_dir / f"run_{timestamp}_{exp_config.name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _build_trainer_spawn_command(self, exp_config: ExperimentConfig) -> list:
        """Build trainer spawn command for snapshot."""
        return [
            "python3",
            "launch/spawner.py",
            "--alpha",
            str(exp_config.trainer.dataset.dirichlet_alpha),
            "--availability",
            exp_config.trainer.availability.mode,
            "--num-trainers",
            str(exp_config.trainer.num_trainers),
            "--start-id",
            str(exp_config.trainer.start_id),
            "--num-gpus",
            str(exp_config.execution.num_gpus),
        ]

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and SIGTERM."""
        print("\n\n" + "=" * 70)
        print("TERMINATION SIGNAL RECEIVED")
        print("=" * 70)
        self._cleanup()
        sys.exit(130)  # Standard exit code for SIGINT

    def _cleanup(self):
        """Cleanup all processes."""
        print("Cleaning up...")

        if self.resource_monitor:
            self.resource_monitor.stop()

        if self.trainer_spawner:
            self.trainer_spawner.terminate_all()

        if self.aggregator_spawner:
            print("Terminating aggregator...")
            self.aggregator_spawner.terminate()

        print("✓ Cleanup complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run federated learning experiments")
    parser.add_argument(
        "config", type=Path, help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--from-snapshot",
        type=Path,
        help="Reproduce experiment from snapshot file",
    )

    args = parser.parse_args()

    # Get example directory
    example_dir = Path(__file__).parent.parent

    # Create runner
    runner = ExperimentRunner(example_dir)

    # Run experiment(s)
    if args.from_snapshot:
        print("Reproducing from snapshot...")
        # TODO: Implement snapshot reproduction
        raise NotImplementedError("Snapshot reproduction not yet implemented")
    else:
        runner.run_experiment_batch(args.config)


if __name__ == "__main__":
    main()
