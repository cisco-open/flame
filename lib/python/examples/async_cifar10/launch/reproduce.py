#!/usr/bin/env python3
"""
Experiment Reproduction Tool.

Reproduces experiments from compact execution configs.
Loads metadata using keys, spawns processes with exact parameters.
"""
import sys
import signal
import yaml
import subprocess
from pathlib import Path
from typing import Dict
from datetime import datetime


class ExperimentReproducer:
    """Reproduces experiments from execution configs."""
    
    def __init__(self, example_dir: Path):
        self.example_dir = example_dir
        self.metadata_dir = example_dir / 'metadata'
        self.aggregator_spawner = None
        self.trainer_spawner = None
        self.current_exp_dir = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def reproduce_from_config(self, exec_config_path: Path):
        """
        Reproduce experiment from execution config.
        
        Args:
            exec_config_path: Path to execution_config.yaml
        """
        print("\n" + "="*70)
        print("EXPERIMENT REPRODUCTION")
        print("="*70)
        
        # Load execution config
        print(f"\n[1/7] Loading execution config: {exec_config_path}")
        with open(exec_config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ Experiment: {config['experiment']['name']}")
        print(f"  ✓ Timestamp: {config['execution_timestamp']}")
        
        # Check git state
        print("\n[2/7] Checking git state...")
        self._check_git_state(config['git_info'])
        
        # Verify metadata exists
        print("\n[3/7] Verifying metadata...")
        self._verify_metadata(config['metadata_refs'])
        
        # Setup new experiment directory
        print("\n[4/7] Setting up experiment directory...")
        self.current_exp_dir = self._create_experiment_directory(config)
        print(f"  ✓ Created: {self.current_exp_dir}")
        
        # Initialize spawners
        print("\n[5/7] Initializing spawners...")
        from launch.spawner import MetadataLoader, ConfigGenerator, TrainerSpawner
        from launch.aggregator_spawner import AggregatorSpawner
        
        metadata_loader = MetadataLoader(self.metadata_dir)
        config_gen = ConfigGenerator(
            metadata_loader,
            self.example_dir / 'configs' / 'trainer_base.yaml'
        )
        
        # Create log files
        log_prefix = self._create_log_prefix(config)
        agg_log_file = self.current_exp_dir / f"{log_prefix}_aggregator.log"
        trainers_log_file = self.current_exp_dir / f"{log_prefix}_trainers.log"
        
        self.aggregator_spawner = AggregatorSpawner(log_file=agg_log_file)
        self.trainer_spawner = TrainerSpawner(
            config_gen,
            num_gpus=config['experiment']['execution']['num_gpus'],
            sleep_between_spawns=config['experiment']['execution']['sleep_between_spawns'],
            log_file=trainers_log_file
        )
        print("  ✓ Spawners initialized")
        
        # Start aggregator
        print("\n[6/7] Starting aggregator...")
        agg_config_path = self.example_dir / config['aggregator']['config_file']
        agg_main_path = self.example_dir / 'aggregator' / 'pytorch' / 'main_oort_agg.py'
        
        if not agg_config_path.exists():
            raise FileNotFoundError(f"Aggregator config not found: {agg_config_path}")
        
        self.aggregator_spawner.spawn(agg_main_path, agg_config_path)
        
        if not self.aggregator_spawner.wait_until_ready(
            config['experiment']['execution']['aggregator_warmup_time']
        ):
            raise RuntimeError("Aggregator failed to start")
        
        # Spawn trainers
        print("\n[7/7] Spawning trainers...")
        exp = config['experiment']
        trainer_ids = list(range(
            exp['trainer']['start_id'],
            exp['trainer']['start_id'] + exp['trainer']['num_trainers']
        ))
        
        self.trainer_spawner.spawn_all(
            trainer_ids,
            alpha=exp['trainer']['dataset']['alpha'],
            availability_mode=exp['trainer']['availability']['mode'],
            trainer_main_path=self.example_dir / 'trainer' / 'pytorch' / 'main.py'
        )
        
        # Monitor
        print("\n" + "─"*70)
        print("REPRODUCTION RUNNING")
        print("─"*70)
        print(f"\nExperiment directory: {self.current_exp_dir}")
        print(f"Aggregator log:       tail -f {agg_log_file}")
        print(f"Trainers log:         tail -f {trainers_log_file}")
        print(f"\nPress Ctrl+C to terminate...")
        print("─"*70 + "\n")
        
        # Wait for completion
        self.trainer_spawner.wait_all()
        
        print("\n✓ Reproduction completed successfully")
    
    def _check_git_state(self, git_info: Dict):
        """Check if current git state matches config."""
        try:
            current_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            current_branch = subprocess.check_output(
                ['git', 'branch', '--show-current'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            print(f"  Original:  {git_info['commit'][:7]} ({git_info['branch']})")
            print(f"  Current:   {current_commit[:7]} ({current_branch})")
            
            if current_commit != git_info['commit']:
                print(f"  ⚠️  Warning: Git commit mismatch")
            elif current_branch != git_info['branch']:
                print(f"  ⚠️  Warning: Different branch")
            else:
                print(f"  ✓ Git state matches")
        except Exception as e:
            print(f"  ⚠️  Could not check git state: {e}")
    
    def _verify_metadata(self, metadata_refs: Dict):
        """Verify all metadata files exist."""
        # Check trainer registry
        registry_path = self.example_dir / metadata_refs['trainer_registry']
        if not registry_path.exists():
            raise FileNotFoundError(f"Trainer registry not found: {registry_path}")
        print(f"  ✓ Trainer registry: {registry_path.name}")
        
        # Check dataset split
        split_key = metadata_refs['dataset_split_key']
        split_path = self.metadata_dir / 'dataset_splits' / f"{split_key}.yaml"
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        print(f"  ✓ Dataset split: {split_key}")
        
        # Check availability trace
        trace_key = metadata_refs['availability_trace_key']
        if trace_key.startswith('syn_'):
            trace_path = self.metadata_dir / 'availability_traces' / 'synthetic_traces.yaml'
        else:
            trace_path = self.metadata_dir / 'availability_traces' / 'mobiperf_traces.yaml'
        
        if not trace_path.exists():
            raise FileNotFoundError(f"Availability traces not found: {trace_path}")
        print(f"  ✓ Availability trace: {trace_key}")
    
    def _create_experiment_directory(self, config: Dict) -> Path:
        """Create experiment directory for reproduction."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config['experiment']['name']
        exp_dir = self.example_dir / 'experiments' / f"reproduce_{timestamp}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the execution config being reproduced
        with open(exp_dir / 'reproduced_from.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return exp_dir
    
    def _create_log_prefix(self, config: Dict) -> str:
        """Create log filename prefix from config."""
        timestamp_str = datetime.now().strftime("%d_%m_%y_%H_%M")
        exp = config['experiment']
        
        alpha = exp['trainer']['dataset']['alpha']
        alpha_str = f"alpha{alpha}".replace(".", "p")
        
        avail = exp['trainer']['availability']['mode'].replace("_", "")
        
        selector = config['aggregator']['selector']
        tracking = config['aggregator']['tracking_mode']
        num_trainers = exp['trainer']['num_trainers']
        
        return f"{timestamp_str}_{selector}_n{num_trainers}_{tracking}_{alpha_str}_{avail}"
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C."""
        print("\n\n" + "="*70)
        print("TERMINATION SIGNAL RECEIVED")
        print("="*70)
        self._cleanup()
        sys.exit(130)
    
    def _cleanup(self):
        """Cleanup processes."""
        print("\nCleaning up...")
        
        if self.trainer_spawner:
            self.trainer_spawner.terminate_all()
        
        if self.aggregator_spawner:
            print("Terminating aggregator...")
            self.aggregator_spawner.terminate()
        
        print("✓ Cleanup complete")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reproduce experiments from execution configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce from execution config
  python3 launch/reproduce.py experiments/run_20260203_143000/execution_config.yaml
  
  # With git checkout
  git checkout <commit>
  python3 launch/reproduce.py experiments/run_*/execution_config.yaml
"""
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to execution_config.yaml'
    )
    
    args = parser.parse_args()
    
    # Verify config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Get example directory
    example_dir = Path(__file__).parent.parent
    
    # Create reproducer and run
    reproducer = ExperimentReproducer(example_dir)
    
    try:
        reproducer.reproduce_from_config(args.config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n✗ Reproduction failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
