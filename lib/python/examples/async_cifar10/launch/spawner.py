#!/usr/bin/env python3
"""
Programmatic trainer spawning system for async_cifar10 experiments.

Phase 2 of Configuration Migration
Loads metadata and generates trainer configs at runtime.
"""
import yaml
import json
import subprocess
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys


class MetadataLoader:
    """Loads and caches experiment metadata."""
    
    def __init__(self, metadata_dir: Path):
        # Ensure metadata_dir is a Path object
        self.metadata_dir = Path(metadata_dir) if not isinstance(metadata_dir, Path) else metadata_dir
        self._load_all()
    
    def _load_all(self):
        """Load all metadata files."""
        # Load trainer registry
        with open(self.metadata_dir / 'trainer_registry.yaml') as f:
            registry_data = yaml.safe_load(f)
            self.trainer_registry = registry_data['trainers']
        
        # Load dataset splits
        self.dataset_splits = {}
        splits_dir = self.metadata_dir / 'dataset_splits'
        for split_file in splits_dir.glob('*.yaml'):
            with open(split_file) as f:
                split_data = yaml.safe_load(f)
                self.dataset_splits[split_file.stem] = split_data
        
        # Load availability traces
        traces_dir = self.metadata_dir / 'availability_traces'
        with open(traces_dir / 'synthetic_traces.yaml') as f:
            self.synthetic_traces = yaml.safe_load(f)
        with open(traces_dir / 'mobiperf_traces.yaml') as f:
            self.mobiperf_traces = yaml.safe_load(f)
    
    def get_trainer_metadata(self, trainer_id: int) -> Dict:
        """Get metadata for a specific trainer."""
        trainer_key = f"trainer_{trainer_id:03d}"
        return self.trainer_registry.get(trainer_key)
    
    def get_dataset_split(self, alpha: float, trainer_id: int) -> List[int]:
        """Get dataset indices for a trainer given Dirichlet alpha."""
        split_key = f"cifar10_alpha{alpha}_n300"
        trainer_key = f"trainer_{trainer_id:03d}"
        return self.dataset_splits[split_key]['trainer_data_splits'][trainer_key]
    
    def get_synthetic_trace(self, trace_name: str) -> List:
        """Get synthetic availability trace."""
        return self.synthetic_traces['traces'][trace_name]['pattern']
    
    def get_mobiperf_trace(self, trainer_id: int, variant: str = '2st') -> List:
        """Get mobiperf trace for a trainer."""
        device_id = f"device_{trainer_id:03d}"
        return self.mobiperf_traces['traces'][device_id][f'states_{variant}']


class ConfigGenerator:
    """Generates runtime trainer configurations."""
    
    def __init__(self, metadata_loader: MetadataLoader, base_config_path: Path):
        self.metadata = metadata_loader
        self.base_config = self._load_base_config(base_config_path)
    
    def _load_base_config(self, path: Path) -> Dict:
        """Load base configuration template."""
        with open(path) as f:
            return yaml.safe_load(f)
    
    def generate_trainer_config(
        self,
        trainer_id: int,
        alpha: float,
        availability_mode: str = 'mobiperf_2st',
        **overrides
    ) -> Dict:
        """
        Generate complete trainer configuration.
        
        Args:
            trainer_id: Trainer ID (1-300)
            alpha: Dirichlet alpha for dataset split
            availability_mode: One of ['mobiperf_2st', 'mobiperf_3st_50', 
                               'mobiperf_3st_75', 'syn_0', 'syn_20', 'syn_50']
            **overrides: Additional config overrides
        
        Returns:
            Complete trainer configuration dict
        """
        # Start with base config
        config = self.base_config.copy()
        
        # Get trainer-specific metadata
        trainer_meta = self.metadata.get_trainer_metadata(trainer_id)
        
        # Update taskid
        config['taskid'] = trainer_meta['task_id']
        
        # Get dataset split
        dataset_indices = self.metadata.get_dataset_split(alpha, trainer_id)
        
        # Update hyperparameters
        if 'hyperparameters' not in config:
            config['hyperparameters'] = {}
        
        config['hyperparameters']['trainer_indices_list'] = dataset_indices
        config['hyperparameters']['training_delay_s'] = trainer_meta['training_delay_s']
        
        # Add availability traces
        if availability_mode.startswith('mobiperf'):
            variant = availability_mode.replace('mobiperf_', '')
            config['hyperparameters'][f'avl_events_mobiperf_{variant}'] = \
                self.metadata.get_mobiperf_trace(trainer_id, variant)
        elif availability_mode.startswith('syn_'):
            trace_name = availability_mode
            config['hyperparameters'][f'avl_events_{trace_name}'] = \
                self.metadata.get_synthetic_trace(trace_name)
        
        # Add all synthetic traces (for flexibility)
        for trace_name in ['syn_0', 'syn_20', 'syn_50']:
            config['hyperparameters'][f'avl_events_{trace_name}'] = \
                self.metadata.get_synthetic_trace(trace_name)
        
        # Add all mobiperf traces
        for variant in ['2st', '3st_50', '3st_75']:
            config['hyperparameters'][f'avl_events_mobiperf_{variant}'] = \
                self.metadata.get_mobiperf_trace(trainer_id, variant)
        
        # Apply any overrides
        for key, value in overrides.items():
            if '.' in key:
                # Nested key like 'hyperparameters.batchSize'
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value
        
        return config


class TrainerSpawner:
    """Spawns trainer processes with GPU affinity."""
    
    def __init__(
        self,
        config_generator: ConfigGenerator,
        num_gpus: int = 8,
        sleep_between_spawns: float = 1.0,
        log_file: Optional[Path] = None
    ):
        self.config_gen = config_generator
        self.num_gpus = num_gpus
        self.sleep_between_spawns = sleep_between_spawns
        self.log_file = log_file
        self.processes = []
        self._log_handle = None
        
        # Open combined log file if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_file, 'w', buffering=1)  # Line buffered
    
    def spawn_trainer(
        self,
        trainer_id: int,
        alpha: float,
        availability_mode: str,
        trainer_main_path: Path,
        **config_overrides
    ) -> subprocess.Popen:
        """
        Spawn a single trainer process.
        
        Args:
            trainer_id: Trainer ID
            alpha: Dirichlet alpha
            availability_mode: Availability trace mode
            trainer_main_path: Path to trainer main.py
            **config_overrides: Additional config overrides
        
        Returns:
            subprocess.Popen object
        """
        # Generate config
        config = self.config_gen.generate_trainer_config(
            trainer_id, alpha, availability_mode, **config_overrides
        )
        
        # Serialize config to JSON string
        config_json = json.dumps(config)
        
        # Determine GPU
        gpu_id = (trainer_id - 1) % self.num_gpus
        
        # Build command
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(trainer_main_path),
            '--config-json', config_json
        ]
        
        # Determine stdout/stderr handling
        if self._log_handle:
            # Write to combined log file with trainer ID prefix
            stdout_target = self._log_handle
            stderr_target = subprocess.STDOUT
        else:
            # Default: pipe
            stdout_target = subprocess.PIPE
            stderr_target = subprocess.PIPE
        
        # Spawn process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True
        )
        
        self.processes.append({
            'trainer_id': trainer_id,
            'gpu_id': gpu_id,
            'process': process
        })
        
        print(f"  Spawned trainer {trainer_id} on GPU {gpu_id} (PID: {process.pid})")
        
        return process
    
    def spawn_all(
        self,
        trainer_ids: List[int],
        alpha: float,
        availability_mode: str,
        trainer_main_path: Path,
        **config_overrides
    ):
        """
        Spawn multiple trainers sequentially.
        
        Args:
            trainer_ids: List of trainer IDs to spawn
            alpha: Dirichlet alpha
            availability_mode: Availability trace mode
            trainer_main_path: Path to trainer main.py
            **config_overrides: Additional config overrides
        """
        print(f"\nSpawning {len(trainer_ids)} trainers...")
        print(f"  Alpha: {alpha}")
        print(f"  Availability: {availability_mode}")
        print(f"  GPUs: {self.num_gpus}")
        print()
        
        for trainer_id in trainer_ids:
            self.spawn_trainer(
                trainer_id, alpha, availability_mode,
                trainer_main_path, **config_overrides
            )
            time.sleep(self.sleep_between_spawns)
        
        print(f"\n✓ Spawned {len(self.processes)} trainers")
    
    def wait_all(self):
        """Wait for all trainer processes to complete."""
        for proc_info in self.processes:
            proc_info['process'].wait()
    
    def terminate_all(self):
        """Terminate all trainer processes."""
        print(f"\nTerminating {len(self.processes)} trainers...")
        for proc_info in self.processes:
            try:
                proc_info['process'].terminate()
            except Exception:
                # Process may have already terminated; ignore
                pass
        
        # Wait a bit, then kill if needed
        time.sleep(2)
        for proc_info in self.processes:
            try:
                if proc_info['process'].poll() is None:
                    proc_info['process'].kill()
            except Exception:
                # Process may have already terminated; ignore
                pass
        
        # Close log file
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Spawn trainers programmatically')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet alpha for dataset split')
    parser.add_argument('--availability', type=str, default='mobiperf_2st',
                        choices=['mobiperf_2st', 'mobiperf_3st_50', 'mobiperf_3st_75',
                                 'syn_0', 'syn_20', 'syn_50'],
                        help='Availability trace mode')
    parser.add_argument('--num-trainers', type=int, default=300,
                        help='Number of trainers to spawn')
    parser.add_argument('--start-id', type=int, default=1,
                        help='Starting trainer ID')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs for load balancing')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: spawn only 5 trainers')
    
    args = parser.parse_args()
    
    # Setup paths
    example_dir = Path(__file__).parent.parent
    metadata_dir = example_dir / 'metadata'
    base_config_path = example_dir / 'configs' / 'trainer_base.yaml'
    trainer_main_path = example_dir / 'trainer' / 'pytorch' / 'main.py'
    
    # Initialize components
    print("="*70)
    print("PROGRAMMATIC TRAINER SPAWNER (Phase 2)")
    print("="*70)
    
    print("\n[1/3] Loading metadata...")
    metadata_loader = MetadataLoader(metadata_dir)
    print(f"  ✓ Loaded {len(metadata_loader.trainer_registry)} trainers")
    print(f"  ✓ Loaded {len(metadata_loader.dataset_splits)} dataset splits")
    
    print("\n[2/3] Initializing config generator...")
    config_gen = ConfigGenerator(metadata_loader, base_config_path)
    print(f"  ✓ Loaded base config from {base_config_path.name}")
    
    print("\n[3/3] Spawning trainers...")
    spawner = TrainerSpawner(config_gen, num_gpus=args.num_gpus)
    
    # Determine trainer IDs to spawn
    if args.test:
        trainer_ids = list(range(args.start_id, args.start_id + 5))
        print("  TEST MODE: Spawning only 5 trainers")
    else:
        trainer_ids = list(range(args.start_id, args.start_id + args.num_trainers))
    
    try:
        spawner.spawn_all(
            trainer_ids,
            alpha=args.alpha,
            availability_mode=args.availability,
            trainer_main_path=trainer_main_path
        )
        
        print("\nTrainers running. Press Ctrl+C to terminate...")
        spawner.wait_all()
        
    except KeyboardInterrupt:
        print("\n\nCaught Ctrl+C, terminating trainers...")
        spawner.terminate_all()
        print("✓ All trainers terminated")
