#!/usr/bin/env python3
"""Programmatic trainer spawning. Loads metadata and generates configs at runtime."""
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
        self.metadata_dir = (
            Path(metadata_dir) if not isinstance(metadata_dir, Path) else metadata_dir
        )
        self._load_all()

    def _load_all(self):
        """Load registry + dataset splits + availability traces, lenient on missing trace files."""
        with open(self.metadata_dir / "trainer_registry.yaml") as f:
            self.trainer_registry = yaml.safe_load(f)["trainers"]

        self.dataset_splits = {}
        splits_dir = self.metadata_dir / "dataset_splits"
        if splits_dir.is_dir():
            for split_file in splits_dir.glob("*.yaml"):
                with open(split_file) as f:
                    self.dataset_splits[split_file.stem] = yaml.safe_load(f)

        traces_dir = self.metadata_dir / "availability_traces"
        self.synthetic_traces = {"traces": {}}
        self.mobiperf_traces = {"traces": {}}
        if (traces_dir / "synthetic_traces.yaml").is_file():
            with open(traces_dir / "synthetic_traces.yaml") as f:
                self.synthetic_traces = yaml.safe_load(f)
        if (traces_dir / "mobiperf_traces.yaml").is_file():
            with open(traces_dir / "mobiperf_traces.yaml") as f:
                self.mobiperf_traces = yaml.safe_load(f)

    def get_trainer_metadata(self, trainer_id: int) -> Dict:
        """Get metadata for a specific trainer."""
        trainer_key = f"trainer_{trainer_id:03d}"
        return self.trainer_registry.get(trainer_key)

    def get_dataset_split(
        self,
        alpha: float,
        trainer_id: int,
        dataset_name: str = "cifar10",
        num_trainers: int = 300,
    ) -> List[int]:
        """Look up indices for trainer in <dataset>_alpha<alpha>_n<num_trainers>."""
        split_key = f"{dataset_name}_alpha{alpha}_n{num_trainers}"
        trainer_key = f"trainer_{trainer_id:03d}"
        return self.dataset_splits[split_key]["trainer_data_splits"][trainer_key]

    def get_synthetic_trace(
        self, trace_name: str, trainer_id: Optional[int] = None
    ) -> List:
        """Get synthetic availability trace.

        trainer_id=None returns the shared `pattern` entry. Pass trainer_id to
        resolve that trainer's own `per_trainer` entry via the same resolver
        (flame.availability.trace.load_trace) the aggregator already uses.
        """
        if trainer_id is None:
            return self.synthetic_traces["traces"][trace_name]["pattern"]
        from flame.availability.trace import load_trace

        trainer_key = f"trainer_{trainer_id:03d}"
        trace_dict = load_trace(
            trace_name, trainer_key, base_dir=str(self.metadata_dir / "availability_traces")
        )
        return [[ts, state] for ts, state in trace_dict.items()]

    def get_mobiperf_trace(self, trainer_id: int, variant: str = "2st") -> List:
        """Get mobiperf trace for a trainer."""
        device_id = f"device_{trainer_id:03d}"
        return self.mobiperf_traces["traces"][device_id][f"states_{variant}"]


class ConfigGenerator:
    """Generates runtime trainer configurations."""

    def __init__(self, metadata_loader: MetadataLoader, base_config_path: Path):
        self.metadata = metadata_loader
        self.base_config = self._load_base_config(base_config_path)
        self.baseline_overrides: Dict = {}

    def _load_base_config(self, path: Path) -> Dict:
        """Load base configuration template."""
        with open(path) as f:
            return yaml.safe_load(f)

    def set_baseline_overrides(self, overrides: Dict) -> None:
        """Dict of fields deep-merged into every generated trainer config
        BEFORE per-trainer values and BEFORE flat-key kwargs overrides.
        """
        from flame.launch.baselines import deep_merge
        self._deep_merge = deep_merge
        self.baseline_overrides = overrides or {}

    def generate_trainer_config(
        self,
        trainer_id: int,
        alpha: float,
        availability_mode: str = "mobiperf_2st",
        dataset_name: str = "cifar10",
        num_trainers: int = 300,
        skip_index_splits: bool = False,
        **overrides,
    ) -> Dict:
        """Build a full trainer config dict from base + metadata + overrides.

        Layer order (later layers win):
          1. trainer_base.yaml (per-example static template)
          2. baseline_overrides (set via set_baseline_overrides, deep-merged)
          3. per-trainer values from shared metadata (taskid, indices, traces)
          4. **overrides dotted-key kwargs (job.id, hyperparameters.X)

        skip_index_splits: True for path-style datasets (e.g. H5 file paths)
        that have no _metadata/dataset_splits/<name>_alpha<a>_n<N>.yaml
        index-list file -- skips the get_dataset_split() lookup and the
        trainer_indices_list injection entirely, instead of KeyError-ing on a
        split file that will never exist for this dataset.
        """
        from copy import deepcopy

        from flame.launch.baselines import deep_merge

        config = deepcopy(self.base_config)
        if self.baseline_overrides:
            config = deep_merge(config, self.baseline_overrides)

        trainer_meta = self.metadata.get_trainer_metadata(trainer_id)
        config["taskid"] = trainer_meta["task_id"]

        # Update hyperparameters
        if "hyperparameters" not in config:
            config["hyperparameters"] = {}

        if not skip_index_splits:
            dataset_indices = self.metadata.get_dataset_split(
                alpha, trainer_id, dataset_name, num_trainers
            )
            config["hyperparameters"]["trainer_indices_list"] = dataset_indices

        config["hyperparameters"]["training_delay_s"] = trainer_meta["training_delay_s"]
        
        # Set training_delay_enabled from overrides (default True)
        training_delay_enabled = overrides.get("hyperparameters.training_delay_enabled", "True")
        config["hyperparameters"]["training_delay_enabled"] = training_delay_enabled

        # Add availability traces
        if availability_mode.startswith("mobiperf"):
            variant = availability_mode.replace("mobiperf_", "")
            config["hyperparameters"][f"avl_events_mobiperf_{variant}"] = (
                self.metadata.get_mobiperf_trace(trainer_id, variant)
            )
        elif availability_mode.startswith("syn_"):
            trace_name = availability_mode
            config["hyperparameters"][f"avl_events_{trace_name}"] = (
                self.metadata.get_synthetic_trace(trace_name, trainer_id)
            )

        # Add all synthetic traces (for flexibility)
        for trace_name in ["syn_0", "syn_20", "syn_50"]:
            config["hyperparameters"][f"avl_events_{trace_name}"] = (
                self.metadata.get_synthetic_trace(trace_name, trainer_id)
            )

        # Add all mobiperf traces
        for variant in ["2st", "3st_50", "3st_75"]:
            config["hyperparameters"][f"avl_events_mobiperf_{variant}"] = (
                self.metadata.get_mobiperf_trace(trainer_id, variant)
            )

        # client_notify defaults only when not provided by base/baseline/overrides.
        cn = config["hyperparameters"].setdefault("client_notify", {})
        cn.setdefault("enabled", "False")
        cn.setdefault("trace", "syn_0")

        # Apply any overrides
        for key, value in overrides.items():
            if "." in key:
                # Nested key like 'hyperparameters.batchSize' or 'job.id'
                parts = key.split(".")
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
        log_file: Optional[Path] = None,
        time_mode: str = "simulated",
        battery_threshold: int = 50,
        cpu_pinning: bool = True,
        reserved_cores: Optional[set] = None,
    ):
        self.config_gen = config_generator
        self.num_gpus = num_gpus
        self.reserved_cores = {int(c) for c in reserved_cores} if reserved_cores else set()
        self.sleep_between_spawns = sleep_between_spawns
        self.log_file = log_file
        # CLI-only trainer knobs: read from argv, not config JSON.
        self.time_mode = time_mode
        self.battery_threshold = battery_threshold
        self.cpu_pinning = cpu_pinning
        self.processes = []
        self._log_handle = None

        # Discover usable CPU cores (respects cgroup/Slurm affinity).
        self._usable_cores: List[int] = []
        if self.cpu_pinning:
            try:
                self._usable_cores = sorted(os.sched_getaffinity(0))
                # Exclude cores reserved for the aggregator so trainers don't
                # time-slice the (bottlenecked) aggregator process.
                if self.reserved_cores:
                    self._usable_cores = [c for c in self._usable_cores
                                          if c not in self.reserved_cores]
                _resv = f", {len(self.reserved_cores)} reserved for aggregator" if self.reserved_cores else ""
                print(f"  CPU pinning ON: {len(self._usable_cores)} usable cores for trainers{_resv}: {self._usable_cores[:8]}{'...' if len(self._usable_cores) > 8 else ''}")
            except AttributeError:
                print("  CPU pinning requested but os.sched_getaffinity unavailable (non-Linux); pinning disabled.")
                self.cpu_pinning = False

        # Open combined log file if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_file, "w", buffering=1)  # Line buffered

    def spawn_trainer(
        self,
        trainer_id: int,
        alpha: float,
        availability_mode: str,
        trainer_main_path: Path,
        skip_index_splits: bool = False,
        dataset_name: str = "cifar10",
        num_trainers: int = 300,
        **config_overrides,
    ) -> subprocess.Popen:
        """
        Spawn a single trainer process.

        Args:
            trainer_id: Trainer ID
            alpha: Dirichlet alpha
            availability_mode: Availability trace mode
            trainer_main_path: Path to trainer main.py
            skip_index_splits: True for path-style datasets with no
                _metadata/dataset_splits/ index-list file (see
                ConfigGenerator.generate_trainer_config).
            dataset_name: Forwarded to generate_trainer_config()'s dataset
                split lookup. Caller MUST pass the experiment's real
                trainer.dataset.name -- the "cifar10" default here exists
                only so direct/manual callers don't need to specify it for
                the common case, not as a silent fallback for the launcher.
            num_trainers: Forwarded to generate_trainer_config()'s dataset
                split lookup (selects which <dataset>_alpha<a>_n<N>.yaml
                split file to read). Caller MUST pass the experiment's real
                trainer.num_trainers -- the 300 default here is the same
                "don't require it for manual calls" exception as
                dataset_name, not a safe fallback. Passing the wrong value
                silently loads a *different*, structurally valid split file
                (e.g. n=300's instead of n=48's) rather than erroring, which
                is exactly the bug this parameter was added to close (no
                caller threaded it through before).
            **config_overrides: Additional config overrides

        Returns:
            subprocess.Popen object
        """
        # Generate config
        config = self.config_gen.generate_trainer_config(
            trainer_id, alpha, availability_mode,
            dataset_name=dataset_name, num_trainers=num_trainers,
            skip_index_splits=skip_index_splits, **config_overrides
        )

        # Serialize config to JSON string
        config_json = json.dumps(config)

        # Determine GPU
        gpu_id = (trainer_id - 1) % self.num_gpus

        # Determine CPU core (round-robin across usable cores when pinning is on)
        cpu_core: Optional[int] = None
        preexec_fn = None
        if self.cpu_pinning and self._usable_cores:
            cpu_core = self._usable_cores[(trainer_id - 1) % len(self._usable_cores)]
            _core_set = {cpu_core}
            preexec_fn = lambda c=_core_set: os.sched_setaffinity(0, c)

        # Build command
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if self.cpu_pinning and cpu_core is not None:
            # Prevent thread oversubscription when pinned to one core.
            for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS"):
                env[_var] = "1"

        cmd = [
            sys.executable,  # Use same Python interpreter
            str(trainer_main_path),
            "--config-json",
            config_json,
            "--time_mode",
            str(self.time_mode),
            "--battery_threshold",
            str(self.battery_threshold),
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
            cmd, env=env, stdout=stdout_target, stderr=stderr_target, text=True,
            preexec_fn=preexec_fn,
        )

        self.processes.append(
            {"trainer_id": trainer_id, "gpu_id": gpu_id, "cpu_core": cpu_core, "process": process}
        )

        core_str = f", CPU core {cpu_core}" if cpu_core is not None else ""
        print(f"  Spawned trainer {trainer_id} on GPU {gpu_id}{core_str} (PID: {process.pid})")

        return process

    def spawn_all(
        self,
        trainer_ids: List[int],
        alpha: float,
        availability_mode: str,
        trainer_main_path: Path,
        skip_index_splits: bool = False,
        dataset_name: str = "cifar10",
        num_trainers: int = 300,
        per_trainer_overrides: Optional[Dict[int, Dict]] = None,
        **config_overrides,
    ):
        """
        Spawn multiple trainers sequentially.

        Args:
            trainer_ids: List of trainer IDs to spawn
            alpha: Dirichlet alpha
            availability_mode: Availability trace mode
            trainer_main_path: Path to trainer main.py
            skip_index_splits: True for path-style datasets with no
                _metadata/dataset_splits/ index-list file.
            dataset_name: The experiment's real trainer.dataset.name --
                selects which <dataset>_alpha<a>_n<N>.yaml split file to
                read (see spawn_trainer's docstring for why the "cifar10"
                default here must not be relied on by the launcher).
            num_trainers: The experiment's real trainer.num_trainers --
                selects which <dataset>_alpha<a>_n<N>.yaml split file to
                read. Pass it explicitly rather than inferring
                len(trainer_ids); the split file's <N> means "this dataset
                was partitioned for N trainers total," a property of the
                experiment, not of whichever id range happens to be passed
                here.
            per_trainer_overrides: Optional {trainer_id: {dotted.key: value}}
                dict of overrides that vary per trainer (e.g. a computed
                hyperparameters.client_idx), merged on top of the shared
                **config_overrides for that specific trainer_id. Unlike
                **config_overrides (identical for every trainer),
                per_trainer_overrides lets each trainer's config differ.
            **config_overrides: Additional config overrides shared by all trainers
        """
        print(f"\nSpawning {len(trainer_ids)} trainers...")
        print(f"  Alpha: {alpha}")
        print(f"  Availability: {availability_mode}")
        print(f"  GPUs: {self.num_gpus}")
        print()

        for trainer_id in trainer_ids:
            trainer_overrides = dict(config_overrides)
            if per_trainer_overrides and trainer_id in per_trainer_overrides:
                trainer_overrides.update(per_trainer_overrides[trainer_id])
            self.spawn_trainer(
                trainer_id,
                alpha,
                availability_mode,
                trainer_main_path,
                skip_index_splits=skip_index_splits,
                dataset_name=dataset_name,
                num_trainers=num_trainers,
                **trainer_overrides,
            )
            time.sleep(self.sleep_between_spawns)

        print(f"\n✓ Spawned {len(self.processes)} trainers")
        # Dump trainer→(gpu, core) assignment table
        if self.cpu_pinning and self._usable_cores:
            print(f"\n  Trainer assignments (cpu_pinning=ON, {len(self._usable_cores)} cores):")
            print(f"  {'Trainer':>8}  {'GPU':>4}  {'CPU core':>9}  {'PID':>7}")
            for p in self.processes:
                print(f"  {p['trainer_id']:>8}  {p['gpu_id']:>4}  {str(p.get('cpu_core', 'N/A')):>9}  {p['process'].pid:>7}")

    def wait_all(self, timeout_per_trainer: float = 30.0):
        """Wait for all trainer processes to complete.

        All trainers share a single ``timeout_per_trainer`` second window
        after the aggregator exits to process the EOT broadcast and
        self-terminate, polled concurrently — not one process at a time, or
        N stragglers would serialize the shutdown into N * timeout_per_trainer
        (e.g. 5 stuck trainers at 30s each adds 2.5 minutes of pure idle
        wait). Trainers blocked in ``await_join`` (waiting for the next task
        from an aggregator that has already left) will never self-exit, so we
        force-terminate whatever is left after the window, then kill anything
        still alive after a short grace period. Without this the runner hangs
        indefinitely and the next sequential experiment never starts.
        """
        deadline = time.time() + timeout_per_trainer
        pending = [p["process"] for p in self.processes]

        while pending and time.time() < deadline:
            pending = [p for p in pending if p.poll() is None]
            if pending:
                time.sleep(0.5)

        if not pending:
            return

        print(
            f"  ({len(pending)} trainer(s) did not exit within "
            f"{timeout_per_trainer:.0f}s; terminating)"
        )
        for proc in pending:
            try:
                proc.terminate()
            except Exception:
                pass

        kill_deadline = time.time() + 5
        while pending and time.time() < kill_deadline:
            pending = [p for p in pending if p.poll() is None]
            if pending:
                time.sleep(0.5)

        for proc in pending:
            try:
                proc.kill()
            except Exception:
                pass

    def terminate_all(self):
        """Terminate all trainer processes."""
        print(f"\nTerminating {len(self.processes)} trainers...")
        for proc_info in self.processes:
            try:
                proc_info["process"].terminate()
            except Exception:
                # Process may have already terminated; ignore
                pass

        # Wait a bit, then kill if needed
        time.sleep(2)
        for proc_info in self.processes:
            try:
                if proc_info["process"].poll() is None:
                    proc_info["process"].kill()
            except Exception:
                # Process may have already terminated; ignore
                pass

        # Close log file
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spawn trainers programmatically")
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Dirichlet alpha for dataset split"
    )
    parser.add_argument(
        "--availability",
        type=str,
        default="mobiperf_2st",
        choices=[
            "mobiperf_2st",
            "mobiperf_3st_50",
            "mobiperf_3st_75",
            "syn_0",
            "syn_20",
            "syn_50",
        ],
        help="Availability trace mode",
    )
    parser.add_argument(
        "--num-trainers", type=int, default=300, help="Number of trainers to spawn"
    )
    parser.add_argument("--start-id", type=int, default=1, help="Starting trainer ID")
    parser.add_argument(
        "--num-gpus", type=int, default=8, help="Number of GPUs for load balancing"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: spawn only 5 trainers"
    )
    parser.add_argument(
        "--cpu_pinning",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Pin each trainer to a single CPU core round-robin (default: on)",
    )

    args = parser.parse_args()

    # Setup paths
    example_dir = Path(__file__).parent.parent
    metadata_dir = example_dir / "metadata"
    base_config_path = example_dir / "configs" / "trainer_base.yaml"
    trainer_main_path = example_dir / "trainer" / "pytorch" / "main.py"

    # Initialize components
    print("=" * 70)
    print("PROGRAMMATIC TRAINER SPAWNER (Phase 2)")
    print("=" * 70)

    print("\n[1/3] Loading metadata...")
    metadata_loader = MetadataLoader(metadata_dir)
    print(f"  ✓ Loaded {len(metadata_loader.trainer_registry)} trainers")
    print(f"  ✓ Loaded {len(metadata_loader.dataset_splits)} dataset splits")

    print("\n[2/3] Initializing config generator...")
    config_gen = ConfigGenerator(metadata_loader, base_config_path)
    print(f"  ✓ Loaded base config from {base_config_path.name}")

    print("\n[3/3] Spawning trainers...")
    spawner = TrainerSpawner(config_gen, num_gpus=args.num_gpus, cpu_pinning=(args.cpu_pinning == "on"))

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
            trainer_main_path=trainer_main_path,
        )

        print("\nTrainers running. Press Ctrl+C to terminate...")
        spawner.wait_all()

    except KeyboardInterrupt:
        print("\n\nCaught Ctrl+C, terminating trainers...")
        spawner.terminate_all()
        print("✓ All trainers terminated")
