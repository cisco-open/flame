#!/usr/bin/env python3
"""
Automated metadata extraction from existing trainer configs.
Handles JSON parsing, deduplication, and YAML generation.

Phase 1.1 of Configuration Migration
"""
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import List


class MetadataExtractor:
    def __init__(self, trainer_config_dirs: List[Path], output_dir: Path):
        self.config_dirs = trainer_config_dirs
        self.output_dir = output_dir
        self.trainer_registry = {}
        self.dataset_splits = defaultdict(dict)
        self.mobiperf_traces = {}
        self.synthetic_traces = {}

    def extract_all(self):
        """Run full extraction pipeline."""
        print("=" * 70)
        print("PHASE 1.1: AUTOMATED METADATA EXTRACTION")
        print("=" * 70)

        print("\n[1/5] Scanning trainer configs...")
        self._scan_all_configs()

        print("\n[2/5] Building trainer registry...")
        self._build_trainer_registry()

        print("\n[3/5] Extracting dataset splits...")
        self._extract_dataset_splits()

        print("\n[4/5] Extracting availability traces...")
        self._extract_traces()

        print("\n[5/5] Writing YAML files...")
        self._write_yaml_files()

        print("\n" + "=" * 70)
        print("✓ EXTRACTION COMPLETE")
        print("=" * 70)

    def _scan_all_configs(self):
        """Scan all config directories and load JSON files."""
        self.all_configs = {}
        total_files = 0

        for config_dir in self.config_dirs:
            if not config_dir.exists():
                print(f"  ⚠ Warning: {config_dir.name} does not exist, skipping")
                continue

            dir_name = config_dir.name
            self.all_configs[dir_name] = {}

            # Load all trainer_N.json files
            config_files = sorted(config_dir.glob("trainer_*.json"))
            for config_file in config_files:
                # Skip test files (trainer_X_test.json)
                if "_test" in config_file.stem:
                    continue

                trainer_num = int(config_file.stem.split("_")[1])
                with open(config_file) as f:
                    self.all_configs[dir_name][trainer_num] = json.load(f)
                total_files += 1

        print(
            f"  ✓ Loaded {total_files} config files from {len(self.all_configs)} directories:"
        )
        for dir_name, configs in sorted(self.all_configs.items()):
            print(f"    • {dir_name}: {len(configs)} trainers")

    def _build_trainer_registry(self):
        """Extract trainer-specific intrinsic properties."""
        # Use first config directory as reference (properties should be consistent)
        reference_dir = list(self.all_configs.keys())[0]
        reference_configs = self.all_configs[reference_dir]

        print(f"  Using reference: {reference_dir}")

        for trainer_num, config in sorted(reference_configs.items()):
            trainer_key = f"trainer_{trainer_num:03d}"

            # Extract core properties
            training_delay = float(config["hyperparameters"]["training_delay_s"])

            # Infer speed class from delay
            if training_delay <= 4.0:
                speed_class = "fast"
            elif training_delay <= 8.0:
                speed_class = "medium"
            elif training_delay <= 12.0:
                speed_class = "slow"
            else:
                speed_class = "very_slow"

            self.trainer_registry[trainer_key] = {
                "trainer_id": trainer_num,
                "task_id": config["taskid"],
                "training_delay_s": training_delay,
                "speed_class": speed_class,
            }

        print(f"  ✓ Built registry for {len(self.trainer_registry)} trainers")

        # Show distribution
        speed_dist = defaultdict(int)
        for trainer_meta in self.trainer_registry.values():
            speed_dist[trainer_meta["speed_class"]] += 1
        print(f"    Speed distribution: {dict(speed_dist)}")

    def _extract_dataset_splits(self):
        """Extract dataset index assignments per trainer per alpha."""
        for dir_name, configs in self.all_configs.items():
            # Parse directory name: config_dir<alpha>_num<N>_<rest>
            # e.g., config_dir0.1_num300_traceFail_6d_3state_oort
            if not dir_name.startswith("config_dir"):
                print(f"  ⚠ Skipping {dir_name} (unexpected format)")
                continue

            # Extract alpha from 'config_dir<alpha>_num...' format
            # Find the part between 'config_dir' and '_num'
            try:
                start_idx = dir_name.index("config_dir") + len("config_dir")
                end_idx = dir_name.index("_num")
                alpha_str = dir_name[start_idx:end_idx]
                alpha = float(alpha_str)
            except (ValueError, IndexError) as e:
                print(f"  ⚠ Warning: Could not parse alpha from {dir_name}: {e}")
                continue

            split_key = f"cifar10_alpha{alpha}_n300"

            for trainer_num, config in configs.items():
                trainer_key = f"trainer_{trainer_num:03d}"
                indices = config["hyperparameters"]["trainer_indices_list"]

                if split_key not in self.dataset_splits:
                    self.dataset_splits[split_key] = {
                        "dataset_name": "cifar10",
                        "num_trainers": 300,
                        "dirichlet_alpha": alpha,
                        "total_samples": 50000,
                        "trainer_data_splits": {},
                    }

                self.dataset_splits[split_key]["trainer_data_splits"][
                    trainer_key
                ] = indices

        print(f"  ✓ Extracted {len(self.dataset_splits)} dataset splits:")
        for split_name, split_data in sorted(self.dataset_splits.items()):
            num_trainers = len(split_data["trainer_data_splits"])
            total_samples = sum(
                len(indices) for indices in split_data["trainer_data_splits"].values()
            )
            print(
                f"    • {split_name}: {num_trainers} trainers, {total_samples} total samples"
            )

    def _extract_traces(self):
        """Extract availability traces (both mobiperf and synthetic)."""
        # Use first config directory as reference
        reference_dir = list(self.all_configs.keys())[0]
        reference_configs = self.all_configs[reference_dir]

        # Extract synthetic traces (uniform across all trainers)
        # Use the first available trainer config
        first_trainer_num = sorted(list(reference_configs.keys()))[0]
        first_config = reference_configs[first_trainer_num]
        print(f"  Using trainer {first_trainer_num} from {reference_dir} as reference")

        hp = first_config["hyperparameters"]

        # Debug: Check what keys are available
        avl_keys = [k for k in hp.keys() if k.startswith("avl_events")]
        print(f"  Found {len(avl_keys)} availability event keys")

        # Helper function to safely evaluate trace strings and convert tuples to lists
        def safe_eval_trace(trace_str):
            if isinstance(trace_str, str):
                result = eval(trace_str)
                # Convert list of tuples to list of lists for YAML compatibility
                if isinstance(result, list) and result and isinstance(result[0], tuple):
                    return [list(item) for item in result]
                return result
            return trace_str

        self.synthetic_traces = {
            "description": "Synthetic availability traces - uniform across all trainers",
            "traces": {
                "syn_0": {
                    "description": "Always available (0% unavailability)",
                    "pattern": safe_eval_trace(hp["avl_events_syn_0"]),
                },
                "syn_20": {
                    "description": "20% unavailability",
                    "pattern": safe_eval_trace(hp["avl_events_syn_20"]),
                },
                "syn_50": {
                    "description": "50% unavailability",
                    "pattern": safe_eval_trace(hp["avl_events_syn_50"]),
                },
            },
        }

        print(f"  ✓ Extracted {len(self.synthetic_traces['traces'])} synthetic traces:")
        for trace_name, trace_data in self.synthetic_traces["traces"].items():
            num_events = len(trace_data["pattern"])
            print(f"    • {trace_name}: {num_events} events")

        # Extract mobiperf traces (per-trainer)
        self.mobiperf_traces = {
            "description": "Real-world MobiPerf availability traces - per device",
            "traces": {},
        }

        for trainer_num, config in reference_configs.items():
            trainer_key = f"trainer_{trainer_num:03d}"
            device_id = f"device_{trainer_num:03d}"
            hp = config["hyperparameters"]

            self.mobiperf_traces["traces"][device_id] = {
                "trainer": trainer_key,
                "states_2st": safe_eval_trace(hp["avl_events_mobiperf_2st"]),
                "states_3st_50": safe_eval_trace(hp["avl_events_mobiperf_3st_50"]),
                "states_3st_75": safe_eval_trace(hp["avl_events_mobiperf_3st_75"]),
            }

            # Link device to trainer in registry
            self.trainer_registry[trainer_key]["mobiperf_device_id"] = device_id

        print(
            f"  ✓ Extracted {len(self.mobiperf_traces['traces'])} mobiperf device traces"
        )
        print(
            f"    (Each device has 3 trace variants: 2-state, 3-state-50%, 3-state-75%)"
        )

    def _write_yaml_files(self):
        """Write all metadata to YAML files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write trainer registry
        registry_file = self.output_dir / "trainer_registry.yaml"
        with open(registry_file, "w") as f:
            yaml.dump(
                {
                    "description": "Global trainer registry with intrinsic properties",
                    "num_trainers": len(self.trainer_registry),
                    "trainers": self.trainer_registry,
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"  ✓ Wrote {registry_file}")

        # Write dataset splits
        splits_dir = self.output_dir / "dataset_splits"
        splits_dir.mkdir(exist_ok=True)
        for split_name, split_data in sorted(self.dataset_splits.items()):
            split_file = splits_dir / f"{split_name}.yaml"
            with open(split_file, "w") as f:
                yaml.dump(split_data, f, default_flow_style=False)
            print(f"  ✓ Wrote {split_file}")

        # Write availability traces
        traces_dir = self.output_dir / "availability_traces"
        traces_dir.mkdir(exist_ok=True)

        synthetic_file = traces_dir / "synthetic_traces.yaml"
        with open(synthetic_file, "w") as f:
            yaml.dump(self.synthetic_traces, f, default_flow_style=False)
        print(f"  ✓ Wrote {synthetic_file}")

        mobiperf_file = traces_dir / "mobiperf_traces.yaml"
        with open(mobiperf_file, "w") as f:
            yaml.dump(self.mobiperf_traces, f, default_flow_style=False)
        print(f"  ✓ Wrote {mobiperf_file}")


# Main execution
if __name__ == "__main__":
    # Define config directories to process
    base_dir = Path(__file__).parent.parent / "trainer"
    config_dirs = [
        base_dir / "config_dir0.1_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir1_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir10_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir100_num300_traceFail_6d_3state_oort",
    ]

    output_dir = Path(__file__).parent.parent / "metadata"

    print(f"\nBase directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nProcessing {len(config_dirs)} config directories...\n")

    extractor = MetadataExtractor(config_dirs, output_dir)
    extractor.extract_all()

    print(f"\n{'─'*70}")
    print("Next step: Run validation script to verify extraction")
    print("Command: python scripts/validate_metadata.py")
    print("─" * 70)
