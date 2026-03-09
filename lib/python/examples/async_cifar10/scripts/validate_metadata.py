#!/usr/bin/env python3
"""
Comprehensive validation: verify YAML metadata matches original JSON configs.
This is the GATE before proceeding to Phase 2.

Phase 1.2 of Configuration Migration
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List
import sys


class MetadataValidator:
    def __init__(self, metadata_dir: Path, trainer_config_dirs: List[Path]):
        self.metadata_dir = metadata_dir
        self.config_dirs = trainer_config_dirs
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if all pass."""
        print("=" * 70)
        print("PHASE 1.2: METADATA VALIDATION SUITE")
        print("=" * 70)

        # Check that metadata directory exists
        if not self.metadata_dir.exists():
            print(f"\n✗ ERROR: Metadata directory does not exist: {self.metadata_dir}")
            print("  Run extract_metadata.py first!")
            return False

        all_passed = True

        checks = [
            ("Trainer Registry", self._validate_trainer_registry),
            ("Dataset Splits", self._validate_dataset_splits),
            ("Synthetic Traces", self._validate_synthetic_traces),
            ("MobiPerf Traces", self._validate_mobiperf_traces),
            ("Cross-Reference Integrity", self._validate_cross_references),
            ("Completeness Check", self._validate_completeness),
        ]

        for check_name, check_func in checks:
            print(f"\n{'─'*70}")
            print(f"Running: {check_name}")
            print("─" * 70)
            try:
                passed = check_func()

                if passed:
                    print(f"✓ {check_name}: PASSED")
                else:
                    print(f"✗ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"✗ {check_name}: EXCEPTION - {e}")
                self.errors.append(f"{check_name}: {e}")
                all_passed = False

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if all_passed:
            print("\n🎉 ✓ ALL VALIDATIONS PASSED")
            print("\nMetadata is verified correct. Safe to proceed to Phase 2.")
            print("\nNext steps:")
            print("  1. Review generated YAML files in metadata/ directory")
            print("  2. Manual spot-check: Compare a few trainers (JSON vs YAML)")
            print(
                "  3. Git commit: 'Phase 1 complete: Extracted and validated metadata'"
            )
        else:
            print(f"\n✗ VALIDATION FAILED with {len(self.errors)} errors")
            print("\nErrors:")
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
            print("\n⚠ DO NOT PROCEED TO PHASE 2 until all errors are fixed.")

        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                print(f"  ⚠ {warning}")

        return all_passed

    def _load_metadata(self):
        """Load all YAML metadata files."""
        with open(self.metadata_dir / "trainer_registry.yaml") as f:
            self.trainer_registry = yaml.safe_load(f)

        self.dataset_splits = {}
        splits_dir = self.metadata_dir / "dataset_splits"
        for split_file in splits_dir.glob("*.yaml"):
            with open(split_file) as f:
                self.dataset_splits[split_file.stem] = yaml.safe_load(f)

        traces_dir = self.metadata_dir / "availability_traces"
        with open(traces_dir / "synthetic_traces.yaml") as f:
            self.synthetic_traces = yaml.safe_load(f)
        with open(traces_dir / "mobiperf_traces.yaml") as f:
            self.mobiperf_traces = yaml.safe_load(f)

    def _load_original_config(self, config_dir: Path, trainer_num: int) -> Dict:
        """Load original JSON config for comparison."""
        config_file = config_dir / f"trainer_{trainer_num}.json"
        with open(config_file) as f:
            return json.load(f)

    def _validate_trainer_registry(self) -> bool:
        """Validate trainer registry against original configs."""
        self._load_metadata()
        passed = True

        # Check count
        expected_count = 300
        actual_count = self.trainer_registry["num_trainers"]
        print(f"  Checking trainer count: {actual_count} trainers")
        if actual_count != expected_count:
            self.errors.append(
                f"Registry has {actual_count} trainers, expected {expected_count}"
            )
            passed = False

        # Validate each trainer against first config directory
        reference_dir = [d for d in self.config_dirs if d.exists()][0]
        print(f"  Validating against: {reference_dir.name}")

        errors_found = 0
        for trainer_key, trainer_meta in self.trainer_registry["trainers"].items():
            trainer_num = trainer_meta["trainer_id"]

            try:
                orig_config = self._load_original_config(reference_dir, trainer_num)
            except FileNotFoundError:
                self.errors.append(f"{trainer_key}: Original config not found")
                passed = False
                errors_found += 1
                continue

            # Check task_id
            if trainer_meta["task_id"] != orig_config["taskid"]:
                self.errors.append(
                    f"{trainer_key}: task_id mismatch - "
                    f"YAML={trainer_meta['task_id']}, JSON={orig_config['taskid']}"
                )
                passed = False
                errors_found += 1

            # Check training_delay_s
            orig_delay = float(orig_config["hyperparameters"]["training_delay_s"])
            if trainer_meta["training_delay_s"] != orig_delay:
                self.errors.append(
                    f"{trainer_key}: training_delay_s mismatch - "
                    f"YAML={trainer_meta['training_delay_s']}, JSON={orig_delay}"
                )
                passed = False
                errors_found += 1

        if passed:
            print(f"  ✓ All {actual_count} trainers validated successfully")
        else:
            print(f"  ✗ Found {errors_found} errors in trainer registry")

        return passed

    def _validate_dataset_splits(self) -> bool:
        """Validate dataset splits against original configs."""
        passed = True

        print(f"  Validating {len(self.dataset_splits)} dataset splits...")

        for split_name, split_data in sorted(self.dataset_splits.items()):
            alpha = split_data["dirichlet_alpha"]
            print(f"    Checking {split_name} (alpha={alpha})...")

            # Find corresponding config directory
            # Handle both integer (1) and float (1.0) alpha values
            config_dir = None
            for cd in self.config_dirs:
                if not cd.exists():
                    continue
                # Extract alpha from directory name
                if "config_dir" in cd.name:
                    try:
                        start_idx = cd.name.index("config_dir") + len("config_dir")
                        end_idx = cd.name.index("_num")
                        dir_alpha_str = cd.name[start_idx:end_idx]
                        dir_alpha = float(dir_alpha_str)
                        if dir_alpha == alpha:
                            config_dir = cd
                            break
                    except (ValueError, IndexError):
                        continue

            if not config_dir:
                self.errors.append(f"{split_name}: No matching config directory found")
                passed = False
                continue

            # Validate a sample of trainers (all 300 would be too verbose)
            sample_trainers = [1, 50, 100, 150, 200, 250, 300]
            errors_in_split = 0

            for trainer_num in sample_trainers:
                trainer_key = f"trainer_{trainer_num:03d}"
                if trainer_key not in split_data["trainer_data_splits"]:
                    self.errors.append(f"{split_name}: Missing {trainer_key}")
                    errors_in_split += 1
                    continue

                indices = split_data["trainer_data_splits"][trainer_key]

                try:
                    orig_config = self._load_original_config(config_dir, trainer_num)
                    orig_indices = orig_config["hyperparameters"][
                        "trainer_indices_list"
                    ]

                    if indices != orig_indices:
                        self.errors.append(
                            f"{split_name}/{trainer_key}: Indices mismatch - "
                            f"lengths YAML={len(indices)}, JSON={len(orig_indices)}"
                        )
                        errors_in_split += 1
                except FileNotFoundError:
                    self.errors.append(
                        f"{split_name}/{trainer_key}: Original config not found"
                    )
                    errors_in_split += 1

            if errors_in_split == 0:
                num_trainers = len(split_data["trainer_data_splits"])
                print(
                    f"      ✓ {num_trainers} trainers (sampled {len(sample_trainers)})"
                )
            else:
                passed = False
                print(f"      ✗ Found {errors_in_split} errors in sample")

        return passed

    def _validate_synthetic_traces(self) -> bool:
        """Validate synthetic traces."""
        passed = True

        # Check against first trainer config from first directory
        reference_dir = [d for d in self.config_dirs if d.exists()][0]
        orig_config = self._load_original_config(reference_dir, 1)
        hp = orig_config["hyperparameters"]

        print(
            f"  Validating {len(self.synthetic_traces['traces'])} synthetic traces..."
        )

        for trace_name in ["syn_0", "syn_20", "syn_50"]:
            yaml_pattern = self.synthetic_traces["traces"][trace_name]["pattern"]
            json_pattern = eval(hp[f"avl_events_{trace_name}"])
            # Convert JSON tuples to lists for comparison
            json_pattern_as_lists = [
                list(item) if isinstance(item, tuple) else item for item in json_pattern
            ]

            if yaml_pattern != json_pattern_as_lists:
                self.errors.append(f"Synthetic trace {trace_name} mismatch")
                passed = False
                print(f"    ✗ {trace_name}: MISMATCH")
            else:
                print(f"    ✓ {trace_name}: {len(yaml_pattern)} events")

        return passed

    def _validate_mobiperf_traces(self) -> bool:
        """Validate mobiperf traces for a sample of trainers."""
        passed = True

        reference_dir = [d for d in self.config_dirs if d.exists()][0]

        # Sample 10 trainers instead of all 300
        sample_trainer_nums = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

        print(
            f"  Validating mobiperf traces (sampling {len(sample_trainer_nums)} devices)..."
        )

        errors_found = 0
        for trainer_num in sample_trainer_nums:
            device_id = f"device_{trainer_num:03d}"

            if device_id not in self.mobiperf_traces["traces"]:
                self.errors.append(f"{device_id}: Not found in YAML")
                errors_found += 1
                continue

            trace_data = self.mobiperf_traces["traces"][device_id]

            try:
                orig_config = self._load_original_config(reference_dir, trainer_num)
                hp = orig_config["hyperparameters"]

                # Check all three trace variants
                for variant in ["2st", "3st_50", "3st_75"]:
                    yaml_trace = trace_data[f"states_{variant}"]
                    json_trace = eval(hp[f"avl_events_mobiperf_{variant}"])
                    # Convert JSON tuples to lists for comparison
                    json_trace_as_lists = [
                        list(item) if isinstance(item, tuple) else item
                        for item in json_trace
                    ]

                    if yaml_trace != json_trace_as_lists:
                        self.errors.append(f"{device_id}: mobiperf_{variant} mismatch")
                        errors_found += 1
            except Exception as e:
                self.errors.append(f"{device_id}: Validation error - {e}")
                errors_found += 1

        if errors_found == 0:
            num_devices = len(self.mobiperf_traces["traces"])
            print(
                f"    ✓ All {num_devices} device traces valid (sampled {len(sample_trainer_nums)})"
            )
            print(
                f"      Each device has 3 variants: 2-state, 3-state-50%, 3-state-75%"
            )
        else:
            passed = False
            print(f"    ✗ Found {errors_found} errors in mobiperf traces")

        return passed

    def _validate_cross_references(self) -> bool:
        """Validate that all cross-references are valid."""
        passed = True

        print("  Checking cross-references...")

        # Check that mobiperf device IDs in registry exist in traces
        missing_devices = 0
        for trainer_key, trainer_meta in self.trainer_registry["trainers"].items():
            device_id = trainer_meta.get("mobiperf_device_id")
            if device_id and device_id not in self.mobiperf_traces["traces"]:
                self.errors.append(
                    f"{trainer_key}: References non-existent device {device_id}"
                )
                missing_devices += 1

        if missing_devices == 0:
            print("    ✓ All mobiperf device references valid")
        else:
            passed = False
            print(f"    ✗ Found {missing_devices} missing device references")

        # Check that dataset splits reference valid trainers
        invalid_refs = 0
        for split_name, split_data in self.dataset_splits.items():
            for trainer_key in split_data["trainer_data_splits"].keys():
                if trainer_key not in self.trainer_registry["trainers"]:
                    self.errors.append(
                        f"{split_name}: References non-existent {trainer_key}"
                    )
                    invalid_refs += 1

        if invalid_refs == 0:
            print("    ✓ All dataset split trainer references valid")
        else:
            passed = False
            print(f"    ✗ Found {invalid_refs} invalid trainer references")

        return passed

    def _validate_completeness(self) -> bool:
        """Check that no data is missing."""
        passed = True

        print("  Checking completeness...")

        # Check all 300 trainers present
        trainer_ids = set(range(1, 301))
        actual_ids = {
            meta["trainer_id"] for meta in self.trainer_registry["trainers"].values()
        }
        missing = trainer_ids - actual_ids

        if missing:
            self.errors.append(f"Missing trainers: {sorted(list(missing)[:10])}")
            passed = False
            print(f"    ✗ Missing {len(missing)} trainers from registry")
        else:
            print("    ✓ All 300 trainers present in registry")

        # Check all dataset splits have 300 trainers
        incomplete_splits = 0
        for split_name, split_data in self.dataset_splits.items():
            expected = 300
            actual = len(split_data["trainer_data_splits"])
            if actual != expected:
                self.errors.append(f"{split_name}: Has {actual}/{expected} trainers")
                incomplete_splits += 1

        if incomplete_splits == 0:
            print(
                f"    ✓ All {len(self.dataset_splits)} splits complete (300 trainers each)"
            )
        else:
            passed = False
            print(f"    ✗ Found {incomplete_splits} incomplete splits")

        return passed


# Main execution
if __name__ == "__main__":
    metadata_dir = Path(__file__).parent.parent / "metadata"
    base_dir = Path(__file__).parent.parent / "trainer"

    config_dirs = [
        base_dir / "config_dir0.1_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir1_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir10_num300_traceFail_6d_3state_oort",
        base_dir / "config_dir100_num300_traceFail_6d_3state_oort",
    ]

    print(f"\nMetadata directory: {metadata_dir}")
    print(f"Config base directory: {base_dir}\n")

    validator = MetadataValidator(metadata_dir, config_dirs)
    passed = validator.validate_all()

    print("\n" + "=" * 70)
    sys.exit(0 if passed else 1)
