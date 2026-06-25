# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""ExperimentRunner: orchestrates a single experiment end-to-end.

Paths are resolved as follows (lowest precedence → highest):
1. Defaults relative to example_dir.
2. ExampleConfig fields in the experiment YAML.
3. MetadataPaths fields in the experiment YAML.
4. Constructor overrides.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from flame.launch.aggregator_spawner import AggregatorSpawner
from flame.launch.baselines import (
    format_provenance,
    load_baselines,
    merge_with_provenance,
)
from flame.launch.execution_config_generator import (
    create_execution_config,
    save_execution_config,
)
from flame.launch.experiment_config import ExperimentConfig
from flame.launch.snapshot import ExperimentSnapshot

try:
    from flame.launch.resource_monitor import create_monitor_from_config
except ImportError:
    create_monitor_from_config = None  # type: ignore
from flame.launch.spawner import ConfigGenerator, MetadataLoader, TrainerSpawner


REPO_ROOT_HINT = "examples"


def _resolve(base: Path, rel: Optional[str]) -> Optional[Path]:
    if rel is None:
        return None
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class ExperimentRunner:
    """Run experiments end-to-end against a generic example layout."""

    def __init__(
        self,
        example_dir: Path,
        metadata_dir: Optional[Path] = None,
    ):
        self.example_dir = Path(example_dir).resolve()
        self.metadata_dir_default = (
            Path(metadata_dir).resolve()
            if metadata_dir is not None
            else self.example_dir / "metadata"
        )
        self.experiments_dir = self.example_dir / "experiments"

        self.current_exp_dir: Optional[Path] = None
        self.aggregator_spawner: Optional[AggregatorSpawner] = None
        self.trainer_spawner: Optional[TrainerSpawner] = None
        self.resource_monitor = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _resolve_example_paths(self, exp: ExperimentConfig) -> dict:
        """Resolve trainer_main/aggregator_main/trainer_base/metadata_dir."""
        ex_dir = (
            Path(exp.example.dir).resolve()
            if exp.example.dir
            else self.example_dir
        )

        meta_dir = (
            Path(exp.metadata.dir).resolve()
            if exp.metadata.dir
            else self.metadata_dir_default
        )
        registry = _resolve(meta_dir, exp.metadata.registry) or (
            meta_dir / "trainer_registry.yaml"
        )

        return {
            "example_dir": ex_dir,
            "trainer_main": ex_dir / exp.example.trainer_main,
            "trainer_base": ex_dir / exp.example.trainer_base,
            "metadata_dir": meta_dir,
            "registry_path": registry,
        }

    def run_experiment(self, exp: ExperimentConfig) -> None:
        print(f"\n{'=' * 70}\nRUNNING EXPERIMENT: {exp.name}\n{'=' * 70}")

        paths = self._resolve_example_paths(exp)

        try:
            self.current_exp_dir = self._create_experiment_directory(exp)
            print(f"  exp dir: {self.current_exp_dir}")

            baselines = load_baselines(paths["metadata_dir"])
            baseline_entry = self._resolve_baseline(exp, baselines)

            agg_config_path = (
                paths["example_dir"] / exp.aggregator.config_template
                if exp.aggregator and exp.aggregator.config_template
                else None
            )
            agg_cfg, agg_provenance = self._build_aggregator_config(
                exp, agg_config_path, baseline_entry
            )
            # Propagate the simulation time mode to the aggregator (it needs to
            # know whether to order updates by a virtual clock or by arrival).
            agg_cfg.setdefault("hyperparameters", {})["time_mode"] = exp.trainer.time_mode
            agg_job_id = agg_cfg.get("job", {}).get("id")
            agg_job_name = agg_cfg.get("job", {}).get("name")
            if not agg_job_id:
                raise ValueError(
                    f"aggregator config missing job.id "
                    f"(template={agg_config_path}, baseline={exp.baseline})"
                )

            # Baseline owns the aggregator stack; resolve it and fail fast on
            # any selector/stack mismatch before spawning.
            agg_main_rel = self._resolve_aggregator_main(exp, baseline_entry)
            paths["aggregator_main"] = paths["example_dir"] / agg_main_rel
            self._validate_stack(paths["aggregator_main"], agg_cfg)

            # Stash + log provenance.
            agg_cfg_path_out = self.current_exp_dir / "aggregator_config.json"
            with open(agg_cfg_path_out, "w") as f:
                json.dump(agg_cfg, f, indent=4)
            print(f"  aggregator config written: {agg_cfg_path_out}")
            print(format_provenance("aggregator", agg_provenance))

            metadata_loader = MetadataLoader(paths["metadata_dir"])
            config_gen = ConfigGenerator(metadata_loader, paths["trainer_base"])

            log_prefix = exp.get_log_prefix()
            agg_log = self.current_exp_dir / f"{log_prefix}_aggregator.log"
            trainers_log = self.current_exp_dir / f"{log_prefix}_trainers.log"
            monitor_log = self.current_exp_dir / f"{log_prefix}_resources.log"

            # Telemetry: every spawned process inherits this dir and writes its
            # own JSONL stream. Set before spawning so children pick it up.
            self.telemetry_dir = self.current_exp_dir / "telemetry"
            os.environ["FLAME_TELEMETRY_DIR"] = str(self.telemetry_dir)
            # UTF-8 child stdio so status glyphs don't crash on latin-1 locales.
            os.environ.setdefault("PYTHONIOENCODING", "utf-8")

            # CPU partition: reserve a few cores for the single, message-processing
            # -bound aggregator so the 300 pinned trainers don't time-slice it
            # (the aggregator's recv/chunk-reassembly throughput sets the sim's
            # commit rate). Trainers pin to the remaining cores.
            reserved_cores: set = set()
            if hasattr(os, "sched_getaffinity"):
                _all = sorted(os.sched_getaffinity(0))
                _n = min(8, max(2, len(_all) // 8))
                reserved_cores = set(_all[:_n])
                print(f"  CPU partition: {len(reserved_cores)} core(s) reserved for "
                      f"aggregator {sorted(reserved_cores)}, "
                      f"{len(_all) - len(reserved_cores)} for trainers")

            self.aggregator_spawner = AggregatorSpawner(log_file=agg_log)
            self.trainer_spawner = TrainerSpawner(
                config_gen,
                num_gpus=exp.execution.num_gpus,
                sleep_between_spawns=exp.execution.sleep_between_spawns,
                log_file=trainers_log,
                # CLI-only knobs passed on the trainer command line.
                time_mode=exp.trainer.time_mode,
                battery_threshold=exp.trainer.battery_threshold,
                reserved_cores=reserved_cores,
            )

            if exp.execution.monitoring.enabled and create_monitor_from_config is not None:
                self.resource_monitor = create_monitor_from_config(
                    monitor_log,
                    {
                        "check_interval_seconds": exp.execution.monitoring.check_interval_seconds,
                        "ram_warning_percent": exp.execution.monitoring.ram_warning_percent,
                        "ram_critical_percent": exp.execution.monitoring.ram_critical_percent,
                        "gpu_warning_percent": exp.execution.monitoring.gpu_warning_percent,
                        "gpu_critical_percent": exp.execution.monitoring.gpu_critical_percent,
                    },
                )

            self.aggregator_spawner.spawn(
                paths["aggregator_main"],
                config_json=json.dumps(agg_cfg),
                log_to_wandb=exp.aggregator.log_to_wandb,
                wandb_run_name=exp.aggregator.wandb_run_name,
                cpu_cores=reserved_cores,
            )
            if not self.aggregator_spawner.wait_until_ready(
                exp.execution.aggregator_warmup_time
            ):
                raise RuntimeError("aggregator failed to start")

            if self.resource_monitor:
                self.resource_monitor.start()

            agg_spawn_cmd = [
                sys.executable, str(paths["aggregator_main"]),
                "--config-json", "<inline>",
            ]
            trainer_spawn_cmd = self._build_trainer_spawn_command(exp)

            exec_config = create_execution_config(
                exp,
                agg_cfg_path_out.relative_to(self.current_exp_dir),
                spawn_commands={
                    "aggregator": [str(c) for c in agg_spawn_cmd],
                    "trainers": [str(c) for c in trainer_spawn_cmd],
                },
            )
            save_execution_config(exec_config, self.current_exp_dir / "execution_config.yaml")

            snapshot = ExperimentSnapshot(self.current_exp_dir)
            snapshot.create_snapshot(
                exp, paths["metadata_dir"], agg_cfg_path_out,
                trainer_spawn_cmd, agg_spawn_cmd,
            )

            trainer_ids = list(
                range(exp.trainer.start_id, exp.trainer.start_id + exp.trainer.num_trainers)
            )
            config_overrides = {
                "job.id": agg_job_id,
                "job.name": agg_job_name,
                "hyperparameters.training_delay_enabled": str(exp.trainer.enable_training_delays),
            }
            if exp.trainer.hyperparameters:
                for key, value in exp.trainer.hyperparameters.items():
                    config_overrides[f"hyperparameters.{key}"] = value

            # Trainer-side baseline + experiment override merge happens inside
            # ConfigGenerator so that per-trainer state can flow through the
            # existing dotted-key override mechanism. The baseline.trainer dict
            # and exp.trainer.config_overrides dict are deep-merged first; then
            # the dotted-key overrides (job.id, etc.) are applied last.
            baseline_trainer = (baseline_entry or {}).get("trainer") or {}
            exp_trainer_overrides = exp.trainer.config_overrides or {}
            if baseline_trainer or exp_trainer_overrides:
                merged_t, t_prov = merge_with_provenance([
                    (f"baseline:{exp.baseline}", baseline_trainer),
                    ("experiment.trainer.config_overrides", exp_trainer_overrides),
                ])
                config_gen.set_baseline_overrides(merged_t)
                print(format_provenance("trainer", t_prov))

            self.trainer_spawner.spawn_all(
                trainer_ids,
                alpha=exp.trainer.dataset.dirichlet_alpha,
                availability_mode=exp.trainer.availability.mode,
                trainer_main_path=paths["trainer_main"],
                **config_overrides,
            )

            print(f"\nexperiment running. logs: {agg_log}, {trainers_log}")
            # Wait for the aggregator to finish all rounds first, then give
            # trainers a short grace window to process the EOT broadcast and
            # exit cleanly. Without this, wait_all()'s per-trainer timeout fires
            # immediately after spawn and kills trainers every 30s regardless of
            # whether training is still in progress.
            # Watchdog: the aggregator self-stops at max_runtime_s (real) / sim_wall_ceiling_s
            # (sim). If it instead DEADLOCKS (MQTT/barrier) it would block this wait forever and
            # hang the batch. Bound the wait at the run's budget + a generous grace and hard-kill
            # on timeout so the batch proceeds to _cleanup/_sweep_stragglers instead of hanging.
            hp = agg_cfg.get("hyperparameters", {}) or {}
            try:
                budget_s = max(float(hp.get("max_runtime_s") or 0.0),
                               float(hp.get("sim_wall_ceiling_s") or 0.0))
            except (TypeError, ValueError):
                budget_s = 0.0
            watchdog_s = (budget_s + 1200.0) if budget_s > 0 else None
            wd_msg = f"{watchdog_s:.0f}s" if watchdog_s else "no limit"
            print(f"  waiting for aggregator to finish... (watchdog {wd_msg})")
            if not self.aggregator_spawner.wait(timeout=watchdog_s):
                print(f"  ⚠ aggregator still running after watchdog {wd_msg} — "
                      f"assuming deadlock; killing it (run budget was {budget_s:.0f}s)")
                self.aggregator_spawner.terminate()
            print("  aggregator done, waiting for trainers to exit...")
            self.trainer_spawner.wait_all(timeout_per_trainer=30.0)
            print("\nexperiment completed.")

            # Auto post-run analysis: parse the telemetry JSONL and emit plots.
            # Best-effort; a plotting failure must not fail the experiment.
            self._run_post_analysis()

        except Exception as e:
            import traceback

            print(f"\nexperiment failed: {e}")
            traceback.print_exc()
            raise
        finally:
            self._cleanup()

    def run_experiment_batch(self, config_file: Path) -> None:
        from flame.launch.experiment_config import load_experiment_config

        batch = load_experiment_config(config_file)
        print(f"loaded {len(batch.experiments)} experiments from {config_file}")
        # Start from a clean slate: a previous invocation (or a crash) can leave
        # orphan trainer/aggregator processes holding GPUs and MQTT state that
        # poison the first experiment. _sweep_stragglers runs only BETWEEN runs,
        # so sweep once up front too (covers the first experiment).
        print("pre-batch cleanup: sweeping any stale trainer/aggregator processes...")
        self._sweep_stragglers()
        # Overnight/CI-safe: when stdin is not a TTY (or FLAME_BATCH_CONTINUE_ON_ERROR
        # is set) a failed experiment is logged and the batch proceeds to the next,
        # rather than blocking on input(). run_experiment already cleans up its own
        # spawners in a finally; we additionally hard-sweep stragglers between runs.
        auto_continue = (
            not sys.stdin.isatty()
            or os.environ.get("FLAME_BATCH_CONTINUE_ON_ERROR", "") not in ("", "0", "false")
        )
        for i, exp in enumerate(batch.experiments, 1):
            print(f"\n[{i}/{len(batch.experiments)}] {exp.name}")
            try:
                self.run_experiment(exp)
            except Exception as e:
                print(f"experiment {exp.name} failed: {e}")
                if not auto_continue:
                    if input("continue? (y/n): ").lower() != "y":
                        break
                else:
                    print("  (non-interactive: continuing to next experiment)")
            finally:
                self._sweep_stragglers()

    def _resolve_baseline(
        self, exp: ExperimentConfig, baselines: dict
    ) -> Optional[dict]:
        if not exp.baseline:
            return None
        if exp.baseline not in baselines:
            raise ValueError(
                f"baseline {exp.baseline!r} not found in baselines.yaml. "
                f"Available: {sorted(baselines)}"
            )
        entry = baselines[exp.baseline]
        print(f"  baseline: {exp.baseline}")
        desc = entry.get("description")
        if desc:
            print(f"    {desc.strip()}")
        return entry

    def _resolve_aggregator_main(
        self, exp: ExperimentConfig, baseline_entry: Optional[dict]
    ) -> str:
        base_val = ((baseline_entry or {}).get("example") or {}).get("aggregator_main")
        if base_val:
            if exp.example.aggregator_main:
                raise ValueError(
                    f"baseline {exp.baseline!r} owns example.aggregator_main "
                    f"({base_val!r}); remove the experiment-level override."
                )
            return base_val
        return exp.example.aggregator_main or "aggregator/pytorch/main.py"

    # async selectors require the asyncfl stack; everything else is sync.
    _ASYNC_STACKS = {"asyncfl", "coord_asyncfl"}
    _ASYNC_SELECTORS = {"async_oort", "async_random", "fedbuff"}

    def _validate_stack(self, agg_main_path: Path, agg_cfg: dict) -> None:
        text = Path(agg_main_path).read_text()
        m = re.search(
            r"from flame\.mode\.horizontal\.(\w+)\.top_aggregator import", text
        )
        stack = m.group(1) if m else "syncfl"
        selector = (agg_cfg.get("selector") or {}).get("sort", "")
        is_async_stack = stack in self._ASYNC_STACKS
        is_async_sel = selector in self._ASYNC_SELECTORS
        if is_async_stack != is_async_sel:
            raise ValueError(
                f"selector/stack mismatch: selector={selector!r} "
                f"(async={is_async_sel}) cannot run on aggregator stack "
                f"{stack!r} (async={is_async_stack}). main={agg_main_path}"
            )

    def _build_aggregator_config(
        self,
        exp: ExperimentConfig,
        template_path: Optional[Path],
        baseline_entry: Optional[dict],
    ) -> tuple[dict, dict]:
        """Merge template + baseline.aggregator + experiment overrides.

        Returns (final_dict, provenance) where provenance maps each leaf path
        to the layer name that contributed it.
        """
        layers: list[tuple[str, dict]] = []
        if template_path is not None:
            if not template_path.exists():
                raise FileNotFoundError(f"aggregator config_template not found: {template_path}")
            with open(template_path) as f:
                tmpl = json.load(f)
            layers.append((f"config_template:{template_path.name}", tmpl))

        if baseline_entry:
            agg_layer = baseline_entry.get("aggregator")
            if agg_layer:
                layers.append((f"baseline:{exp.baseline}", agg_layer))

        if exp.aggregator and exp.aggregator.config_overrides:
            layers.append(
                ("experiment.aggregator.config_overrides", exp.aggregator.config_overrides)
            )

        merged, provenance = merge_with_provenance(layers)
        return merged, provenance

    def _create_experiment_directory(self, exp: ExperimentConfig) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.experiments_dir / f"run_{timestamp}_{exp.name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_trainer_spawn_command(self, exp: ExperimentConfig) -> list:
        return [
            "python3",
            "-m",
            "flame.launch.spawn_trainer_cli",
            "--alpha", str(exp.trainer.dataset.dirichlet_alpha),
            "--availability", exp.trainer.availability.mode,
            "--num-trainers", str(exp.trainer.num_trainers),
            "--start-id", str(exp.trainer.start_id),
            "--num-gpus", str(exp.execution.num_gpus),
        ]

    def _signal_handler(self, signum, frame):
        print("\ntermination signal received")
        self._cleanup()
        sys.exit(130)

    def _run_post_analysis(self):
        """Run telemetry analysis to generate plots into <exp_dir>/plots/.

        Best-effort: never raise. Skips silently if no telemetry was produced.
        """
        telemetry_dir = getattr(self, "telemetry_dir", None)
        if not telemetry_dir or not telemetry_dir.exists():
            return
        # repo root: lib/python/flame/launch/runner.py -> parents[4]
        analyzer = (
            Path(__file__).resolve().parents[4]
            / "scripts"
            / "analysis"
            / "analyze_run.py"
        )
        if not analyzer.exists():
            print(f"  (skipping analysis: {analyzer} not found)")
            return
        try:
            print(f"\nrunning telemetry analysis on {telemetry_dir} ...")
            subprocess.run(
                [sys.executable, str(analyzer), str(telemetry_dir)],
                check=False,
            )
        except Exception as e:
            print(f"  (telemetry analysis failed: {e})")

    def _cleanup(self):
        if self.resource_monitor:
            self.resource_monitor.stop()
        if self.trainer_spawner:
            self.trainer_spawner.terminate_all()
        if self.aggregator_spawner:
            self.aggregator_spawner.terminate()

    def _sweep_stragglers(self, gpu_settle_timeout_s: float = 45.0) -> None:
        """Between batch experiments, hard-kill any example trainer/aggregator
        processes that outlived ``_cleanup`` (e.g. a hung trainer) and wait for
        GPU memory to drain, so the next experiment starts from a clean slate.

        Matches ONLY the example's main scripts — never the batch runner itself
        (``run_experiment``) — so it is safe to call from inside the batch loop."""
        for pat in ("trainer/pytorch/main.py", "aggregator/pytorch/main_"):
            try:
                subprocess.run(["pkill", "-9", "-f", pat], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # best-effort: wait until our user's GPU procs are gone (or timeout)
        deadline = time.time() + gpu_settle_timeout_s
        while time.time() < deadline:
            try:
                out = subprocess.run(
                    ["pgrep", "-f", "trainer/pytorch/main.py"],
                    capture_output=True, text=True, check=False)
                if not out.stdout.strip():
                    break
            except Exception:
                break
            time.sleep(2)
        # drop spawner handles so stale references aren't reused next iteration
        self.trainer_spawner = None
        self.aggregator_spawner = None
