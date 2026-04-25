#!/usr/bin/env python3
"""
Launches a single FL experiment with run-isolated MQTT namespacing.

Each invocation generates a unique run_id (datetime + tag), patches the JSON
configs in-memory to embed that run_id into job.id and taskid, writes patched
configs to a temp directory, spawns the aggregator and trainers, and cleans up
all temp files when the run finishes or is terminated.

Layout:
  /tmp/fl_run_{run_id}/    <- TEMP: patched JSON configs only; deleted on exit
  {log_dir}/{run_id}/      <- PERMANENT: aggregator.log, trainers.log, pids.json

Usage:
    python3 launch_single_run.py \\
        --tag          baseline \\
        --agg-json     aggregator_async_base.json \\
        --num-trainers 100 \\
        --gpus         0,1,2 \\
        --log-level    INFO
"""

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_DIR   = SCRIPT_DIR / "json_scripts"
REPO_PATH  = Path(os.environ.get("REPO_PATH", str(SCRIPT_DIR / "../../../../../..")))

AGG_MAIN   = REPO_PATH / "lib/python/examples/fwdllm/aggregator/fl_main.py"
TRAIN_MAIN = REPO_PATH / "lib/python/examples/fwdllm/trainer/fl_main.py"

_TAG = "launcher"  # overwritten with --tag value in main() so signal handlers print the right prefix
_spawned_procs: List[subprocess.Popen] = []
_open_log_handles: list = []


def _log(msg: str) -> None:
    print(f"[{_TAG}] {msg}", flush=True)


def load_and_patch(json_path: Path, run_id: str) -> dict:
    """Load a JSON config and suffix job.id + taskid with run_id."""
    with open(json_path) as f:
        cfg = json.load(f)
    cfg["taskid"]    = f"{cfg['taskid']}_{run_id}"
    cfg["job"]["id"] = f"{cfg['job']['id']}_{run_id}"
    return cfg


def write_config(config: dict, dest: Path) -> None:
    with open(dest, "w") as f:
        json.dump(config, f, indent=4)

def _close_log_handles() -> None:
    for fh in _open_log_handles:
        try:
            fh.close()
        except Exception:
            pass


def _kill_all() -> None:
    """SIGTERM all spawned processes; SIGKILL any that survive 3 seconds."""
    alive = [p for p in _spawned_procs if p.poll() is None]
    if alive:
        _log(f"Sending SIGTERM to {len(alive)} process(es)…")
        for proc in alive:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
        time.sleep(3)
        for proc in _spawned_procs:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGKILL)
                except OSError:
                    pass
    _close_log_handles()


def _signal_handler(signum, _frame) -> None:
    print(f"\n[{_TAG}] Signal {signum} — cleaning up.", flush=True)
    _kill_all()
    sys.exit(128 + signum)  # triggers atexit (rmtree of temp dir)


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def spawn(cmd: List[str], env: dict, log_path: Path, append: bool = False) -> subprocess.Popen:
    mode  = "ab" if append else "wb"
    log_f = open(log_path, mode)
    _open_log_handles.append(log_f)
    proc  = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        # fl_main.py resolves imports via os.getcwd()-relative sys.path inserts;
        # it must be launched from run_tc_expts/ so that cwd/../../../../ == lib/python.
        cwd=str(SCRIPT_DIR),
        # Own session: won't receive terminal SIGINT directly; we manage cleanup.
        start_new_session=True,
    )
    _spawned_procs.append(proc)
    return proc

def write_pid_registry(log_dir: Path, run_id: str, tag: str,
                       agg_pid: int, trainer_pids: List[int]) -> None:
    registry = {
        "run_id":         run_id,
        "run_tag":        tag,
        "start_time":     datetime.now().isoformat(),
        "aggregator_pid": agg_pid,
        "trainer_pids":   trainer_pids,
    }
    with open(log_dir / "pids.json", "w") as f:
        json.dump(registry, f, indent=2)

def main() -> int:
    global _TAG

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag",           required=True,
                        help="Short label for this run, e.g. 'baseline'")
    parser.add_argument("--agg-json",      required=True,
                        help="Aggregator template filename inside json_scripts/")
    parser.add_argument("--num-trainers",  type=int, required=True,
                        help="Number of trainers to spawn (trainer_0 .. trainer_N-1)")
    parser.add_argument("--gpus",          required=True,
                        help="Comma-separated physical GPU IDs for this run, e.g. '0,1,2'")
    parser.add_argument("--log-level",     default="INFO",
                        help="Log level passed to fl_main.py (default: INFO)")
    parser.add_argument("--log-dir",       default=None,
                        help="Directory to write permanent logs (default: "
                             "<script_dir>/logs/<run_id>)")
    parser.add_argument("--sleep-between-trainers", type=float, default=8.0,
                        help="Seconds between successive trainer spawns (default: 8)")
    parser.add_argument("--aggregator-warmup", type=float, default=10.0,
                        help="Seconds to wait after spawning aggregator (default: 10)")
    args = parser.parse_args()

    # Set tag early so signal handlers print the right prefix.
    _TAG = args.tag

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        _log("ERROR: --gpus must not be empty")
        return 1

    # 1. Build run ID and directories
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{args.tag}"

    # Temp dir: patched JSON configs only — always deleted on exit.
    run_dir = Path(f"/tmp/fl_run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log dir: aggregator.log, trainers.log, pids.json — preserved after exit.
    log_dir = Path(args.log_dir) / run_id if args.log_dir else SCRIPT_DIR / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # atexit is LIFO: register rmtree first so _kill_all runs before temp cleanup.
    atexit.register(shutil.rmtree, run_dir, True)
    atexit.register(_kill_all)

    _log(f"run_id   = {run_id}")
    _log(f"tmp_dir  = {run_dir}  [deleted on exit]")
    _log(f"log_dir  = {log_dir}  [preserved after exit]")
    _log(f"gpus     = {gpu_list}")
    _log(f"trainers = {args.num_trainers}")

    # 2. Pre-flight: verify all source JSON files exist before touching /tmp
    agg_src = JSON_DIR / args.agg_json
    if not agg_src.exists():
        _log(f"ERROR: aggregator config not found: {agg_src}")
        return 1
    missing = [f"trainer_{x}.json" for x in range(args.num_trainers)
               if not (JSON_DIR / f"trainer_{x}.json").exists()]
    if missing:
        _log(f"ERROR: {len(missing)} trainer config(s) missing from {JSON_DIR}: "
             f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
        return 1

    # 3. Patch aggregator config
    agg_dest = run_dir / "aggregator.json"
    write_config(load_and_patch(agg_src, run_id), agg_dest)

    # 4. Patch trainer configs
    trainer_dests: List[Path] = []
    for x in range(args.num_trainers):
        src  = JSON_DIR / f"trainer_{x}.json"
        dest = run_dir / f"trainer_{x}.json"
        write_config(load_and_patch(src, run_id), dest)
        trainer_dests.append(dest)

    _log(f"Patched {1 + args.num_trainers} configs → {run_dir}")

    # 5. Spawn aggregator
    agg_log = log_dir / "aggregator.log"
    agg_cmd = [
        sys.executable, str(AGG_MAIN),
        "--config",    str(agg_dest),
        "--log_level", args.log_level,
    ]
    _log("Spawning aggregator…")
    agg_proc = spawn(agg_cmd, env=os.environ.copy(), log_path=agg_log)
    _log(f"aggregator pid={agg_proc.pid}  log → {agg_log}")

    _log(f"Waiting {args.aggregator_warmup:.0f}s for aggregator to initialise…")
    time.sleep(args.aggregator_warmup)

    if agg_proc.poll() is not None:
        _log(f"ERROR: aggregator exited early (rc={agg_proc.returncode}). "
             f"Check {agg_log}")
        return 1

    # 6. Spawn trainers (round-robin over GPU set)
    trainer_log  = log_dir / "trainers.log"
    trainer_pids: List[int] = []

    for x, cfg_path in enumerate(trainer_dests):
        gpu_id = gpu_list[x % len(gpu_list)]
        env    = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            sys.executable, str(TRAIN_MAIN),
            "--config",    str(cfg_path),
            "--log_level", args.log_level,
        ]
        proc = spawn(cmd, env=env, log_path=trainer_log, append=True)
        trainer_pids.append(proc.pid)
        _log(f"trainer_{x:3d}  pid={proc.pid}  gpu={gpu_id}")

        time.sleep(args.sleep_between_trainers)

    _log(f"All {args.num_trainers} trainers spawned.")

    # 7. Write PID registry
    write_pid_registry(log_dir, run_id, args.tag, agg_proc.pid, trainer_pids)
    _log(f"PID registry → {log_dir / 'pids.json'}")

    # 8. Wait for aggregator, then clean up
    _log(f"Waiting for aggregator (pid={agg_proc.pid}) to finish…")
    agg_proc.wait()
    rc = agg_proc.returncode
    _log(f"Aggregator exited (rc={rc}). Terminating remaining trainers…")

    # atexit will call _kill_all() then shutil.rmtree(run_dir) on return.
    _log(f"Run {run_id} complete.")
    _log(f"Logs preserved at {log_dir}")
    _log(f"Temp configs at {run_dir} will be removed now.")
    return rc if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
