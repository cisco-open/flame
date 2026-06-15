"""
Aggregator spawner for Phase 3.

Spawns aggregator process with log capture.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class AggregatorSpawner:
    """Spawns aggregator process with log capture."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.process = None
        self._log_handle = None

    def spawn(
        self,
        aggregator_main_path: Path,
        config_path: Optional[Path] = None,
        config_json: Optional[str] = None,
        log_to_wandb: bool = False,
        wandb_run_name: Optional[str] = None,
        cpu_cores: Optional[set] = None,
    ) -> subprocess.Popen:
        """Spawn aggregator process.

        Pass either `config_path` (file) or `config_json` (serialized dict).
        ``cpu_cores``: optional set of CPU core ids to pin the aggregator to. The
        aggregator is a single, message-processing-bound process (chunk reassembly
        + recv loop + serial commit); pinning it to cores reserved away from the
        trainer pool keeps the 300 pinned trainers from time-slicing it.
        """
        if (config_path is None) == (config_json is None):
            raise ValueError("provide exactly one of config_path or config_json")

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_file, "w", buffering=1)
            stdout_target = self._log_handle
            stderr_target = subprocess.STDOUT
        else:
            stdout_target = subprocess.PIPE
            stderr_target = subprocess.PIPE

        cmd = [sys.executable, str(aggregator_main_path)]
        if config_json is not None:
            cmd.extend(["--config-json", config_json])
        else:
            cmd.append(str(config_path))

        # Add optional wandb flags
        if log_to_wandb:
            # Validate that wandb is available before enabling wandb logging
            wandb_available = False
            try:
                # Local import to avoid making wandb a hard dependency of this module
                # wandb is an optional dependency that may not be installed
                import wandb  # type: ignore[import-untyped]

                # If import succeeds, assume wandb is available; detailed config checks
                # (e.g., API key) are handled by the aggregator process itself.
                wandb_available = True
            except ImportError:
                print(
                    "  ⚠ wandb logging was requested, but the 'wandb' package is not "
                    "installed. Continuing without wandb logging. "
                    "Install with: pip install wandb"
                )
            except Exception as exc:
                print(
                    f"  ⚠ wandb logging was requested, but wandb import failed: {exc}. "
                    "Continuing without wandb logging."
                )

            if wandb_available:
                cmd.append("--log_to_wandb")
                if wandb_run_name:
                    cmd.extend(["--wandb_run_name", wandb_run_name])

        # CPU pinning: confine the aggregator to its reserved cores and let its
        # math libs use exactly that many threads (it benefits from a few cores
        # for chunk reassembly / aggregation, unlike a 1-core-pinned trainer).
        env = os.environ.copy()
        preexec_fn = None
        if cpu_cores:
            _cores = {int(c) for c in cpu_cores}
            if hasattr(os, "sched_setaffinity"):
                preexec_fn = lambda c=_cores: os.sched_setaffinity(0, c)
                _nthreads = str(len(_cores))
                for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[_var] = _nthreads
                print(f"  ✓ Aggregator pinned to {len(_cores)} core(s): {sorted(_cores)}")
            else:
                print("  (aggregator pinning requested but sched_setaffinity unavailable)")

        # Spawn process
        self.process = subprocess.Popen(
            cmd, stdout=stdout_target, stderr=stderr_target, text=True,
            env=env, preexec_fn=preexec_fn,
        )

        print(f"  ✓ Aggregator started (PID: {self.process.pid})")
        if self.log_file:
            print(f"    Logs: {self.log_file}")

        return self.process

    def is_running(self) -> bool:
        """Check if aggregator is still running."""
        if not self.process:
            return False
        return self.process.poll() is None

    def wait_until_ready(self, timeout: int = 30) -> bool:
        """
        Wait for aggregator to be ready.

        Simple implementation: just wait fixed time and check process is alive.
        Could be enhanced with log monitoring for "ready" message.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if aggregator appears ready, False otherwise
        """
        print(f"  Waiting for aggregator to be ready (up to {timeout}s)...")

        start_time = time.time()
        check_interval = 1.0

        while time.time() - start_time < timeout:
            if not self.is_running():
                print(f"  ✗ Aggregator process died")
                return False

            time.sleep(check_interval)
            elapsed = time.time() - start_time

            # Simple heuristic: if process is alive for 5 seconds, assume ready
            if elapsed >= 5:
                print(f"  ✓ Aggregator ready (process alive for {elapsed:.1f}s)")
                return True

        print(f"  ⚠ Timeout waiting for aggregator")
        return self.is_running()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the aggregator process exits, or ``timeout`` seconds pass.

        Returns True if the process exited, False if the timeout fired while it
        was still running (deadlock guard — caller should then ``terminate``).
        ``timeout=None`` blocks indefinitely (legacy behavior).
        """
        if not self.process:
            return True
        try:
            self.process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def terminate(self):
        """Terminate aggregator process."""
        if self.process:
            try:
                self.process.terminate()
                time.sleep(2)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as exc:
                # Process may have already terminated; ignore cleanup errors
                print(f"  ⚠ Failed to terminate aggregator cleanly: {exc}")

        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
