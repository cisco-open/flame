#!/usr/bin/env python3
"""
Resource monitoring for experiment runs.

Tracks RAM and GPU utilization to diagnose OOM kills and resource exhaustion.
Logs warnings when thresholds are exceeded.
"""
import time
import threading
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ResourceMonitor:
    """Monitor system resources during experiment execution."""

    def __init__(
        self,
        log_file: Path,
        check_interval: int = 30,
        ram_warning_threshold: float = 80.0,
        ram_critical_threshold: float = 90.0,
        gpu_warning_threshold: float = 80.0,
        gpu_critical_threshold: float = 90.0,
    ):
        """
        Initialize resource monitor.

        Args:
            log_file: Path to write monitoring logs
            check_interval: Seconds between resource checks
            ram_warning_threshold: RAM usage % to trigger warning
            ram_critical_threshold: RAM usage % to trigger critical alert
            gpu_warning_threshold: GPU memory % to trigger warning
            gpu_critical_threshold: GPU memory % to trigger critical alert
        """
        self.log_file = log_file
        self.check_interval = check_interval
        self.ram_warning_threshold = ram_warning_threshold
        self.ram_critical_threshold = ram_critical_threshold
        self.gpu_warning_threshold = gpu_warning_threshold
        self.gpu_critical_threshold = gpu_critical_threshold

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_available = self._check_gpu_availability()

        # Track if we've already warned to avoid spam
        self._ram_warned = False
        self._ram_critical = False
        self._gpu_warned = {}
        self._gpu_critical = {}

    def _check_gpu_availability(self) -> bool:
        """Check if GPUs are available."""
        try:
            import pynvml

            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False

    def start(self):
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._log("Resource monitoring started")

    def stop(self):
        """Stop monitoring."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._log("Resource monitoring stopped")

    def _log(self, message: str, level: str = "INFO"):
        """Write log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {level:8s} | {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_line)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_resources()
            except Exception as e:
                self._log(f"Error during resource check: {e}", "ERROR")

            # Wait for next check or stop signal
            self._stop_event.wait(self.check_interval)

    def _check_resources(self):
        """Check and log current resource usage."""
        # Check RAM
        ram_stats = self._get_ram_stats()
        self._check_ram_thresholds(ram_stats)

        # Check GPU if available
        gpu_stats = []
        if self._gpu_available:
            gpu_stats = self._get_gpu_stats()
            self._check_gpu_thresholds(gpu_stats)

        # Log current state
        self._log_status(ram_stats, gpu_stats)

    def _get_ram_stats(self) -> Dict[str, Any]:
        """Get current RAM statistics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total_gb": mem.total / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "percent": mem.percent,
            "swap_used_gb": swap.used / (1024 ** 3),
            "swap_percent": swap.percent,
        }

    def _get_gpu_stats(self) -> list[Dict[str, Any]]:
        """Get GPU statistics using nvidia-smi or pynvml."""
        try:
            import pynvml

            gpu_stats = []
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # Handle both bytes and str returned by nvmlDeviceGetName
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")

                gpu_stats.append(
                    {
                        "id": i,
                        "name": gpu_name,
                        "mem_used_gb": mem_info.used / (1024 ** 3),
                        "mem_total_gb": mem_info.total / (1024 ** 3),
                        "mem_percent": (mem_info.used / mem_info.total) * 100,
                        "gpu_util_percent": util.gpu,
                        "temperature_c": temp,
                    }
                )

            return gpu_stats

        except Exception as e:
            self._log(f"Failed to get GPU stats: {e}", "WARNING")
            return []

    def _check_ram_thresholds(self, ram_stats: Dict[str, Any]):
        """Check RAM against thresholds and log warnings."""
        percent = ram_stats["percent"]

        if percent >= self.ram_critical_threshold:
            if not self._ram_critical:
                self._log(
                    f"CRITICAL: RAM usage at {percent:.1f}% (>= {self.ram_critical_threshold}%) - "
                    f"Used: {ram_stats['used_gb']:.1f}GB / {ram_stats['total_gb']:.1f}GB - "
                    f"OOM kill risk HIGH!",
                    "CRITICAL",
                )
                self._ram_critical = True
                self._ram_warned = True
        elif percent >= self.ram_warning_threshold:
            if not self._ram_warned:
                self._log(
                    f"WARNING: RAM usage at {percent:.1f}% (>= {self.ram_warning_threshold}%) - "
                    f"Used: {ram_stats['used_gb']:.1f}GB / {ram_stats['total_gb']:.1f}GB",
                    "WARNING",
                )
                self._ram_warned = True
        else:
            # Reset flags when usage drops below warning
            if self._ram_warned and percent < self.ram_warning_threshold - 5:
                self._log(f"INFO: RAM usage recovered to {percent:.1f}%", "INFO")
                self._ram_warned = False
                self._ram_critical = False

    def _check_gpu_thresholds(self, gpu_stats: list[Dict[str, Any]]):
        """Check GPU memory against thresholds and log warnings."""
        for gpu in gpu_stats:
            gpu_id = gpu["id"]
            mem_percent = gpu["mem_percent"]

            if mem_percent >= self.gpu_critical_threshold:
                if not self._gpu_critical.get(gpu_id, False):
                    self._log(
                        f"CRITICAL: GPU {gpu_id} memory at {mem_percent:.1f}% "
                        f"(>= {self.gpu_critical_threshold}%) - "
                        f"Used: {gpu['mem_used_gb']:.1f}GB / {gpu['mem_total_gb']:.1f}GB - "
                        f"OOM risk HIGH!",
                        "CRITICAL",
                    )
                    self._gpu_critical[gpu_id] = True
                    self._gpu_warned[gpu_id] = True
            elif mem_percent >= self.gpu_warning_threshold:
                if not self._gpu_warned.get(gpu_id, False):
                    self._log(
                        f"WARNING: GPU {gpu_id} memory at {mem_percent:.1f}% "
                        f"(>= {self.gpu_warning_threshold}%) - "
                        f"Used: {gpu['mem_used_gb']:.1f}GB / {gpu['mem_total_gb']:.1f}GB",
                        "WARNING",
                    )
                    self._gpu_warned[gpu_id] = True
            else:
                # Reset flags when usage drops
                if self._gpu_warned.get(gpu_id, False) and mem_percent < self.gpu_warning_threshold - 5:
                    self._log(
                        f"INFO: GPU {gpu_id} memory recovered to {mem_percent:.1f}%",
                        "INFO",
                    )
                    self._gpu_warned[gpu_id] = False
                    self._gpu_critical[gpu_id] = False

    def _log_status(self, ram_stats: Dict[str, Any], gpu_stats: list[Dict[str, Any]]):
        """Log current resource status."""
        # RAM summary
        ram_msg = (
            f"RAM: {ram_stats['used_gb']:.1f}GB / {ram_stats['total_gb']:.1f}GB "
            f"({ram_stats['percent']:.1f}%) | "
            f"Swap: {ram_stats['swap_used_gb']:.1f}GB ({ram_stats['swap_percent']:.1f}%)"
        )

        # GPU summary
        if gpu_stats:
            gpu_msgs = []
            for gpu in gpu_stats:
                gpu_msgs.append(
                    f"GPU{gpu['id']}: {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.1f}GB "
                    f"({gpu['mem_percent']:.1f}%) Util:{gpu['gpu_util_percent']}% "
                    f"Temp:{gpu['temperature_c']}°C"
                )
            gpu_msg = " | ".join(gpu_msgs)
            full_msg = f"{ram_msg} | {gpu_msg}"
        else:
            full_msg = ram_msg

        self._log(full_msg)


def create_monitor_from_config(
    log_file: Path, config: Optional[Dict[str, Any]] = None
) -> ResourceMonitor:
    """
    Create resource monitor from configuration dict.

    Args:
        log_file: Path to monitoring log file
        config: Optional configuration with thresholds and intervals

    Returns:
        Configured ResourceMonitor instance
    """
    if config is None:
        config = {}

    return ResourceMonitor(
        log_file=log_file,
        check_interval=config.get("check_interval_seconds", 30),
        ram_warning_threshold=config.get("ram_warning_percent", 80.0),
        ram_critical_threshold=config.get("ram_critical_percent", 90.0),
        gpu_warning_threshold=config.get("gpu_warning_percent", 80.0),
        gpu_critical_threshold=config.get("gpu_critical_percent", 90.0),
    )
