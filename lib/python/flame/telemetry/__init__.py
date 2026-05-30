# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Structured, process-local telemetry for FLAME experiments.

Each process (aggregator, each trainer) writes typed JSONL event records to a
per-run directory so post-run analysis can compare selectors / aggregators /
trainers on an identical schema -- no log-regex scraping.

The writer is a process-global singleton configured once at startup. When no
run directory is configured (neither ``configure(run_dir=...)`` nor the
``FLAME_TELEMETRY_DIR`` env var), every :func:`emit` is a cheap no-op, so
existing runs and unit tests are unaffected unless telemetry is explicitly
enabled.

Generic by design: nothing here is example-specific. The async_cifar10 example
wires it first, but any example/aggregator/selector/trainer inherits the same
hooks.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import time
from typing import Any, Optional

from .events import (  # noqa: F401  (re-exported for convenience)
    EVENT_AGG_EVAL,
    EVENT_AGG_ROUND,
    EVENT_AVAIL_CHANGE,
    EVENT_RUN_META,
    EVENT_SELECTION,
    EVENT_TRAINER_ROUND,
    EVENT_UTIL_DISPARITY,
    KNOWN_EVENTS,
)

logger = logging.getLogger(__name__)

ENV_DIR = "FLAME_TELEMETRY_DIR"

_writer: Optional["TelemetryWriter"] = None
_config_lock = threading.Lock()


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for the value types telemetry carries."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    # numpy scalars / arrays without importing numpy
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


class TelemetryWriter:
    """Append-only JSONL writer. Thread-safe (trainers emit from threads)."""

    def __init__(self, path: str, role: str, end_id: Optional[str] = None) -> None:
        self.path = path
        self.role = role
        self.end_id = end_id
        self._lock = threading.Lock()
        # line-buffered so a killed process still leaves complete records
        self._fh = open(path, "a", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        rec: dict[str, Any] = {
            "ts": time.time(),
            "role": self.role,
            "event": event,
        }
        if self.end_id is not None:
            rec["end_id"] = self.end_id
        rec.update(fields)
        line = json.dumps(rec, default=_json_default)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass


def configure(
    role: str,
    end_id: Optional[str] = None,
    run_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[TelemetryWriter]:
    """Initialize the process-global telemetry writer.

    Parameters
    ----------
    role: short role label, e.g. "aggregator" or "trainer".
    end_id: per-process identity (e.g. trainer/task id); used in the filename.
    run_dir: output directory. Falls back to ``$FLAME_TELEMETRY_DIR``. If still
        unset, telemetry stays disabled and this returns ``None``.
    filename: override the default ``<role>[_<end_id>].jsonl`` filename.

    Returns the writer, or ``None`` if telemetry is disabled.
    """
    global _writer
    run_dir = run_dir or os.environ.get(ENV_DIR)
    if not run_dir:
        return None

    if filename is None:
        safe_id = str(end_id).replace("/", "_") if end_id is not None else None
        filename = f"{role}_{safe_id}.jsonl" if safe_id is not None else f"{role}.jsonl"

    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with _config_lock:
            if _writer is not None:
                _writer.close()
            _writer = TelemetryWriter(path, role, end_id)
        logger.info(f"telemetry enabled: writing to {path}")
        return _writer
    except Exception as e:  # never let telemetry setup break a run
        logger.warning(f"telemetry disabled (configure failed): {e}")
        _writer = None
        return None


def emit(event: str, **fields: Any) -> None:
    """Emit one event. No-op when telemetry is disabled. Never raises."""
    w = _writer
    if w is None:
        return
    try:
        w.emit(event, **fields)
    except Exception as e:  # telemetry must never crash the experiment
        logger.debug(f"telemetry emit failed for {event}: {e}")


def is_enabled() -> bool:
    return _writer is not None


def get_run_dir() -> Optional[str]:
    w = _writer
    return os.path.dirname(w.path) if w is not None else None


def shutdown() -> None:
    global _writer
    with _config_lock:
        if _writer is not None:
            _writer.close()
            _writer = None
