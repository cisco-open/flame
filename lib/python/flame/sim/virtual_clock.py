# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Virtual (simulated) clock + reorder buffer for simulated time mode.

These are pure, dependency-free helpers so they can be unit-tested in
isolation. The async aggregator composes them to order trainer updates by a
*simulated* completion time instead of physical arrival, which is what makes
``simulated`` mode reproduce ``real`` mode's per-round behavior without sleeping.
"""

from __future__ import annotations

from typing import Any, Optional


class VirtualClock:
    """Monotone simulated clock measured in simulated seconds.

    The aggregator stamps `now` on each distributed task (`sim_send_ts`) and
    advances to a committed update's `sim_completion_ts`. It never moves
    backward, so committing updates in ascending completion order yields a
    monotone timeline.
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    @property
    def now(self) -> float:
        return self._t

    def advance(self, ts: float) -> float:
        """Advance to ``ts`` if it is in the future; return the new time."""
        ts = float(ts)
        if ts > self._t:
            self._t = ts
        return self._t

    def reset(self) -> None:
        self._t = 0.0


def sim_ordered_ends(end_to_completion: dict[str, float]) -> list[str]:
    """Return ends sorted by ascending simulated completion time.

    Ties broken by end id for determinism (so ordering is reproducible across
    runs and independent of dict/arrival order).
    """
    return sorted(end_to_completion, key=lambda e: (end_to_completion[e], str(e)))


class SimReorderBuffer:
    """Collects arrived-but-uncommitted updates and pops them in simulated
    completion order.

    In simulated mode trainers do not sleep, so all in-flight updates land in a
    tiny physical window; buffering them and popping the minimum
    `sim_completion_ts` reconstructs the order they *would* have arrived in real
    mode — independent of physical arrival jitter.
    """

    def __init__(self) -> None:
        # end_id -> (sim_completion_ts, payload)
        self._items: dict[str, tuple[float, Any]] = {}

    def add(self, end_id: str, sim_completion_ts: float, payload: Any = None) -> None:
        self._items[end_id] = (float(sim_completion_ts), payload)

    def has(self, end_id: str) -> bool:
        return end_id in self._items

    def pending_ends(self) -> set[str]:
        return set(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def peek_min_ts(self) -> Optional[float]:
        if not self._items:
            return None
        return min(ts for ts, _ in self._items.values())

    def pending_after(self, ts: float) -> set[str]:
        """Buffered ends whose completion time is still in the future (> ``ts``)
        — i.e. modeled as STILL COMPUTING at virtual time ``ts`` (their update has
        arrived physically but is not yet "available" in sim time). Used by the
        sync sim stack to keep such trainers occupying their selection slot /
        out of the eligible pool until ``vclock >= sct``."""
        return {e for e, (sct, _) in self._items.items() if sct > ts}

    def pop_min(self) -> Optional[tuple[str, float, Any]]:
        """Remove and return ``(end_id, sim_completion_ts, payload)`` with the
        smallest completion time (ties by end id). ``None`` if empty."""
        if not self._items:
            return None
        end_id = min(self._items, key=lambda e: (self._items[e][0], str(e)))
        ts, payload = self._items.pop(end_id)
        return end_id, ts, payload

    def discard(self, end_id: str) -> None:
        self._items.pop(end_id, None)

    def clear(self) -> None:
        """Discard all buffered entries (e.g. at round end to prevent cross-round stale commits)."""
        self._items.clear()
