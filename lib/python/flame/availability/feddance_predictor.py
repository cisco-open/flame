# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDance Poisson availability predictor.

V_m(r) = 1 - exp(-lambda_m(r) * K),
lambda_m(r) = (sum of binary check-ins in [r-K_h, r-1]) / K_h.
"""

import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class FedDancePredictor:
    def __init__(
        self,
        history_window: int = 50,
        prediction_window: int = 5,
        lambda_cold_start: float = 0.5,
    ):
        if history_window <= 0 or prediction_window <= 0:
            raise ValueError("history_window and prediction_window must be positive")
        self.history_window = history_window
        self.prediction_window = prediction_window
        self.lambda_cold_start = lambda_cold_start
        self._checkins: dict[str, deque[int]] = {}
        self._last_round: dict[str, int] = {}

    def record_checkin(self, end_id: str, round_num: int) -> None:
        if end_id not in self._checkins:
            self._checkins[end_id] = deque(maxlen=self.history_window)
        if self._last_round.get(end_id) == round_num:
            return
        self._checkins[end_id].append(round_num)
        self._last_round[end_id] = round_num

    def seed_history(self, end_id: str, rounds_present: list[int]) -> None:
        """Seed historical check-in rounds (e.g. from a trace)."""
        if end_id not in self._checkins:
            self._checkins[end_id] = deque(maxlen=self.history_window)
        for r in rounds_present[-self.history_window:]:
            self._checkins[end_id].append(r)
        if rounds_present:
            self._last_round[end_id] = rounds_present[-1]

    def lambda_m(self, end_id: str, round_num: int) -> float:
        history = self._checkins.get(end_id)
        if not history:
            return self.lambda_cold_start
        lower = round_num - self.history_window
        count = sum(1 for r in history if lower <= r < round_num)
        return count / self.history_window

    def V_m(self, end_id: str, round_num: int) -> float:
        lam = self.lambda_m(end_id, round_num)
        return 1.0 - math.exp(-lam * self.prediction_window)

    def is_tracked(self, end_id: str) -> bool:
        return end_id in self._checkins
