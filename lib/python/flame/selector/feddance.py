# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDance participant selector.

U_m(r) = (V_m * I_m * A_m) * (1 + log10(R+1) / (10 * (1 + J_m)))
  V_m: Poisson availability over next K rounds
  I_m: average training loss in last engaged round (Oort stat_utility)
  A_m: slope of training accuracy across last beta engagements
  J_m: last round in which device m was engaged

The selector picks top-N by U_m, with cold-start substitution from the
previous round's mean I, A for devices that have never returned an update.
"""

import logging
import math
import random
from collections import deque
from typing import Optional

from flame.availability.feddance_predictor import FedDancePredictor
from flame.common.typing import Scalar
from flame.end import End
from flame.selector import AbstractSelector, SelectorReturnType
from flame.selector.properties import (
    PROP_A,
    PROP_I,
    PROP_LAST_ENGAGED_ROUND,
    PROP_LOCAL_ACCURACY,
    PROP_SELECTED_COUNT,
    PROP_STAT_UTILITY,
    PROP_U,
    PROP_V,
)

logger = logging.getLogger(__name__)

EPS = 1e-6


class FedDanceSelector(AbstractSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            self.aggr_num = kwargs["aggr_num"]
        except KeyError:
            raise KeyError("aggr_num is not specified in config")

        self.history_window = kwargs.get("history_window", 50)
        self.prediction_window = kwargs.get("prediction_window", 5)
        self.accuracy_window = kwargs.get("accuracy_window", 5)
        self.overcommitment = kwargs.get("overcommitment", 1.0)
        self.negative_accuracy_handling = kwargs.get(
            "negative_accuracy_handling", "relu"
        )
        if self.negative_accuracy_handling not in ("relu", "abs", "raw"):
            raise ValueError(
                f"negative_accuracy_handling must be relu/abs/raw, got {self.negative_accuracy_handling}"
            )

        self.num_of_ends = max(1, int(self.aggr_num * self.overcommitment))

        self.predictor = FedDancePredictor(
            history_window=self.history_window,
            prediction_window=self.prediction_window,
            lambda_cold_start=kwargs.get("lambda_cold_start", 0.5),
        )

        self.accuracy_history: dict[str, deque] = {}
        self.last_loss: dict[str, float] = {}
        self.last_engaged_round: dict[str, int] = {}

        self.prev_round_mean_I: Optional[float] = None
        self.prev_round_mean_A: Optional[float] = None

        self.round = 0
        self.newly_selected_this_round: set = set()

        # Drained in on_round_completed to update prev_round_mean_I/A.
        self._round_loss_buffer: list[float] = []
        self._round_acc_buffer: list[float] = []

        logger.info(
            f"FedDanceSelector init: N={self.aggr_num}, K_h={self.history_window}, "
            f"K={self.prediction_window}, beta={self.accuracy_window}, "
            f"neg_A={self.negative_accuracy_handling}"
        )

    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: Optional[list] = None,
        task_to_perform: str = "train",
        **kwargs,
    ) -> SelectorReturnType:
        num_of_ends = min(len(ends), self.num_of_ends)
        if num_of_ends == 0:
            return {}

        round_num = channel_props.get("round", 0)

        if round_num <= self.round and len(self.newly_selected_this_round) != 0:
            return {key: None for key in self.newly_selected_this_round}

        unavail = set(trainer_unavail_list or [])

        for end_id in ends:
            if end_id not in unavail:
                self.predictor.record_checkin(end_id, round_num)

        eligible = {
            eid: e
            for eid, e in ends.items()
            if eid not in unavail and eid not in self.selected_ends
        }

        if not eligible:
            logger.error(
                f"[FEDDANCE] round {round_num}: no eligible trainers "
                f"(total={len(ends)}, unavail={len(unavail)}, in_flight={len(self.selected_ends)})"
            )
            return {}

        num_to_select = min(num_of_ends, len(eligible))
        if num_to_select < num_of_ends:
            logger.warning(
                f"[FEDDANCE] round {round_num}: only {num_to_select}/{num_of_ends} trainers eligible"
            )

        utilities = self._compute_utilities(eligible, round_num)
        selected = self._pick_top_n(utilities, num_to_select)

        self.newly_selected_this_round = set(selected)
        self.selected_ends = self.selected_ends | self.newly_selected_this_round

        for end_id in selected:
            end = ends[end_id]
            count = end.get_property(PROP_SELECTED_COUNT) or 0
            end.set_property(PROP_SELECTED_COUNT, count + 1)

        logger.info(
            f"[FEDDANCE] round {round_num}: picked {len(selected)} new, "
            f"in-flight total {len(self.selected_ends)}"
        )

        self.round = round_num
        # Log FedDance's believed scoring factors per candidate (V_m, I_m, A_m, U_m)
        # so the offline staleness audit can compare believed-vs-true per factor.
        ptx = {
            eid: {
                "feddance_V": eligible[eid].get_property(PROP_V),
                "feddance_I": eligible[eid].get_property(PROP_I),
                "feddance_A": eligible[eid].get_property(PROP_A),
                "feddance_U": eligible[eid].get_property(PROP_U),
                "last_engaged_round": self.last_engaged_round.get(eid),
            }
            for eid in eligible
        }
        self.emit_selection(
            round_num, task_to_perform, ends, eligible.keys(), selected,
            per_trainer_extra=ptx,
            extra={"num_unavail": len(unavail)},
        )
        return {key: None for key in selected}

    def _compute_utilities(
        self, eligible: dict[str, End], round_num: int
    ) -> list[tuple[str, float]]:
        mean_I = self.prev_round_mean_I if self.prev_round_mean_I is not None else 1.0
        mean_A = self.prev_round_mean_A if self.prev_round_mean_A is not None else 1.0

        utilities = []
        for end_id, end in eligible.items():
            v = self.predictor.V_m(end_id, round_num)

            i = self.last_loss.get(end_id)
            if i is None:
                stat_util = end.get_property(PROP_STAT_UTILITY)
                i = stat_util if stat_util is not None else mean_I

            a = self._accuracy_slope(end_id)
            if a is None:
                a = mean_A
            a = self._handle_negative_a(a)

            base = v * i * a
            ucb = 1.0 + (
                math.log10(round_num + 1) / (10.0 * (1.0 + self.last_engaged_round.get(end_id, 0)))
            )
            u = base * ucb

            end.set_property(PROP_V, v)
            end.set_property(PROP_I, i)
            end.set_property(PROP_A, a)
            end.set_property(PROP_U, u)

            utilities.append((end_id, u))

        return utilities

    def _accuracy_slope(self, end_id: str) -> Optional[float]:
        hist = self.accuracy_history.get(end_id)
        if hist is None or len(hist) < 2:
            return None
        return (hist[-1] - hist[0]) / (len(hist) - 1)

    def _handle_negative_a(self, a: float) -> float:
        if self.negative_accuracy_handling == "relu":
            return max(a, EPS)
        if self.negative_accuracy_handling == "abs":
            return abs(a) if a != 0 else EPS
        return a

    def _pick_top_n(
        self, utilities: list[tuple[str, float]], n: int
    ) -> list[str]:
        if n >= len(utilities):
            return [eid for eid, _ in utilities]
        utilities.sort(key=lambda kv: kv[1], reverse=True)
        return [eid for eid, _ in utilities[:n]]

    def on_update_received(
        self, end_id: str, msg: dict, round_num: int
    ) -> None:
        from flame.mode.message import MessageType

        self.ordered_updates_recv_ends.append(end_id)

        loss = msg.get(MessageType.STAT_UTILITY)
        acc = msg.get(MessageType.LOCAL_ACCURACY)

        if loss is not None:
            try:
                loss_f = float(loss)
            except (TypeError, ValueError):
                loss_f = None
            if loss_f is not None:
                self.last_loss[end_id] = loss_f
                self._round_loss_buffer.append(loss_f)

        if acc is not None:
            try:
                acc_f = float(acc)
            except (TypeError, ValueError):
                acc_f = None
            if acc_f is not None:
                hist = self.accuracy_history.setdefault(
                    end_id, deque(maxlen=self.accuracy_window)
                )
                hist.append(acc_f)
                self._round_acc_buffer.append(acc_f)

        self.last_engaged_round[end_id] = round_num

    def on_round_completed(
        self, ends: dict[str, End], round_num: int
    ) -> None:
        if self._round_loss_buffer:
            self.prev_round_mean_I = sum(self._round_loss_buffer) / len(
                self._round_loss_buffer
            )
        if self._round_acc_buffer:
            self.prev_round_mean_A = sum(self._round_acc_buffer) / len(
                self._round_acc_buffer
            )
        self._round_loss_buffer = []
        self._round_acc_buffer = []

        for end_id in self.ordered_updates_recv_ends:
            self.selected_ends.discard(end_id)
        self.ordered_updates_recv_ends = []
