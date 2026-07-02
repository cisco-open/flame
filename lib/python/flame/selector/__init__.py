# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""selector abstract class."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import hashlib
import logging
import time
# Import classes directly: bare `import random` here resolves to the sibling
# flame/selector/random.py submodule, not stdlib.
from random import Random as _StdRandom
from numpy.random import RandomState as _NpRandomState

from .. import telemetry
from ..common.typing import Scalar
from ..end import End
from ..telemetry.events import build_selection
from .properties import (
    PROP_AVL_STATE,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_STAT_UTILITY,
)

SelectorReturnType = dict[str, Union[None, Tuple[str, Scalar]]]

logger = logging.getLogger(__name__)


def _round_or_none(v, ndigits: int = 4):
    """Round for the decision fingerprint; pass through None / non-numerics."""
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


class AbstractSelector(ABC):
    """Abstract base class for selector implementation."""

    def __init__(self, **kwargs) -> None:
        # Reserved kwarg (consumed, not setattr'd as a hyperparameter).
        _seed = kwargs.pop("_seed", None)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.selected_ends: set = set()
        self.ordered_updates_recv_ends: list = []
        # Dedicated, seed-able RNGs insulated from the process-global np.random/
        # random. Selectors MUST draw from these (never bare np.random/random) so
        # selection is reproducible across real/sim. seed=None = unseeded (legacy).
        self._seed = _seed
        self._rng = _NpRandomState(_seed)
        self._pyrng = _StdRandom(_seed)
        if _seed is not None:
            logger.info(
                f"[SELECTOR_SEED] {type(self).__name__} dedicated RNGs seeded "
                f"with seed={_seed}"
            )

    def enforce_min_start(self, ends_count: int) -> bool:
        """Return True if selection should wait due to min-start threshold."""
        threshold = (
            int(self.minInitialTrainers)
            if hasattr(self, "minInitialTrainers")
            and self.minInitialTrainers is not None
            else -1
        )
        if ends_count < threshold:
            logger.debug(
                f"Not enough ends to start selection, need at least {threshold}"
            )
            time.sleep(0.1)
            return True
        return False

    @abstractmethod
    def select(
        self, ends: dict[str, End], channel_props: dict[str, Scalar]
    ) -> SelectorReturnType:
        """Abstract method to select ends.

        Parameters
        ----------
        ends: a dictionary whose key is end id and value is End object
        channel_props: properties set in channel

        Returns
        -------
        dictionary: key is end id and value is a property (as tuple)
                    used/created during selection process; value can be none
        """

    def emit_selection(
        self,
        round_num: int,
        task: str,
        ends: dict[str, End],
        eligible_ids,
        chosen_ids,
        per_trainer_extra: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Emit a structured selector-decision event (no-op if telemetry off).

        Centralized here so every selector produces an identical schema, which
        is what makes cross-selector comparison possible. ``ends`` is the full
        candidate pool; availability composition and per-trainer utility/speed
        are derived from end properties.
        """
        if not telemetry.is_enabled():
            return
        try:
            chosen_set = set(chosen_ids)
            avail_composition: dict[str, int] = {}
            per_trainer: dict[str, dict] = {}
            for end_id, end in ends.items():
                state = end.get_property(PROP_AVL_STATE)
                state_name = getattr(state, "value", None) or (
                    str(state) if state is not None else "UNKNOWN"
                )
                avail_composition[state_name] = (
                    avail_composition.get(state_name, 0) + 1
                )
                util = end.get_property(PROP_STAT_UTILITY)
                speed = end.get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
                entry = {
                    "utility": util,
                    "speed_s": speed.total_seconds()
                    if hasattr(speed, "total_seconds")
                    else speed,
                    "selected": end_id in chosen_set,
                    "avl_state": state_name,
                }
                if per_trainer_extra and end_id in per_trainer_extra:
                    entry.update(per_trainer_extra[end_id])
                per_trainer[end_id] = entry

            # in-flight count: selected_ends is a set/list for most selectors,
            # but a {requester: set(ends)} dict for fedbuff-style selectors.
            sel = self.selected_ends
            if isinstance(sel, dict):
                vals = list(sel.values())
                in_flight = (
                    sum(len(v) for v in vals)
                    if vals and all(isinstance(v, (set, list)) for v in vals)
                    else len(sel)
                )
            elif isinstance(sel, (set, list)):
                in_flight = len(sel)
            else:
                in_flight = 0

            # Determinism fingerprints: eligible = candidate set; decision = set +
            # per-candidate utility/speed + k. Same fingerprint but different
            # `chosen` => RNG desync; different fingerprint => input drift.
            elig = sorted(set(eligible_ids))
            elig_fp = hashlib.sha1(
                "|".join(elig).encode()
            ).hexdigest()[:12]
            dec_payload = ";".join(
                f"{e}:{_round_or_none(per_trainer.get(e, {}).get('utility'))}"
                f":{_round_or_none(per_trainer.get(e, {}).get('speed_s'))}"
                for e in elig
            ) + f"#k={len(chosen_set)}"
            dec_fp = hashlib.sha1(dec_payload.encode()).hexdigest()[:12]
            extra = dict(extra or {})
            extra.update({
                "seed": self._seed,
                "eligible_fingerprint": elig_fp,
                "decision_fingerprint": dec_fp,
            })

            ev, fields = build_selection(
                round_num=int(round_num),
                task=task,
                selector=type(self).__name__,
                num_candidates=len(ends),
                num_eligible=len(set(eligible_ids)),
                avail_composition=avail_composition,
                chosen=list(chosen_set),
                in_flight=in_flight,
                per_trainer=per_trainer,
                extra=extra,
            )
            telemetry.emit(ev, **fields)
        except Exception as e:  # telemetry must never break selection
            logger.debug(f"emit_selection failed: {e}")

    def on_update_received(
        self, end_id: str, msg: dict, round_num: int
    ) -> None:
        """Hook: aggregator calls this when a trainer update arrives.

        Default records the end_id for later cleanup. Subclasses override to
        extract per-update metrics (e.g. FedDance pulls LOCAL_ACCURACY).
        """
        if isinstance(self.selected_ends, set):
            self.ordered_updates_recv_ends.append(end_id)

    def on_round_completed(
        self, ends: dict[str, End], round_num: int
    ) -> None:
        """Hook: aggregator calls this after aggregation finishes.

        Default frees received-ends from the in-flight set. Subclasses with
        custom legacy cleanup (_cleanup_recvd_ends) get that called too.
        """
        if isinstance(self.selected_ends, set):
            for end_id in self.ordered_updates_recv_ends:
                self.selected_ends.discard(end_id)
            self.ordered_updates_recv_ends = []
        elif hasattr(self, "_cleanup_recvd_ends"):
            self._cleanup_recvd_ends(ends)
