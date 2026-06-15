# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Oracle selector: the streaming-misprioritization performance ceiling.

Practical selectors (OORT/REFL/Felix/FedDance) rank clients by a *stale*
statistical utility -- the value a client reported the last time it trained, on
the data it had unlocked back then. Under streaming data that belief drifts away
from the client's *true current* utility, so selection mis-prioritizes.

The oracle removes that staleness. Run it in the async stack with
``c = num_trainers`` and a large ``evalGoalFactor`` so **every** client is
evaluated on the *current* global model each round; its reported
``PROP_STAT_UTILITY`` is therefore fresh (≈ true utility) for all candidates.
The oracle then selects the top ``aggGoal`` purely by that fresh utility -- no
temporal/exploration bonus, no speed penalty, no overcommitment. This is the
"selector that knows the exact client utilities and picks the true top-K" used
as the Claim-3 upper bound; an offline replay (scripts/analysis/
oracle_misselection.py) cross-checks that its picks match the ground-truth top-K.
"""

import logging

from flame.selector.async_oort import AsyncOortSelector
from flame.selector.properties import PROP_END_ID, PROP_UTILITY

logger = logging.getLogger(__name__)


class OracleSelector(AsyncOortSelector):
    """AsyncOort with staleness/exploration/speed terms stripped out.

    Reuses all of AsyncOortSelector's concurrency / eval-scheduling / send-recv
    machinery; only the *ranking* is changed to greedy-by-true-utility.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pure exploitation: never inject random exploration picks.
        self.exploration_factor = 0.0
        self.exploration_factor_decay = 1.0
        self.min_exploration_factor = 0.0
        # No speed-based overcommitment: pick exactly the true top-K.
        self.overcommitment = 1.0
        self.num_of_ends = int(self.agg_goal * self.overcommitment)
        logger.info(
            "OracleSelector active: greedy top-K by fresh (true) utility, "
            "no temporal/system/exploration terms."
        )

    def calculate_total_utility(self, utility_list, ends, model_version):
        """Score == fresh statistical utility (true utility under eval-all).

        Deliberately drops the (stat + temporal) * system_util combination of the
        parent so ranking reflects only the client's true current utility. Keeps
        ``_audit_components`` populated (believed_I) so the existing selection
        plots / offline audit keep working.
        """
        if not utility_list:
            return []

        if getattr(self, "_audit_round", None) != model_version:
            self._audit_components = {}
            self._audit_round = model_version

        for entry in utility_list:
            stat_utility = entry[PROP_UTILITY]
            self._audit_components[entry[PROP_END_ID]] = {
                "believed_I": stat_utility,
                "temporal": 0.0,
                "system_util": 1.0,
            }
            # score is already the (fresh) stat utility; leave PROP_UTILITY as-is.

        return sorted(utility_list, key=lambda x: x[PROP_UTILITY])
