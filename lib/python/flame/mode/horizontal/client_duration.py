# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the REAL-mode client task-train duration.

The selector speed signal / `trainer_speed` telemetry must be the client's
INTRINSIC task time (compute+sleep), matching sim's
`SIM_CLIENT_TASK_TRAIN_DURATION_S = max(gpu, D)`. Every horizontal aggregator
that tracks it (oort/refl, asyncfl/felix, syncfl/feddance+fedavg) calls THIS
function so the definition can't drift between baselines. See PARITY.md §S.dur.
"""

from datetime import datetime, timedelta
from typing import Optional

from flame.mode.message import MessageType


def real_client_task_train_duration(
    msg: dict, dispatch_ts=None, recv_ts=None
) -> Optional[timedelta]:
    """Client's intrinsic task-train duration = ``WALL_SEND_TS - WALL_RECV_TS``
    (both stamped on the TRAINER, bracketing compute+sleep). Returns a timedelta,
    or None when it can't be measured.

    It EXCLUDES BOTH server-side waits by anchoring on the two CLIENT stamps:
      (1) aggregator read-wait (``recv_ts`` side) — a stale straggler's finished
          update sits unread until a later round drains the reorder buffer;
      (2) dispatch->recv delivery lag (``WALL_RECV_TS - dispatch``) — the agg stamps
          ``dispatch`` at selection, but a slow client (held one-in-flight) receives
          the weights later, so ``WALL_SEND - dispatch`` adds that lag (≈8s for the
          slowest clients).
    Neither is client device speed; folding either in inflates slow-trainer
    durations, raises the Oort selector's ``round_preferred_duration`` percentile,
    and makes real under-penalize slow clients vs sim (the oort+refl K2 root). The
    trainer measures its own duration as exactly D (``wall_send - wall_recv ==
    budget`` via sleep), matching sim. Falls back to the dispatch-anchored measures
    (which carry the lag) only when a client stamp is absent. See PARITY.md §S.dur /
    project_oort_a2c_root.
    """
    wst = msg.get(MessageType.WALL_SEND_TS)
    wrt = msg.get(MessageType.WALL_RECV_TS)
    # Primary: both client stamps -> intrinsic compute+sleep, server-wait-free.
    if wst is not None and wrt is not None:
        dur = float(wst) - float(wrt)
        if dur > 0:
            return timedelta(seconds=dur)
    # Fallback 1: WALL_SEND - dispatch (carries the delivery lag; pre-§S.dur path).
    if wst is not None and hasattr(dispatch_ts, "timestamp"):
        dur = float(wst) - dispatch_ts.timestamp()
        if dur > 0:
            return timedelta(seconds=dur)
    # Fallback 2: recv - dispatch (also carries read-wait; last resort).
    if isinstance(recv_ts, datetime) and isinstance(dispatch_ts, datetime):
        dur = (recv_ts - dispatch_ts).total_seconds()
        if dur > 0:
            return timedelta(seconds=dur)
    return None
