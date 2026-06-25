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
"""Guard the single-sourced real client task-train duration (§S.dur) shared by all
horizontal aggregators (oort/refl, asyncfl/felix, syncfl/feddance+fedavg, fwdllm)."""

from datetime import datetime

from flame.mode.message import MessageType
from flame.mode.horizontal.client_duration import real_client_task_train_duration


def test_prefers_intrinsic_client_stamps_excluding_delivery_lag():
    """WALL_SEND - WALL_RECV (intrinsic compute+sleep), NOT WALL_SEND - dispatch.
    dispatch=t0, client receives weights 8s later (delivery lag), computes 12s,
    sends at t=20; agg reads at t=26 (read-wait). Duration must be 12, isolating
    the client work from both server-side waits."""
    dispatch = datetime(2026, 1, 1, 0, 0, 0)
    msg = {
        MessageType.WALL_RECV_TS: dispatch.timestamp() + 8.0,
        MessageType.WALL_SEND_TS: dispatch.timestamp() + 20.0,
    }
    recv = datetime.fromtimestamp(dispatch.timestamp() + 26.0)
    assert real_client_task_train_duration(msg, dispatch, recv).total_seconds() == 12.0


def test_fallback_wall_send_minus_dispatch_without_recv_stamp():
    dispatch = datetime(2026, 1, 1, 0, 0, 0)
    msg = {MessageType.WALL_SEND_TS: dispatch.timestamp() + 12.0}
    assert real_client_task_train_duration(msg, dispatch, None).total_seconds() == 12.0


def test_fallback_recv_minus_dispatch_without_client_stamps():
    dispatch = datetime(2026, 1, 1, 0, 0, 0)
    recv = datetime.fromtimestamp(dispatch.timestamp() + 18.0)
    assert real_client_task_train_duration({}, dispatch, recv).total_seconds() == 18.0


def test_none_when_unmeasurable():
    assert real_client_task_train_duration({}, None, None) is None


def test_non_positive_intrinsic_falls_through():
    """A clock-skew/degenerate case (send <= recv) must not return <=0; it falls
    through to the dispatch-anchored fallbacks."""
    dispatch = datetime(2026, 1, 1, 0, 0, 0)
    msg = {
        MessageType.WALL_RECV_TS: dispatch.timestamp() + 20.0,
        MessageType.WALL_SEND_TS: dispatch.timestamp() + 20.0,  # zero intrinsic
    }
    recv = datetime.fromtimestamp(dispatch.timestamp() + 25.0)
    # zero intrinsic skipped -> WALL_SEND - dispatch = 20.0
    assert real_client_task_train_duration(msg, dispatch, recv).total_seconds() == 20.0
