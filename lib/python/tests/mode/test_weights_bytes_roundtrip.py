# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the lazy-deserialize weight encoding (WEIGHTS_BYTES).

Sim/real trainers ship a model update as pre-serialized raw bytes
(``MessageType.WEIGHTS_BYTES``) rather than a live tensor, so the aggregator
reconstructs the tensor only for the updates it commits (not the surplus/stale
ones it discards). This regression bit twice:

  * oort sync aggregator raised ``UnboundLocalError`` on the first update
    because ``weights`` was only bound inside ``if MessageType.WEIGHTS in msg``;
  * the async aggregator's train-vs-eval router keyed off ``WEIGHTS in msg`` and
    misrouted a (bytes-carrying) train update to the eval branch, so it never
    aggregated and never logged accuracy.

These tests pin the two invariants that prevent recurrence:
  1. ``materialize_weights`` recovers the original weights from WEIGHTS_BYTES,
     is backward compatible with a live-tensor message, idempotent, and returns
     None for a weight-less (eval-only) message.
  2. A train update is recognized as a model update whether it carries WEIGHTS
     or WEIGHTS_BYTES (the classifier predicate the async aggregator uses).
"""

import cloudpickle

from flame.common.util import materialize_weights
from flame.mode.message import MessageType


def _is_model_update(msg):
    """The predicate the aggregator must use to tell a train update (which also
    carries STAT_UTILITY) from an eval-only update."""
    return MessageType.WEIGHTS in msg or MessageType.WEIGHTS_BYTES in msg


def test_materialize_recovers_weights_from_bytes():
    orig = {"layer.w": [1, 2, 3], "layer.b": [4.0]}
    msg = {
        MessageType.WEIGHTS_BYTES: cloudpickle.dumps(orig),
        MessageType.STAT_UTILITY: 2.3,
        MessageType.MODEL_VERSION: 7,
    }
    got = materialize_weights(msg)
    assert got == orig
    # converted in place: WEIGHTS now present, BYTES consumed
    assert msg[MessageType.WEIGHTS] == orig
    assert MessageType.WEIGHTS_BYTES not in msg


def test_materialize_is_backward_compatible_and_idempotent():
    orig = {"w": [1.0]}
    # a live-tensor message is returned untouched
    msg = {MessageType.WEIGHTS: orig}
    assert materialize_weights(msg) is orig
    # second call on an already-materialized message is a no-op
    msg2 = {MessageType.WEIGHTS_BYTES: cloudpickle.dumps(orig)}
    assert materialize_weights(msg2) == orig
    assert materialize_weights(msg2) == orig


def test_materialize_returns_none_for_eval_only_message():
    # eval-only update carries STAT_UTILITY but no weights in any encoding
    assert materialize_weights({MessageType.STAT_UTILITY: 1.0}) is None


def test_train_update_classified_as_model_update_with_bytes():
    # a TRAIN update now carries WEIGHTS_BYTES *and* STAT_UTILITY; it must read as
    # a model update, not get misrouted to the eval branch.
    train = {MessageType.WEIGHTS_BYTES: b"x", MessageType.STAT_UTILITY: 1.0}
    assert _is_model_update(train) is True
    # an EVAL update (no weights) must NOT read as a model update
    eval_only = {MessageType.STAT_UTILITY: 1.0}
    assert _is_model_update(eval_only) is False
    # a legacy live-tensor train update still reads as a model update
    legacy = {MessageType.WEIGHTS: {"w": [1.0]}, MessageType.STAT_UTILITY: 1.0}
    assert _is_model_update(legacy) is True
