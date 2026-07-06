# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Trainer._fetch_weights logged a debug line that indexed
msg[MessageType.DATA_ID]/msg[MessageType.ITERATION_PER_DATA_ID]
unconditionally, before the existing `if MessageType.DATA_ID in msg and ...`
guard a few lines below it. The aggregator's end-of-training broadcast
(inform_end_of_training in flame/mode/horizontal/syncfl/top_aggregator.py)
sends only {MessageType.EOT: self._work_done} -- no DATA_ID/
ITERATION_PER_DATA_ID keys -- so any trainer still waiting in _fetch_weights
when the aggregator wraps up hit an uncaught KeyError and crashed instead of
setting self._work_done and exiting its loop cleanly. Confirmed via three
real n=100 overnight runs (2026-07-02): every idle/unselected trainer in
each run (82-90 per run) crashed with this exact KeyError at the run's stop
timestamp. Fixed by switching the log line to msg.get(...), matching the
guarded access pattern already used a few lines below (line 267) for the
identical MessageType.WEIGHTS log line.
"""

from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
from flame.mode.message import MessageType


class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    def __init__(self, msg):
        self._msg = msg
        self._selector = _FakeSelector()

    def await_join(self):
        pass

    def one_end(self, state):
        return "end_1"

    def recv(self, end_id):
        return self._msg, None

    def cleanup_recvd_ends(self):
        pass


class _FakeChannelManager:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class _FakeTrainer:
    _fetch_weights = Trainer._fetch_weights

    def __init__(self, channel):
        self.cm = _FakeChannelManager(channel)
        self.trainer_id = "trainer_1"
        self.fetch_success = False
        self._work_done = False
        self.data_id = None
        self.iteration_per_data_id = None
        self._round = 1
        self._model_version = 0


class TestFetchWeightsEndOfTrainingBroadcast:
    def test_eot_only_message_does_not_raise(self):
        channel = _FakeChannel({MessageType.EOT: True})
        trainer = _FakeTrainer(channel)

        trainer._fetch_weights("fetch")

        assert trainer._work_done is True
        assert trainer.fetch_success is True
