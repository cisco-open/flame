# Copyright 2022 Cisco Systems, Inc. and its affiliates
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
"""horizontal FL top level aggregator."""

import logging
import os
import time
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.config import Config
from flame.datasamplers import datasampler_provider
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.optimizers import optimizer_provider
from flame.plugin import PluginManager, PluginType
from flame.registries import registry_provider
from flame.monitor.runtime import timer_decorator, FwdLLMStage
from flame.selector.properties import (
    PROP_LOCAL_ACCURACY,
    PROP_ROUND_DURATION,
    PROP_ROUND_END_TIME,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
)
from flame import telemetry
from flame.telemetry.events import build_agg_eval, build_agg_round
from flame.sim import VirtualClock, SimReorderBuffer
from flame.selector.properties import PROP_SIM_SEND_TS, PROP_SIM_COMPLETION_TS

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
TAG_HEARTBEAT = "heartbeat_recv"

# Simulated-mode receive bounds (sync): how long to keep draining selected ends
# before committing, and the per-probe wait. In simulated mode trainers do not
# sleep, so available responders land in a tiny physical window; we collect them
# and commit the first_k with the smallest sim_completion_ts (the k that would
# finish first in real mode), independent of physical arrival jitter.
# Real MQTT delivery overhead (agg→trainer + trainer→agg) expected in both real
# and sim (localhost). Added to budget_s before firing [TIMING_OVERRUN_AGG].
_NETWORK_SLACK_S = 2.0
SYNC_SIM_RECV_DEADLINE_S = 30
SYNC_SIM_RECV_FILL_TIMEOUT_S = 0.5

# Startup join barrier: how long to wait for the trainer cohort to join before
# the first selection (see _await_min_trainers). Bounded so a crashed/slow
# trainer can't deadlock startup.
MIN_TRAINERS_JOIN_TIMEOUT_S = 180


class TopAggregator(Role, metaclass=ABCMeta):
    """Top level Aggregator implements an ML aggregation role."""

    @abstract_attribute
    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset(self):
        """
        Abstract attribute for datset.

        dataset's type is Dataset (in flame/dataset.py).
        """

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # Optional deterministic seeding for real/sim parity. The selector runs
        # in this (the aggregator) process and draws from the process-global
        # np.random / random RNGs, so seeding here makes selection reproducible
        # across runs/modes (given identical decision-point ordering). Also
        # seeds torch for reproducible model init. seed=None (default) preserves
        # the legacy unseeded behaviour.
        _seed = getattr(self.config.hyperparameters, "seed", None)
        if _seed is not None:
            import random as _random

            _seed = int(_seed)
            np.random.seed(_seed)
            _random.seed(_seed)
            try:
                import torch as _torch

                _torch.manual_seed(_seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(_seed)
            except Exception:
                pass
            logger.info(f"[SEED] aggregator seeded RNGs with seed={_seed}")

        # global variable for plugin manager
        self.plugin_manager = PluginManager()
        logger.info("Intializing Channel Manager in Top Aggregator for SYNC")
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config)

        base_model = self.config.base_model
        if base_model and base_model.name != "" and base_model.version > 0:
            self.model = self.registry_client.load_model(
                base_model.name, base_model.version
            )

        self.registry_client.setup_run()
        self.metrics = dict()

        # disk cache is used for saving memory in case model is large
        # automatic eviction of disk cache is disabled with cull_limit
        # 0
        self.cache = Cache()
        self.cache.reset("size_limit", 1e15)
        self.cache.reset("cull_limit", 0)

        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).aggregator_data_sampler

        self._round = 1
        self._rounds = 1
        self._rounds = self.config.hyperparameters.rounds
        self._work_done = False

        # Simulation time mode (replaces speedup_factor). "simulated": order
        # updates by a virtual clock fed by trainer-reported completion times;
        # "real": order by physical arrival (legacy/authentic baseline).
        self.time_mode = getattr(
            self.config.hyperparameters, "time_mode", "simulated"
        )
        self.simulated = self.time_mode == "simulated"
        self._vclock = VirtualClock()

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        self._trainers_used_in_curr_round = []
        self.agg_start_time_ts = time.time()

        self._updates_recevied = {}

        self._agg_training_stats = {}
        self._round_update_stat_keys = [
            "staleness",
            "stat_utility",
            "trainer_speed",
        ]
        self._round_update_values = {key: [] for key in self._round_update_stat_keys}
        # TODO Add "wt_contrib_stats" as a key later but cannot
        # directly populate it here since it is only in the optimizer.
        # For now, do it in post-proc script.

    def _compute_aggregator_stats(self) -> None:
        for key in self._round_update_stat_keys:
            raw_values = self._round_update_values.get(key, [])
            if raw_values is None:
                self._agg_training_stats[key] = {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }
                continue

            # Filter out None values
            values = [v for v in raw_values if v is not None]

            if not values:
                self._agg_training_stats[key] = {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }
                continue

            arr = np.array(values, dtype=float)
            self._agg_training_stats[key] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
            }

    def _reset_aggregator_stats(self) -> None:
        self._per_round_update_list = []
        for key in self._round_update_stat_keys:
            self._round_update_values[key] = []

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        logger.debug(f"Invoking get() with tag {tag}")
        if tag == TAG_AGGREGATE:
            logger.debug(
                f"In get(), got message for tag {tag},"
                f"invoking _aggregate_weights({tag})"
            )
            self._aggregate_weights(tag)
        elif tag == TAG_HEARTBEAT:
            logger.debug(
                f"In get(), got message for tag {tag},"
                f" will invoke _read_heartbeat({tag})"
            )
            self._read_heartbeat(tag)

    def _read_heartbeat(self, tag: str) -> None:
        logger.debug("In syncfl _read_heartbeat()")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug("No channel found for read_heartbeat")
            return

        logger.debug(f"Channel {channel} found for _read_heartbeat and tag {tag}")
        logger.debug(f"channel.ends(): {channel.ends()}")
        # receive heartbeat message from trainers TODO: (DG) Check if
        # it processes all heartbeats at once before proceeding to the
        # next sampling?
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if MessageType.HEARTBEAT in msg:
                heartbeat_timestamp = msg[MessageType.HEARTBEAT]
                logger.debug(
                    f"received heartbeat from {end} "
                    f"at timestamp {heartbeat_timestamp}"
                )
            else:
                logger.warm(
                    f"Tried to read message in _read_heartbeat()"
                    f"but got message of type {msg}"
                )

    def _sync_sim_recv_first_k(self, channel, ends, first_k):
        """Simulated mode: commit the first_k updates with the SMALLEST
        sim_completion_ts (the k that would physically finish first in real),
        independent of arrival jitter, and advance the virtual clock to the
        k-th smallest. Returns an ascending-sct list of (msg, metadata); also
        stamps each committed end's PROP_ROUND_DURATION from SIM_ROUND_DURATION
        so OORT/REFL see the correct simulated speed.

        Sync aggregation is order-independent (weighted average), so parity only
        requires the right *set* of k committers and the round duration.
        """
        buf = SimReorderBuffer()
        ends = list(ends)
        deadline = time.time() + SYNC_SIM_RECV_DEADLINE_S
        while len(buf) < len(ends) and time.time() < deadline:
            pending = [e for e in ends if not buf.has(e) and channel.has(e)]
            if not pending:
                break
            got_any = False
            for msg, md in channel.recv_fifo(
                pending, first_k=len(pending),
                timeout=SYNC_SIM_RECV_FILL_TIMEOUT_S,
            ):
                if not msg:
                    break  # per-probe timeout: nothing more ready this pass
                end = md[0]
                sct = msg.get(MessageType.SIM_COMPLETION_TS)
                sct = float(sct) if sct is not None else self._vclock.now
                buf.add(end, sct, (msg, md))
                got_any = True
            # Once we have at least k and a pass added nothing new, the rest are
            # unavailable/quiet — stop waiting (real mode would time them out).
            if len(buf) >= first_k and not got_any:
                break

        committed = []
        for _ in range(min(first_k, len(buf))):
            popped = buf.pop_min()
            if popped is None:
                break
            end, sct, (msg, md) = popped
            self._vclock.advance(sct)
            _sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
            _srd = msg.get(MessageType.SIM_ROUND_DURATION)
            if _srd is not None:
                channel.set_end_property(end, PROP_ROUND_DURATION,
                                         timedelta(seconds=float(_srd)))
            elif _sst is not None:
                channel.set_end_property(end, PROP_ROUND_DURATION,
                                         timedelta(seconds=max(0.0, sct - float(_sst))))
            logger.info(
                f"[SYNC_SIM_RECV] committed {end[-4:]} sct={sct:.1f} "
                f"T_v={self._vclock.now:.1f}"
            )
            committed.append((msg, md))
        return committed

    def _aggregate_weights(self, tag: str) -> None:
        logger.info("Agg weights inside top_aggregator syncfl")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # For REFL/Oort with overcommitment: wait for aggGoal responses, not all selected
        agg_goal = self.config.hyperparameters.aggregation_goal
        first_k = agg_goal if agg_goal and agg_goal > 0 else 0

        # RECV state: receive from the in-flight set we already sent to. A
        # buffered selector (random) returns its selected_ends here rather than
        # picking new trainers (SEND would return none once concurrency is full,
        # stalling aggregation); stateless selectors ignore the state.
        # ends() can be None transiently before selections populate (notably in
        # simulated mode where distribute/aggregate run back-to-back) — skip and
        # retry rather than crash on len(None).
        ends = channel.ends(VAL_CH_STATE_RECV)
        if not ends:
            time.sleep(0.5)
            return
        logger.info(
            f"Waiting for first_k={first_k} responses from {len(ends)} selected trainers"
        )

        # simulated: commit k-smallest-sim_completion_ts (reorder by sim time);
        # real: commit the first_k by physical arrival (authentic baseline).
        if self.simulated:
            _resolved_k = first_k if first_k > 0 else len(ends)
            updates = self._sync_sim_recv_first_k(channel, ends, _resolved_k)
        else:
            updates = channel.recv_fifo(ends, first_k=first_k)

        # receive local model parameters from trainers
        for msg, metadata in updates:
            end, timestamp = metadata
            _t_msg_start = datetime.now()  # start of per-message processing (vii)
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.info(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            # Send→recv lag: mirrors asyncFL's [SEND_RECV_LAG] so the same
            # post-processing/plots work for both sync and async baselines.
            _send_prop = channel.get_end_property(end, PROP_ROUND_START_TIME)
            if _send_prop is not None:
                # PROP_ROUND_START_TIME is stored as (round, datetime)
                _sent_ts = _send_prop[1] if isinstance(_send_prop, tuple) else _send_prop
                recv_ts = timestamp if isinstance(timestamp, datetime) else datetime.now()
                wall_lag_s = (recv_ts - _sent_ts).total_seconds()
                logger.info(
                    f"[SEND_RECV_LAG] end={end} version={self._round} "
                    f"wall_lag_s={wall_lag_s:.3f}"
                )
                # Full per-message lag decomposition into 6 components.
                _wst = msg.get(MessageType.WALL_SEND_TS)   # trainer send (float unix)
                _wrt = msg.get(MessageType.WALL_RECV_TS)   # trainer recv of agg weights (float unix)
                _rcs = msg.get(MessageType.ROUND_COMPUTE_S) # modeled compute duration (float s)
                _agg_sent_unix = _sent_ts.timestamp() if hasattr(_sent_ts, "timestamp") else None
                _agg_recv_unix = recv_ts.timestamp() if hasattr(recv_ts, "timestamp") else None
                _agg_to_trainer = f"{float(_wrt) - _agg_sent_unix:.3f}" if (_wrt and _agg_sent_unix) else "-"
                _compute = f"{float(_rcs):.3f}" if _rcs is not None else "-"
                _post_wait = f"{float(_wst) - float(_wrt) - float(_rcs):.3f}" if (_wst and _wrt and _rcs is not None) else "-"
                _mqtt_lag = f"{_agg_recv_unix - float(_wst):.3f}" if (_wst and _agg_recv_unix) else "-"
                _queue_wait = f"{(_t_msg_start - recv_ts).total_seconds():.3f}"
                _process = f"{(datetime.now() - _t_msg_start).total_seconds():.3f}"
                logger.info(
                    f"[LAG_DECOMP] end={end} version={self._round} "
                    f"wall_lag_s={wall_lag_s:.3f} "
                    f"agg_to_trainer_s={_agg_to_trainer} "
                    f"compute_s={_compute} "
                    f"post_wait_s={_post_wait} "
                    f"mqtt_lag_s={_mqtt_lag} "
                    f"queue_wait_s={_queue_wait} "
                    f"process_s={_process}"
                )
                _budget_s = float(msg.get(MessageType.TRAINING_BUDGET_S, 0.0))
                if _budget_s > 0:
                    if self.simulated:
                        _sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
                        _sct = msg.get(MessageType.SIM_COMPLETION_TS)
                        if _sst is not None and _sct is not None:
                            _virt_elapsed = float(_sct) - float(_sst)
                            if _virt_elapsed > _budget_s:
                                logger.warning(
                                    f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={self._round} "
                                    f"budget={_budget_s:.1f}s overrun: "
                                    f"virtual_elapsed={_virt_elapsed:.2f}s "
                                    f"(excess={_virt_elapsed - _budget_s:.2f}s). "
                                    f"Reduce trainers-per-GPU or add GPUs."
                                )
                    else:
                        if wall_lag_s > _budget_s + _NETWORK_SLACK_S:
                            logger.warning(
                                f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={self._round} "
                                f"budget={_budget_s:.1f}s+slack={_NETWORK_SLACK_S:.1f}s "
                                f"overrun: wall_lag={wall_lag_s:.2f}s "
                                f"(excess={wall_lag_s - _budget_s - _NETWORK_SLACK_S:.2f}s). "
                                f"Reduce trainers-per-GPU or add GPUs."
                            )

            logger.debug(f"received message in agg_weights {msg} from {end}")

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA],
                    end,
                    channel,
                )

            stat_utility = 0
            if MessageType.STAT_UTILITY in msg:
                channel.set_end_property(
                    end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
                )
                stat_utility = msg[MessageType.STAT_UTILITY]

            if MessageType.LOCAL_ACCURACY in msg:
                channel.set_end_property(
                    end, PROP_LOCAL_ACCURACY, msg[MessageType.LOCAL_ACCURACY]
                )

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                self.cache[end] = tres

                if channel._selector is not None:
                    channel._selector.on_update_received(end, msg, self._round)

                update_staleness_val = self._round - tres.version

                # Populate round statistics vars
                self._round_update_values["staleness"].append(update_staleness_val)
                self._round_update_values["stat_utility"].append(stat_utility)
                # PROP_ROUND_DURATION is only populated by the Oort stack; on the
                # base (fedavg / feddance) flow it's unset -> guard against None.
                _rd = channel.get_end_property(end_id=end, key=PROP_ROUND_DURATION)
                self._round_update_values["trainer_speed"].append(
                    _rd.total_seconds() if _rd is not None else 0.0
                )

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        if telemetry.is_enabled():
            agg_obs = {}
            for eid in list(self.cache):
                _rd = channel.get_end_property(end_id=eid, key=PROP_ROUND_DURATION)
                if _rd is not None:
                    agg_obs[eid] = _rd.total_seconds() if hasattr(_rd, "total_seconds") else _rd
            ev, fields = build_agg_round(
                round_num=self._round,
                in_flight=len(channel.ends(VAL_CH_STATE_RECV) or []),
                staleness=list(self._round_update_values.get("staleness", [])),
                stat_utility=list(self._round_update_values.get("stat_utility", [])),
                trainer_speed_s=list(
                    self._round_update_values.get("trainer_speed", [])
                ),
                contributing_trainers=list(self.cache),  # diskcache iterates keys
                agg_observed_s=agg_obs or None,
            )
            telemetry.emit(ev, **fields)

        self._compute_aggregator_stats()
        if self._round % 5 == 0:
            logger.info(f"_agg_training_stats: {self._agg_training_stats}")
        self._reset_aggregator_stats()

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights),
            self.cache,
            total=total,
            num_trainers=len(channel.ends(VAL_CH_STATE_RECV) or []),
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        self.weights = global_weights
        self._update_model()

        if channel._selector is not None:
            channel._selector.on_round_completed(channel._ends, self._round)

    def put(self, tag: str, task_to_perform: str = "train") -> None:
        """Set data to remote role(s)."""
        logger.info(
            f"Sync distributing weights with task_to_perform = {task_to_perform}"
        )
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag, task_to_perform)

    def _await_min_trainers(self, channel) -> None:
        """One-shot startup barrier: block until ``min_trainers_to_start`` ends
        have joined the channel before the first selection.

        Trainers are real processes that spawn + join over wall-clock time in
        BOTH real and simulated mode — simulated only virtualizes training
        *sleeps*, not process startup. Without this barrier, simulated mode races
        through the early rounds before the cohort finishes joining, so the
        selector picks from a partially-joined pool and selection diverges from
        real (which, pacing at true speed, sees the full pool by then). Waiting
        for the same join threshold in both modes makes the candidate set — and
        hence the seeded selection — match. Bounded by a timeout so a crashed or
        slow trainer cannot deadlock startup; runs once (it is a startup-only
        concern, and gating every round would stall on any mid-run dropout)."""
        if getattr(self, "_join_barrier_done", False):
            return
        min_start = getattr(self.config.hyperparameters, "min_trainers_to_start", None)
        if not min_start or int(min_start) <= 0:
            self._join_barrier_done = True
            return
        min_start = int(min_start)
        # Allow override: large cohorts (e.g. n300 at sleep_between_spawns=1s take
        # ~5 min to all spawn/join) need a longer wait than the default.
        timeout_s = float(getattr(
            self.config.hyperparameters, "min_trainers_join_timeout_s",
            MIN_TRAINERS_JOIN_TIMEOUT_S))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            n = len(channel._ends)
            if n >= min_start:
                logger.info(f"[JOIN_BARRIER] {n}/{min_start} trainers joined; starting")
                self._join_barrier_done = True
                return
            logger.info(f"[JOIN_BARRIER] waiting for {min_start} trainers to join; have {n}")
            time.sleep(1.0)
        logger.warning(
            f"[JOIN_BARRIER] timed out after {timeout_s:.0f}s; "
            f"proceeding with {len(channel._ends)}/{min_start} trainers"
        )
        self._join_barrier_done = True

    @timer_decorator
    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        # data_id / iteration_per_data_id are FwdLLM-only; default them so
        # non-FwdLLM aggregators (fedavg, feddance) on this base stack don't
        # AttributeError here.
        self.fwd_llm_stage = FwdLLMStage(
            self._round,
            getattr(self, "data_id", 0),
            getattr(self, "iteration_per_data_id", 0),
        )

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()
        # then wait for the configured cohort so real/sim select from the same pool
        self._await_min_trainers(channel)

        # before distributing weights, update it from global model
        self._update_weights()

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )
        # SEND state: pick (new) trainers to send the model to. With a buffered
        # selector (random) this fills concurrency; stateless selectors ignore
        # the state and return their normal selection.
        selected_ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        if not selected_ends:
            # ends() can be None/empty before trainers join + get selected
            # (notably in simulated mode where the loop spins without sleeps).
            time.sleep(0.5)
            return
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        for idx, end in enumerate(selected_ends):
            logger.info(
                f"sending weights to {end} with model_version: {self._round} for task: {task_to_perform}"
            )
            msg = {
                MessageType.WEIGHTS: weights_to_device(self.weights, DeviceType.CPU),
                MessageType.ROUND: self._round,
                MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                MessageType.MODEL_VERSION: self._round,
                MessageType.TASK_TO_PERFORM: task_to_perform,
            }
            # simulated mode: stamp the virtual send time so the trainer can
            # report sim_completion_ts = sim_send_ts + D back to us.
            if self.simulated:
                sim_send_ts = self._vclock.now
                msg[MessageType.SIM_SEND_TS] = sim_send_ts
                channel.set_end_property(end, PROP_SIM_SEND_TS, sim_send_ts)
            channel.send(end, msg)
            # register round start time on each end for round duration
            # measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

            # Add small delay between sends to distribute MQTT broker load
            # This prevents overwhelming the broker with many concurrent large messages
            # and allows the event loop to process keepalive packets
            if idx < len(selected_ends) - 1:  # Don't sleep after last send
                time.sleep(0.5)

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})
        logger.debug("done broadcasting end-of-training")

    def run_analysis(self):
        """Run analysis plugins and update results to metrics."""
        logger.debug("running analyzer plugins")

        plugins = self.plugin_manager.get_plugins(PluginType.ANALYZER)
        for plugin in plugins:
            # get callback function and call it
            func = plugin.callback()
            metrics = func(self.model, self.dataset)
            if not metrics:
                continue

            self.update_metrics(metrics)

    def save_metrics(self):
        """Save metrics in a model registry."""
        # update metrics with metrics from metric collector
        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")
        self.metrics = dict()

    def increment_round(self):
        """Increment the round counter."""
        self._trainers_used_in_curr_round = []
        logger.debug(
            f"Incrementing current round: {self._round} and "
            f"cleared self._trainers_used_in_curr_round "
            f"{self._trainers_used_in_curr_round}"
        )
        self._round += 1
        self._work_done = self._round > self._rounds

        # Optional runtime cap: stop once max_runtime_s has elapsed.
        # In simulated mode use the virtual clock (vclock_now = simulated seconds
        # elapsed) so the run covers max_runtime_s of *virtual* time, not wall
        # time. In real mode use wall-clock elapsed.
        _max_rt = getattr(self.config.hyperparameters, "max_runtime_s", None)
        # sim_wall_ceiling_s: tight wall-clock guard for sim mode (iii-b).
        # A sim run should finish in <= max_runtime_s wall (it runs faster than
        # real when the parity bug is fixed). Default = max_runtime_s (1×).
        # Separate from max_wall_runtime_s (kept for backward compat, used as
        # secondary fallback if sim_wall_ceiling_s is absent).
        _sim_wall_ceil = getattr(self.config.hyperparameters, "sim_wall_ceiling_s", None)
        _max_wall_rt = getattr(self.config.hyperparameters, "max_wall_runtime_s", None)
        if _max_rt:
            if self.simulated and hasattr(self, "_vclock"):
                elapsed = float(self._vclock.now)
                clock_label = "sim"
            else:
                elapsed = time.time() - self.agg_start_time_ts
                clock_label = "wall"
            if elapsed > float(_max_rt):
                logger.info(
                    f"max_runtime_s={_max_rt}s reached ({clock_label}_elapsed={elapsed:.0f}s) "
                    f"at round {self._round}; stopping run."
                )
                self._work_done = True
            # Failsafe: in sim mode the primary check is virtual time.
            # sim_wall_ceiling_s (default = max_runtime_s = 1×) caps the wall time
            # a sim may use — a well-behaved sim finishes in ≤ real-mode wall time.
            if self.simulated and not self._work_done:
                _wall_elapsed = time.time() - self.agg_start_time_ts
                if _sim_wall_ceil:
                    _failsafe_s = float(_sim_wall_ceil)
                elif _max_wall_rt:
                    _failsafe_s = float(_max_wall_rt)
                else:
                    _failsafe_s = float(_max_rt)  # default: 1× virtual budget
                if _wall_elapsed > _failsafe_s:
                    logger.warning(
                        f"[SIM_WALL_CEILING] sim_wall_ceiling={_failsafe_s:.0f}s reached "
                        f"(wall_elapsed={_wall_elapsed:.0f}s, "
                        f"vclock={self._vclock.now:.0f}s, max_runtime_s={_max_rt}s) "
                        f"at round {self._round}. "
                        f"Sim is slower than real — investigate per-round parity (bug iii-c). "
                        f"Stopping run."
                    )
                    self._work_done = True

        # Periodic virtual-clock progress log (sim mode only).
        # sim_rate = vclock/wall (virtual-seconds per wall-second; < 1 when sim is slow).
        # wall_speedup is computed post-hoc in compare_clock_parity.py as real_wall/sim_wall
        # for matched virtual time — that is the true "sim is faster/slower than real" measure.
        if self.simulated and hasattr(self, "_vclock"):
            _now = getattr(self, "_last_vclock_log_wall_ts", 0.0)
            if time.time() - _now >= 30.0:
                _wall_e = time.time() - self.agg_start_time_ts
                _v = float(self._vclock.now)
                _sim_rate = _v / _wall_e if _wall_e > 0 else 0.0
                logger.info(
                    f"[VCLOCK_PROGRESS] vclock={_v:.1f}s wall={_wall_e:.1f}s "
                    f"sim_rate={_sim_rate:.3f} (virtual-s/wall-s) round={self._round}"
                )
                self._last_vclock_log_wall_ts = time.time()

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        logger.debug(f"Incremented round to {self._round}")
        # set necessary properties to help channel decide how to
        # select ends
        channel.set_property("round", self._round)

    def save_params(self):
        """Save hyperparamets in a model registry."""
        if self.config.hyperparameters:
            self.registry_client.save_params(self.config.hyperparameters)

    def save_model(self):
        """Save model in a model registry."""
        if self.model:
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)

    def save_round_checkpoint(self):
        """Periodically checkpoint the global model for the offline oracle.

        Writes a plain ``state_dict`` (decoupled from the model class location)
        tagged with round + sim/wall time to ``<run>/checkpoints/`` (sibling of
        the ``telemetry/`` dir). The post-run ``oracle_misselection.py`` script
        recomputes each trainer's *true* current utility on its
        deterministically-unlocked data from these checkpoints, so we can
        measure mis-selection against the selector's stale belief.

        Self-contained (lazy config load) so it works identically whether the
        async stack overrides ``internal_init`` or not. Gated by the
        ``checkpoint`` hyperparameter and a no-op unless enabled. Telemetry-grade:
        must never break the training loop.
        """
        try:
            if not getattr(self, "_checkpoint_cfg_loaded", False):
                ckpt_cfg = (
                    getattr(self.config.hyperparameters, "checkpoint", None) or {}
                )
                self._checkpoint_enabled = (
                    str(ckpt_cfg.get("enabled", "False")) == "True"
                )
                self._checkpoint_every_n = int(
                    ckpt_cfg.get("every_n_rounds", 10) or 10
                )
                tdir = os.environ.get("FLAME_TELEMETRY_DIR")
                self._checkpoint_dir = (
                    os.path.join(os.path.dirname(tdir.rstrip("/")), "checkpoints")
                    if tdir
                    else None
                )
                self._checkpoint_cfg_loaded = True

            if not self._checkpoint_enabled or not self._checkpoint_dir:
                return
            if self.model is None or self.framework != MLFramework.PYTORCH:
                return
            if self._checkpoint_every_n > 1 and (
                self._round % self._checkpoint_every_n != 0
            ):
                return

            import torch

            # Aggregator sim-clock at this round: the virtual clock in simulated
            # mode (same clock that stamps trainers' sim_send_ts), else wall
            # elapsed. The oracle feeds this into _visible_sample_count.
            if self.simulated and hasattr(self, "_vclock"):
                sim_time_s = float(self._vclock.now)
            else:
                sim_time_s = float(time.time() - self.agg_start_time_ts)

            os.makedirs(self._checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self._checkpoint_dir, f"round_{self._round:05d}.pt"
            )
            torch.save(
                {
                    "round": int(self._round),
                    "sim_time_s": sim_time_s,
                    "wall_ts": time.time(),
                    "time_mode": self.time_mode,
                    "state_dict": self.model.state_dict(),
                },
                path,
            )
            logger.info(
                f"Saved round checkpoint: {path} (sim_time_s={sim_time_s:.2f})"
            )
        except Exception as e:  # checkpointing must never break training
            logger.warning(f"save_round_checkpoint failed (non-fatal): {e}")

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics
        # Telemetry: aggregator eval metrics (generic hook for all examples).
        if telemetry.is_enabled():
            ev, fields = build_agg_eval(round_num=self._round, metrics=metrics)
            telemetry.emit(ev, **fields)

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def get_curr_unavail_trainers(self) -> list:
        curr_unavail_trainer_list = []

        # Ensure trainer_event_dict exists
        if self.trainer_event_dict is not None:
            # Aggregator time since start, on the SAME timeline the trace's event
            # timestamps live on. In simulated mode that is the virtual clock
            # (sim-seconds); wall-clock would be a few seconds total while the
            # virtual timeline spans the whole trace, making every window look
            # available. Trainer-side availability already keys off _sim_now()
            # (sim_send_ts); this mirrors it for the aggregator-side oracular path.
            agg_time_since_start_s = (
                self._vclock.now if self.simulated
                else time.time() - self.agg_start_time_ts
            )

            for trainer_id, event_dict in list(self.trainer_event_dict.items()):
                logger.debug(
                    f"Checking trainer {trainer_id}'s availability. Event_dict is: {event_dict}"
                )

                if not event_dict:
                    continue  # Skip if no events for trainer

                # Binary search for closest past event
                idx = event_dict.bisect_right(agg_time_since_start_s) - 1
                logger.debug(f"Trainer_id: {trainer_id} got index: {idx}")

                if idx >= 0:
                    most_recent_event = event_dict.peekitem(idx)
                    logger.debug(
                        f"Trainer_id: {trainer_id} got most_recent_event: {most_recent_event}"
                    )

                    most_recent_event_ts = most_recent_event[0]
                    most_recent_event_state = most_recent_event[1]

                    if most_recent_event_state == "UN_AVL":
                        logger.debug(
                            f"Trainer {trainer_id} is unavailable since time {most_recent_event_ts}."
                        )
                        curr_unavail_trainer_list.append(trainer_id)
                    elif most_recent_event_state == "AVL_TRAIN":
                        logger.debug(
                            f"Trainer {trainer_id} is available since time {most_recent_event_ts}."
                        )
                    else:
                        logger.warning(
                            f"Trainer {trainer_id} was in state {most_recent_event_state} since time {most_recent_event_ts}, needs to be handled."
                        )

                # TODO: To be more memory efficient, we can delete
                # events that are way past their time and already used

        # Return the list of currently unavailable trainers
        logger.debug(
            f"Current curr_unavail_trainer_list: {curr_unavail_trainer_list} has {len(curr_unavail_trainer_list)} ends out of total {len(self.trainer_event_dict)} ends dict"
        )

        return curr_unavail_trainer_list

    def compose(self) -> None:
        """Compose role with tasklets."""

        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get_weights = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_get_heartbeat = Tasklet("heartbeat_recv", self.get, TAG_HEARTBEAT)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_analysis = Tasklet("analysis", self.run_analysis)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_end_of_training = Tasklet(
                "inform_end_of_training", self.inform_end_of_training
            )

            task_save_params = Tasklet("save_params", self.save_params)

            task_save_model = Tasklet("save_model", self.save_model)

            task_checkpoint = Tasklet("checkpoint", self.save_round_checkpoint)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_put
                >> task_get_weights
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_checkpoint
                >> task_increment_round
                >> task_get_heartbeat
            )
            >> task_end_of_training
            >> task_save_params
            >> task_save_model
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
