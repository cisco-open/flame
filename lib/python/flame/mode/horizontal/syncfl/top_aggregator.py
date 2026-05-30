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
import time
from copy import deepcopy
from datetime import datetime
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
from flame.sim import VirtualClock
from flame.selector.properties import PROP_SIM_SEND_TS, PROP_SIM_COMPLETION_TS

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
TAG_HEARTBEAT = "heartbeat_recv"


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

    def _aggregate_weights(self, tag: str) -> None:
        logger.info("Agg weights inside top_aggregator syncfl")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # For REFL/Oort with overcommitment: wait for aggGoal responses, not all selected
        agg_goal = self.config.hyperparameters.aggregation_goal
        first_k = agg_goal if agg_goal and agg_goal > 0 else 0
        logger.info(
            f"Waiting for first_k={first_k} responses from {len(channel.ends())} selected trainers"
        )

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends(), first_k=first_k):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.info(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

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
                self._round_update_values["trainer_speed"].append(
                    channel.get_end_property(
                        end_id=end, key=PROP_ROUND_DURATION
                    ).total_seconds()
                )

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        if telemetry.is_enabled():
            ev, fields = build_agg_round(
                round_num=self._round,
                in_flight=len(channel.ends()),
                staleness=list(self._round_update_values.get("staleness", [])),
                stat_utility=list(self._round_update_values.get("stat_utility", [])),
                trainer_speed_s=list(
                    self._round_update_values.get("trainer_speed", [])
                ),
                contributing_trainers=list(self.cache.keys()),
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
            num_trainers=len(channel.ends()),
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

    @timer_decorator
    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id
        )

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )
        selected_ends = channel.ends()
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
            # Get aggregator time since start
            agg_time_since_start_s = time.time() - self.agg_start_time_ts

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
