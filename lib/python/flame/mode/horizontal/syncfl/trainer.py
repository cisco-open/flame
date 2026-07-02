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
"""horizontal FL trainer."""
import gc
import inspect
import logging
import math
import time

import cloudpickle
from contextlib import contextmanager

import torch
from flame.channel import VAL_CH_STATE_HTBT_SEND, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
    get_ml_framework_in_use,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.config import Config, TrainerAvailState
from flame.datasamplers import datasampler_provider
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizers import optimizer_provider
from flame.privacies import privacy_provider
from flame.registries import registry_provider
from flame import telemetry
from flame.telemetry.events import build_task_recv, build_task_send

# TODO: (DG) torch is needed for asyncoort in oort_loss() function,
# but need to comment / uncomment based on the backend used. If it is
# commented, Flame can detect and use either of the backends. But if
# torch code is uncommented, it will be used and will not work for
# trainers wanting to use backends like tensorflow.


logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"
TAG_HEARTBEAT = "heartbeat_send"


class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    @abstract_attribute
    def loss_fn(self):
        # Added for OORT
        """Abstract attribute for loss function."""

    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset_size(self):
        """Abstract attribute for size of dataset used to train."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config)

        self.registry_client.setup_run()
        self.metrics = dict()

        # needed for trainer-side optimization algorithms such as
        # fedprox
        temp_opt = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )
        self.regularizer = temp_opt.regularizer

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).trainer_data_sampler

        self.privacy = privacy_provider.get(
            self.config.privacy.sort, **self.config.privacy.kwargs
        )

        self._round = 1
        self._work_done = False

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        if self.framework == MLFramework.PYTORCH:
            self._delta_weights_fn = delta_weights_pytorch

        elif self.framework == MLFramework.TENSORFLOW:
            self._delta_weights_fn = delta_weights_tensorflow

        self.fetch_success = False

        self.trainer_id = self.config.task_id

        # for tracking trainer round progress and checking before
        # sending updates
        self._updates_returned_upto_round = 0
        self._trainer_online_channel_status = True

        self.task_to_perform = "train"

        # Per-round phase timing accumulator; reset at each round boundary in _fetch_weights.
        self._phase_times: dict = {}

    @contextmanager
    def _phase(self, name: str):
        """Time a named phase and accumulate into self._phase_times."""
        t0 = time.time()
        try:
            yield
        finally:
            self._phase_times[name] = self._phase_times.get(name, 0.0) + (time.time() - t0)

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug(
            f"### FETCH WEIGHTS start for tag: {tag}, "
            f"trainer_id: {self.trainer_id}, current_model_version: {self._round}"
        )

        # Reset per-round phase accumulator at the round boundary.
        self._phase_times = {}

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(
                f"fetch weights, channel not found with tag {tag} "
                f"for trainer_id {self.trainer_id}"
            )
            # we don't want to keep calling this too fast so let's
            # sleep 1 second
            time.sleep(1)
            return

        # this call waits for at least one peer joins this channel
        logger.debug(
            f"_fetch_weights: waiting for someone to join channel: {channel} "
            f"for trainer_id {self.trainer_id}"
        )
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        _recv_wall_start = time.time()
        msg, _ = channel.recv(end)
        # Stamp as early as possible so the aggregator can measure
        # the agg→trainer delivery leg (i).
        self._wall_recv_ts = time.time()
        self._phase_times["mqtt_fetch_s"] = self._wall_recv_ts - _recv_wall_start

        if not msg:
            logger.debug(f"NO msg received for trainer_id {self.trainer_id}")
            if self._work_done:
                # when the work is done, we cancel continue condition
                # (i.e., we set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast so let's
            # sleep 1 second
            time.sleep(1)
            return

        logger.debug(f"New message received for trainer_id {self.trainer_id}")

        if MessageType.ROUND in msg:
            prev_round = self._round
            self._round = msg[MessageType.ROUND]
            logger.debug(f"[TRAINER_FETCH] Updated round from {prev_round} to {self._round} for trainer_id {self.trainer_id}")

        if MessageType.WEIGHTS in msg:
            # Before proceeding, check if this model version is newer
            # than previously processed NOTE: The condition could have
            # been round <= updates_retuned. But there are scenarios
            # where the channel.leave() executes before the aggregator
            # processes the weight update. Hence, with <= condition,
            # the trainer would never make progress. We allow to
            # trainer to re-train for == round condition if the
            # message was dropped.
            if self._round <= self._updates_returned_upto_round:
                logger.info(
                    f"[TRAINER_FETCH_ABORT] Fetch weights aborted for given model version "
                    f"{self._round} while trainer_id {self.trainer_id} has "
                    f"already sent updates "
                    f"upto round: {self._updates_returned_upto_round}"
                )

                # Received old data but still allow aggregator cleanup
                # state to occur so as to receive the next update
                logger.debug(
                    f"Cleaning up recvd ends for trainer_id {self.trainer_id}"
                    f" to allow fetch from aggregator "
                    "again and returning from function"
                )
                channel._selector.ordered_updates_recv_ends.append(end)
                logger.debug(
                    f"After appending {end} to ordered_updates_recv_ends: "
                    f"{channel._selector.ordered_updates_recv_ends}"
                )
                channel.cleanup_recvd_ends()
                return

            # Load the model onto GPU if self.model is None:
            # self._load_model_onto_gpu()

            # Update the model
            with self._phase("weights_to_ram_s"):
                self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            with self._phase("weights_to_gpu_s"):
                self._update_model()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Capture virtual send-time stamped by aggregator (sim mode); used for sim_completion_ts.
        if MessageType.SIM_SEND_TS in msg:
            self._sim_send_ts = msg[MessageType.SIM_SEND_TS]

        # Cache the aggregator's trace-read origin (real mode only) so this
        # trainer's own wall-clock availability lookups share the exact
        # origin the aggregator uses, instead of deriving one from its own
        # process-start time. Re-cached every dispatch (cheap, idempotent,
        # self-healing if an early message was missed).
        if MessageType.AGG_START_TS in msg:
            self._agg_start_origin = msg[MessageType.AGG_START_TS]

        if telemetry.is_enabled():
            _sim_send_ts_val = getattr(self, "_sim_send_ts", None)
            _time_mode = getattr(self, "time_mode", "real")
            _avl = getattr(getattr(self, "avl_state", None), "value", None)
            ev, fields = build_task_recv(
                round_num=int(self._round),
                trainer_id=str(getattr(self, "trainer_id", "")),
                time_mode=_time_mode,
                sim_send_ts=float(_sim_send_ts_val) if _sim_send_ts_val is not None else None,
                avl_state=_avl,
            )
            telemetry.emit(ev, **fields)

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]
            # Give a sim-mode trainer that hasn't been dispatched in a while
            # one last chance to catch its avl_state/telemetry up to the
            # trace, using the SIM_SEND_TS this EOT broadcast may carry
            # (captured above). hasattr-guarded: example-specific hook, not
            # every Trainer subclass defines it.
            if hasattr(self, "_refresh_avl_state"):
                self._refresh_avl_state()

        if MessageType.DATASAMPLER_METADATA in msg:
            self.datasampler.handle_metadata_from_aggregator(
                msg[MessageType.DATASAMPLER_METADATA]
            )
        if MessageType.TASK_TO_PERFORM in msg:
            self.task_to_perform = msg[MessageType.TASK_TO_PERFORM]
            logger.debug(f"Found task_to_perform in msg: {self.task_to_perform}")
        else:
            logger.debug(f"Didn't find TASK_TO_PERFORM in msg")

        self.fetch_success = True

        logger.info(
            f"### FETCH WEIGHTS complete for trainer_id {self.trainer_id}, "
            f"round: {self._round}, model_version: {self._round}, task_to_perform (can be default): {self.task_to_perform}, work_done: {self._work_done} ###"
        )

        logger.debug(
            "Model weights received, so resetting aggregator end states in "
            "the channel"
        )

        channel._selector.ordered_updates_recv_ends.append(end)
        logger.debug(
            f"After appending {end} to ordered_updates_recv_ends: "
            f"{channel._selector.ordered_updates_recv_ends}"
        )

        channel.cleanup_recvd_ends()

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        logger.debug(
            f"avl_state of trainer {self.trainer_id} when put is invoked, is: {self.avl_state}"
        )
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        elif tag == TAG_HEARTBEAT:
            self._send_heartbeat_to_agg(tag)

    def _send_heartbeat_to_agg(self, tag: str) -> None:
        logger.debug(
            f"### SEND heartbeat for tag: {tag} " f"and trainer_id: {self.trainer_id}"
        )
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_heartbeat] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_send_heartbeat: waiting for someone to join channel: {channel} "
            f"for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_HTBT_SEND)

        msg = {
            MessageType.HEARTBEAT: time.time(),
        }
        channel.send(end, msg)
        logger.debug(f"sending heartbeat done for trainer_id: {self.trainer_id}")

        return

    def _send_weights(self, tag: str) -> None:
        logger.debug(
            f"### SEND WEIGHTS for tag: {tag} "
            f"and trainer_id: {self.trainer_id}, model_version: {self._round}, and avl_state = {self.avl_state}"
        )
        # [SEND_GATE] real-mode send-time gate: hold the upload (already-
        # completed result) until the trainer is AVL_* again. Decoupled from
        # client_notify["enabled"] (v1 keeps it OFF -- the aggregator learns
        # oracularly, not via this push). Sim-only is a no-op here since sim
        # time can't advance while blocked on time.sleep -- sim availability
        # is instead enforced agg-side by ClientAvailability's send-time
        # withhold, keyed on the trainer-reported completion time.
        #
        # Sample send_gate_sct (trainer's own trace-time clock) right before
        # the gate check regardless of whether it engages, so A8 can confirm
        # the no-wait case too. Real mode only; hasattr-guarded since
        # _sim_now() is example-specific, not on this generic base class.
        _send_gate_sct = (
            self._sim_now()
            if not getattr(self, "simulated", False) and hasattr(self, "_sim_now")
            else None
        )
        if (
            not getattr(self, "simulated", False)
            and self.avl_state == TrainerAvailState.UN_AVL
        ):
            if self.wait_until_next_avl == "True":
                logger.warning(
                    f"Trainer id {self.trainer_id} is unavailable to send weights. Waiting for it to be available again"
                )
                with self._phase("send_gate_wait_s"):
                    while self.avl_state == TrainerAvailState.UN_AVL:
                        time.sleep(1)
            else:
                logger.warning(
                    f"Trainer id {self.trainer_id} is unavailable to send weights since wait_until_next_avl = {self.wait_until_next_avl}. Exiting sending weights."
                )
                return

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_send_weights: waiting for someone to join channel: {channel} "
            f"for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        if self.task_to_perform == "train":
            # trainer is expected to train and it is also available to
            # train - best case
            with self._phase("weights_from_gpu_s"):
                # model.state_dict() copies GPU tensors to CPU; sync for timing accuracy.
                self._update_weights()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            with self._phase("post_cpu_s"):
                delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

                delta_weights = self.privacy.apply_dp_fn(delta_weights)

                self.regularizer.update()

                self.finalize_local_accuracy()

                msg = {
                    MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
                    MessageType.DATASET_SIZE: self.dataset_size,
                    MessageType.MODEL_VERSION: self._round,
                    MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
                    MessageType.STAT_UTILITY: self._stat_utility,
                    MessageType.LOCAL_ACCURACY: self._local_accuracy,
                }
        else:
            msg = {
                MessageType.MODEL_VERSION: self._round,
                MessageType.STAT_UTILITY: self._stat_utility,
                MessageType.LOCAL_ACCURACY: self._local_accuracy,
            }

        # simulated-time mode only: report the modeled completion time/duration
        # so the aggregator orders this update by a virtual clock. No-op in real
        # mode and for trainers that don't set these (other examples).
        _sim_completion = getattr(self, "_sim_completion_ts", None)
        if getattr(self, "simulated", False) and _sim_completion is not None:
            msg[MessageType.SIM_COMPLETION_TS] = _sim_completion
            msg[MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S] = getattr(
                self, "_sim_round_duration", 0.0
            )

        # Lazy-deserialize (real and sim): ship the weight update as raw pre-serialized bytes so
        # the aggregator reconstructs the tensor only for updates it commits, not the surplus/
        # stale ones it discards (the channel otherwise cloudpickle.loads every received tensor).
        # The aggregator restores it via common.util.materialize_weights at its read site.
        if MessageType.WEIGHTS in msg:
            msg[MessageType.WEIGHTS_BYTES] = cloudpickle.dumps(
                msg.pop(MessageType.WEIGHTS)
            )

        _budget = getattr(self, "_training_budget_s", None)
        if _budget is not None:
            msg[MessageType.TRAINING_BUDGET_S] = float(_budget)

        # Modeled round compute: max(real_gpu_time, training_delay_s). Stamped
        # unconditionally (real + sim) so the aggregator can decompose the
        # trainer-side lag into delivery + compute + post-wait in both modes.
        _compute_s = getattr(self, "_sim_round_duration", None)
        if _compute_s is not None:
            msg[MessageType.CLIENT_TASK_TRAIN_COMPUTE_S] = float(_compute_s)

        # Trainer recv timestamp: when channel.recv() returned the distributed
        # weights. Used by the aggregator for the agg→trainer delivery leg (i).
        _wrt = getattr(self, "_wall_recv_ts", None)
        if _wrt is not None:
            msg[MessageType.WALL_RECV_TS] = float(_wrt)

        # Stamp wall-clock send time so aggregator can decompose wall_lag_s.
        _wall_send_ts = time.time()
        msg[MessageType.WALL_SEND_TS] = _wall_send_ts

        with self._phase("mqtt_send_s"):
            channel.send(end, msg)

        # In-flight window for validate_real: wall_send_ts is stamped here,
        # AFTER the real-mode budget sleep in train(), so [wall_recv_ts, wall_send_ts]
        # brackets the trainer's true busy window — which trainer_round (emitted
        # pre-sleep) cannot. No-op when telemetry is disabled.
        if telemetry.is_enabled():
            ev, fields = build_task_send(
                round_num=int(getattr(self, "_round", 0)),
                trainer_id=str(getattr(self, "trainer_id", "")),
                task_to_perform=getattr(self, "task_to_perform", None),
                wall_recv_ts=getattr(self, "_wall_recv_ts", None),
                wall_send_ts=_wall_send_ts,
                time_mode=getattr(self, "time_mode", "real"),
                send_gate_wait_s=(
                    float(self._phase_times.get("send_gate_wait_s", 0.0))
                    if not getattr(self, "simulated", False)
                    else None
                ),
                send_gate_sct=_send_gate_sct,
            )
            telemetry.emit(ev, **fields)

        if self.task_to_perform == "train":
            # To allow the trainer to participate in eval AND train in
            # the same round, we set _updates_returned_upto_round only
            # over here.
            self._updates_returned_upto_round = self._round

            logger.info(
                f"[TRAINER_SEND_WEIGHTS] Sent weights for trainer_id: {self.trainer_id}, "
                f"model_version: {self._round}, "
                f"wall_send_ts={_wall_send_ts:.3f}, "
                f"_updates_returned_upto_round: {self._updates_returned_upto_round}, "
                f"stat_utility: {self._stat_utility}, dataset_size: {self.dataset_size}"
            )
        elif self.task_to_perform == "eval":
            logger.info(
                f"[TRAINER_SEND_EVAL] Sent stat utility of {self._stat_utility} for trainer_id: {self.trainer_id} "
                f"at round: {self._round}, model_version: {self._round}"
            )
        else:
            logger.error(
                f"Task to perform is not defined for trainer_id: {self.trainer_id}"
            )

        # Evict model from gpu to free up space
        # self._evict_model_from_gpu()

        channel._selector._cleanup_send_ends()

    def _perform_channel_leave(self, tag: str) -> None:
        logger.debug(
            f"In _perform_channel_leave for tag: {tag} "
            f"and trainer_id: {self.trainer_id}"
        )
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_perform_channel_leave] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_perform_channel_leave: waiting for someone to join channel: "
            f"{channel} for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        # Setting the channel status to False. Means that trainer
        # should not send updates during this time.
        self._trainer_online_channel_status = False

        channel.leave()
        logger.info(
            f"Sent channel leave message for channel: "
            f"{channel._name} and trainer: {self.trainer_id}."
            f" Set trainer_online_channel_status: "
            f"{self._trainer_online_channel_status}"
        )

    def _perform_channel_join(self, tag: str) -> None:
        logger.debug(
            f"In _perform_channel_join for tag: {tag} "
            f"and trainer_id: {self.trainer_id}"
        )
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_perform_channel_join] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_perform_channel_join: waiting for someone to join channel: "
            f"{channel} for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        channel.join()

        # Setting the channel status to True. Means that trainer can
        # now resume sending updates.
        self._trainer_online_channel_status = True

        logger.info(
            f"Sent channel join message for channel: "
            f"{channel._name} and trainer: {self.trainer_id}."
            f" Set trainer_online_channel_status: "
            f"{self._trainer_online_channel_status}"
        )

    def _perform_channel_state_update(
        self, tag: str, state: TrainerAvailState, timestamp: str
    ) -> None:
        logger.debug(
            f"In _perform_channel_state_update for tag: {tag}, "
            f"trainer_id: {self.trainer_id}, "
            f"new state: {state}, "
            f"from timestamp: {timestamp}"
        )
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(
                f"[_perform_channel_state_update] channel not found with {tag}"
            )
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_perform_channel_state_update: waiting for someone to join channel: "
            f"{channel} for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        channel.update_trainer_state(state, timestamp)
        logger.info(
            f"Sent channel state update message for channel: "
            f"{channel._name} and trainer: {self.trainer_id}, new state: {state} at timestamp: {timestamp}"
        )

    def save_metrics(self):
        """Save metrics in a model registry."""
        # update self.metrics with metrics from MetricCollector
        # instance
        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")
        self.metrics = dict()

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            # if self.model is None: self._load_model_onto_gpu()
            #     logger.debug(f"Trainer_id: {self.trainer_id} came to
            #     update_model but " f"model was not on GPU. Load
            #                  completed.")
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        # save weights before updating it
        self.prev_weights = self.weights

        if self.framework == MLFramework.PYTORCH:
            # if self.model is None: self._load_model_onto_gpu()
            #     logger.error(f"Trainer {self.trainer_id} came to
            #     update_weights before " f"sending. But the model had
            #                  to be loaded on the device.")
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def _load_model_onto_gpu(self):
        self.model = self.model_arch().to(self.device)
        logger.debug(f"Loaded model on gpu for trainer_id: {self.trainer_id}")

    def _evict_model_from_gpu(self):
        self.model.cpu()
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear the CUDA cache again, just in case
        logger.debug(f"Evicted model from gpu for trainer_id: {self.trainer_id}")

    def send_heartbeat_to_agg(self) -> None:
        logger.debug("Inside trainer.py will call self.put(heartbeat)")
        self.put(TAG_HEARTBEAT)

    # #### ADDED OORT RELATED FUNCTIONALITY
    def init_oort_variables(self) -> None:
        """Initialize Oort variables."""
        self._stat_utility = 0
        self._local_accuracy = 0.0
        self._local_accuracy_correct = 0
        self._local_accuracy_total = 0

        if "reduction" not in inspect.signature(self.loss_fn).parameters:
            msg = "Parameter 'reduction' not found in loss function "
            msg += f"'{self.loss_fn.__name__}', which is required for Oort"
            raise TypeError(msg)

    def update_local_accuracy(
        self, output: "torch.Tensor", target: "torch.Tensor"
    ) -> None:
        """Accumulate top-1 classification accuracy. Override for non-classification tasks."""
        with torch.no_grad():
            pred = output.argmax(dim=-1)
            # Accumulate the correct-count on-device and defer the single
            # GPU->CPU sync to finalize_local_accuracy(). A per-batch .item()
            # here forces a synchronization every batch, which stalls badly when
            # many trainers share one GPU (the sync waits on the shared queue).
            # numel() is a Python int from the tensor shape (no sync).
            self._local_accuracy_correct = (
                self._local_accuracy_correct + (pred == target).sum()
            )
            self._local_accuracy_total += int(target.numel())

    def finalize_local_accuracy(self) -> None:
        correct = self._local_accuracy_correct
        if torch.is_tensor(correct):
            correct = int(correct.item())  # one sync per round, not per batch
        if self._local_accuracy_total > 0:
            self._local_accuracy = correct / self._local_accuracy_total
        else:
            self._local_accuracy = 0.0

    def reset_local_accuracy(self) -> None:
        self._local_accuracy = 0.0
        self._local_accuracy_correct = 0
        self._local_accuracy_total = 0

    # TODO: Enable this in trainer code using a flag based on selector
    # used. Needs to also pass to trainer/main.py
    def oort_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        batch_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Measure the loss of a trainer during training. The trainer's
        statistical utility is measured at epoch 1.
        """
        if epoch == 1 and batch_idx == 0:
            if "reduction" in kwargs.keys():
                reduction = kwargs["reduction"]
            else:
                reduction = "mean"  # default reduction policy is mean
            kwargs_wo_reduction = {
                key: value for key, value in kwargs.items() if key != "reduction"
            }

            criterion = self.loss_fn(reduction="none", **kwargs_wo_reduction)
            loss_list = criterion(output, target)
            self._stat_utility += torch.square(loss_list).sum()

            if reduction == "mean":
                loss = loss_list.mean()
            elif reduction == "sum":
                loss = loss_list.sum()
        else:
            criterion = self.loss_fn(**kwargs)
            loss = criterion(output, target)

        return loss

    def normalize_stat_utility(self, epoch) -> None:
        """
        Normalize statistical utility of a trainer based on the size
        of the trainer's datset, at epoch 1.
        """
        if epoch == 1:
            self._stat_utility = len(self.train_loader.dataset) * math.sqrt(
                self._stat_utility / len(self.train_loader.dataset)
            )
        else:
            return

    def reset_stat_utility(self) -> None:
        """Reset the trainer's statistical utility to zero."""
        self._stat_utility = 0

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init_oort_variables = Tasklet(
                "init_oort_variables", self.init_oort_variables
            )

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_sleep_after_get = Tasklet("sleep_after_get", self.check_and_sleep)

            task_sleep_after_train = Tasklet("sleep_after_train", self.check_and_sleep)

            task_sleep_after_eval = Tasklet("sleep_after_eval", self.check_and_sleep)

            task_sleep_after_put_weight = Tasklet(
                "sleep_after_put_weight", self.check_and_sleep
            )

            task_sleep_after_save_metrics = Tasklet(
                "sleep_after_save_metrics", self.check_and_sleep
            )

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put_weight = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)

            # Now start the rest of the tasks
            (
                task_init
                >> task_internal_init
                >> task_init_oort_variables
                # Added code here to check for the status of the task
                # i.e., "train + eval" vs "eval only"
                >> task_load_data
                >> loop(
                    task_get
                    >> task_sleep_after_get
                    >> task_train
                    >> task_sleep_after_train
                    >> task_eval
                    >> task_sleep_after_eval
                    >> task_put_weight
                    >> task_sleep_after_put_weight
                    >> task_save_metrics
                    >> task_sleep_after_save_metrics
                )
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer
        role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_HEARTBEAT]
