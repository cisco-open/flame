# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""horizontal FwdLLM FL trainer."""
import gc
import inspect
import logging
import math
import time

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

# TODO: (DG) torch is needed for asyncoort in oort_loss() function, but need to
# comment / uncomment based on the backend used. If it is commented, Flame can
# detect and use either of the backends. But if torch code is uncommented, it
# will be used and will not work for trainers wanting to use backends like
# tensorflow.


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
        logger.debug(
            f"self.cm._config.selector.sort: {self.cm._config.selector.sort}, self.config.selector.sort: {self.config.selector.sort}"
        )

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config)

        self.registry_client.setup_run()
        self.metrics = dict()

        # needed for trainer-side optimization algorithms such as fedprox
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

        # for tracking trainer round progress and checking before sending
        # updates
        self._updates_returned_upto_round = 0
        self._trainer_online_channel_status = True

        self.task_to_perform = "train"
        self.iteration_per_data_id = None
        self.abort_training = False

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug(
            f"### FETCH WEIGHTS start for tag: {tag} "
            f"and trainer_id {self.trainer_id}"
        )

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info(
                f"fetch weights, channel not found with tag {tag} "
                f"for trainer_id {self.trainer_id}"
            )
            # we don't want to keep calling this too fast so let's sleep 1
            # second
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
        msg, _ = channel.recv(end)

        if not msg:
            logger.debug(f"NO msg received for trainer_id {self.trainer_id}")
            if self._work_done:
                # when the work is done, we cancel continue condition (i.e., we
                # set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast so let's sleep 1
            # second
            time.sleep(1)
            return

        logger.info(f"New message received for trainer_id {self.trainer_id}")

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        if MessageType.DATA_ID in msg and MessageType.ITERATION_PER_DATA_ID in msg:
            if (
                self.data_id is not None
                and self.data_id == msg[MessageType.DATA_ID]
                and self.iteration_per_data_id is not None
                and self.iteration_per_data_id == msg[MessageType.ITERATION_PER_DATA_ID]
            ):
                self.abort_training = True
                logger.info(
                    f"Fetch weights aborted for given model version "
                    f"{self._round} while trainer_id {self.trainer_id} has "
                    f"already sent updates "
                    f"upto iteration_per_data_id: {self.iteration_per_data_id}"
                )

                # Received old data but still allow aggregator cleanup state to
                # occur so as to receive the next update
                logger.info(
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
            else:
                self.abort_training = False
                self.iteration_per_data_id = msg[MessageType.ITERATION_PER_DATA_ID]

        if MessageType.VAR in msg:
            logger.info(
                f"Calc more variance received for trainer id: {self.trainer_id} and round {self._round}. Not updating weights"
            )

        elif MessageType.WEIGHTS in msg:
            # Before proceeding, check if this model version is newer than
            # previously processed NOTE: The condition could have been round <=
            # updates_retuned. But there are scenarios where the channel.leave()
            # executes before the aggregator processes the weight update. Hence,
            # with <= condition, the trainer would never make progress. We allow
            # to trainer to re-train for == round condition if the message was
            # dropped.
            logger.info("message type weights received")

            # if self._round <= self._updates_returned_upto_round: logger.info(
            #     f"Fetch weights aborted for given model version "
            #         f"{self._round} while trainer_id {self.trainer_id} has "
            #         f"already sent updates " f"upto round:
            #         {self._updates_returned_upto_round}" )

            #     # Received old data but still allow aggregator cleanup
            #     # state to occur so as to receive the next update
            #     logger.info( f"Cleaning up recvd ends for trainer_id
            #         {self.trainer_id}" f" to allow fetch from aggregator "
            #         "again and returning from function" )
            #         channel._selector.ordered_updates_recv_ends.append(end)
            #     logger.debug( f"After appending {end} to
            #     ordered_updates_recv_ends: "
            #     f"{channel._selector.ordered_updates_recv_ends}" )
            #         channel.cleanup_recvd_ends() return

            # Load the model onto GPU if self.model is None:
            # self._load_model_onto_gpu()

            # Update the model logger.info(f"Weights received:
            # {msg[MessageType.WEIGHTS]}") self.weights =
            # weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            trainable_weights = weights_to_model_device(
                msg[MessageType.WEIGHTS], self.model
            )
            full_state_dict = self.model.state_dict()
            full_state_dict.update(trainable_weights)
            self.weights = full_state_dict
            self._update_model()
            
            if MessageType.DATA_ID in msg:
                logger.info(
                    f"Trainer id {self.trainer_id} received data id for training : {msg[MessageType.DATA_ID]}"
                )
                self.data_id = msg[MessageType.DATA_ID]

            if MessageType.GRAD_POOL in msg:
                partial_grad = msg[MessageType.GRAD_POOL]
                full_grad = []
                if self.args.var_control:
                    if self.args.perturbation_sampling:
                        logger.info(
                            f"Trainer id {self.trainer_id} using grad_pool from message"
                        )
                        trainable_idx = 0
                        if partial_grad == None:
                            full_grad = None
                        else:
                            for param in self.model.parameters():
                                if param.requires_grad:
                                    full_grad.append(partial_grad[trainable_idx])
                                    trainable_idx += 1
                                else:
                                    # Append a zero tensor of same shape as
                                    # param instead of None
                                    full_grad.append(
                                        torch.zeros_like(param, device="cpu")
                                    )
                        if self.data_id % 2:
                            self.trainer.model_trainer.old_grad = full_grad
                        else:
                            self.trainer.model_trainer.old_grad = None
                        # logger.info(
                        #     f"Trainer id {self.trainer_id} using grad_pool from message {self.trainer.model_trainer.old_grad}"
                        # )
                del full_grad
        else:
            logger.info(
                f"Invalid message received from agg for trainer id: {self.trainer_id} - skipping "
            )

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.DATASAMPLER_METADATA in msg:
            self.datasampler.handle_metadata_from_aggregator(
                msg[MessageType.DATASAMPLER_METADATA]
            )
        if MessageType.TASK_TO_PERFORM in msg:
            self.task_to_perform = msg[MessageType.TASK_TO_PERFORM]
            logger.debug(f"Found task_to_perform in msg: {self.task_to_perform}")
        else:
            logger.info(f"Didn't find TASK_TO_PERFORM in msg")

        self.fetch_success = True

        logger.info(
            f"### FETCH WEIGHTS complete for trainer_id {self.trainer_id}, "
            f"round: {self._round}, data id: {self.data_id} and work_done: {self._work_done} ###"
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
        logging.info(f"Put is invoked for {self.trainer_id}")
        if tag == TAG_UPLOAD:
            self._send_grads(tag)
        elif tag == TAG_HEARTBEAT:
            logger.info("calling send heartbeat")
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
        logger.info(f"sending heartbeat done for trainer_id: {self.trainer_id}")

        return

    def _send_grads(self, tag: str) -> None:
        # Added a 1 second sleep so as to not overwhelm mqtt time.sleep(1)

        if self.abort_training == True:
            logger.info(
                f"Aborting sending grads for trainer id: {self.trainer_id} because it has already sent updates for iteration_per_data_id: {self.iteration_per_data_id}"
            )
            return
        logger.debug(
            f"### SEND GRADS for tag: {tag} " f"and trainer_id: {self.trainer_id}"
        )

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_grads] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(
            f"_send_grads: waiting for someone to join channel: {channel} "
            f"for trainer_id: {self.trainer_id}"
        )
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        if self.task_to_perform == "train":
            # trainer is expected to train and it is also available to train -
            # best case self._update_weights()

            # delta_weights = self._delta_weights_fn(self.weights,
            # self.prev_weights)

            # delta_weights = self.privacy.apply_dp_fn(delta_weights)

            # self.regularizer.update()

            # NOTE: Also sending stat_utility for OORT

            # Retrieve forward gradients before sending it from trainer to
            # aggregator Collect gradients into a dictionary where keys are
            # layer names and values are tensors
            grad_dict = {
                name: p.grad.clone()
                for i, (name, p) in enumerate(self.model.named_parameters())
                if p.grad is not None
            }

            # Log the gradient dictionary details
            if grad_dict:
                total_bytes = sum(v.element_size() * v.nelement() for v in grad_dict.values())
                size_mb = total_bytes / (1024 * 1024)

                logger.info(
                    f"Going to send gradients dictionary with {len(grad_dict)} entries "
                    f"({size_mb:.2f} MB)."
                )
            else:
                logger.info("No gradients exist; sending an empty dictionary.")

            msg = {
                MessageType.GRADIENTS: grad_dict,
                MessageType.GRADIENTS_FOR_VAR_CHECK: self.grad_for_var_check,
                MessageType.DATASET_SIZE: self.dataset_size,
                MessageType.MODEL_VERSION: self._round,
                MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
                # MessageType.STAT_UTILITY: self._stat_utility, #uncomment later
                # - rn FedSgdTrainer has no utility
                MessageType.TOTAL_DATA_BINS: self.total_data_bins,
            }
        else:
            msg = {
                MessageType.MODEL_VERSION: self._round,
                MessageType.STAT_UTILITY: self._stat_utility,
            }

        channel.send(end, msg)

        if self.task_to_perform == "train":
            # To allow the trainer to participate in eval AND train in the same
            # round, we set _updates_returned_upto_round only over here.
            self._updates_returned_upto_round = self._round

            logger.info(
                f"sending grads done for trainer_id: {self.trainer_id} "
                f"and _updates_returned_upto_round "
                f"{self._updates_returned_upto_round}"
            )
        elif self.task_to_perform == "eval":
            logger.info(
                f"sending eval stat utility done for trainer_id: {self.trainer_id} "
                f"for model version: {self._round}"
            )
        else:
            logger.error(
                f"Task to perform is not defined for trainer_id: {self.trainer_id}"
            )

        # Evict model from gpu to free up space self._evict_model_from_gpu()

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

        # Setting the channel status to False. Means that trainer should not
        # send updates during this time.
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

        # Setting the channel status to True. Means that trainer can now resume
        # sending updates.
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
            f"{channel._name} and trainer: {self.trainer_id}, with state: {state} from timestamp: {timestamp}"
        )

    def save_metrics(self):
        """Save metrics in a model registry."""
        # update self.metrics with metrics from MetricCollector instance
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
            #     update_model but " f"model was not on GPU. Load completed.")
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        # save weights before updating it
        self.prev_weights = self.weights

        if self.framework == MLFramework.PYTORCH:
            # if self.model is None: self._load_model_onto_gpu()
            #     logger.error(f"Trainer {self.trainer_id} came to
            #     update_weights before " f"sending. But the model had to be
            #                  loaded on the device.")
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    ## TODO: This function isn't currently being used. Need to use this so that even first _send_grads() from aggregator need not send 250MB weights. Should only send grads.
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

        if "reduction" not in inspect.signature(self.loss_fn).parameters:
            msg = "Parameter 'reduction' not found in loss function "
            msg += f"'{self.loss_fn.__name__}', which is required for Oort"
            raise TypeError(msg)

    # TODO: Enable this in trainer code using a flag based on selector used.
    # Needs to also pass to trainer/main.py
    def oort_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        batch_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Measure the loss of a trainer during training. The trainer's statistical
        utility is measured at epoch 1.
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
        Normalize statistical utility of a trainer based on the size of the
        trainer's datset, at epoch 1.
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

    def pause_execution(self):
        time.sleep(1)
        return

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            # task_init_oort_variables = Tasklet( "init_oort_variables",
            #     self.init_oort_variables )

            # task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            # task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            # task_sleep_after_get = Tasklet("sleep_after_get",
            # self.check_and_sleep)

            # task_sleep_after_train = Tasklet("sleep_after_train",
            # self.check_and_sleep)

            # task_sleep_after_eval = Tasklet("sleep_after_eval",
            # self.check_and_sleep)

            # task_sleep_after_put_weight = Tasklet("sleep_after_put_weight",
            #   self.check_and_sleep)

            # task_sleep_after_save_metrics =
            #                                         Tasklet("sleep_after_save_metrics",
            #                                         self.check_and_sleep)

            task_train = Tasklet("train", self.train_with_data_id)

            # task_eval = Tasklet("evaluate", self.evaluate)

            task_put_grad = Tasklet("upload", self.put, TAG_UPLOAD)
            task_pause_exec = Tasklet("pause_exec", self.pause_execution)

            # task_save_metrics = Tasklet("save_metrics", self.save_metrics)
            task_send_heartbeat = Tasklet("upload", self.put, TAG_HEARTBEAT)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)

            # Now start the rest of the tasks
            (
                task_init
                >> task_internal_init
                >> loop(
                    task_get
                    >> task_train
                    # >> task_pause_exec
                    >> task_put_grad
                    # >> asyncfl_loop(task_put >> task_get_weights >>
                    # >> task_get_heartbeat
                    >> task_pause_exec
                )
                # >> task_init_oort_variables Added code here to check for the
                # status of the task i.e., "train + eval" vs "eval only"
                # task_load_data loop( task_get >> task_sleep_after_get >>
                # >> task_train >> task_sleep_after_train >> task_eval >>
                # >> task_sleep_after_eval >> task_put_weight >>
                #     task_sleep_after_put_weight >> task_save_metrics >>
                #     task_sleep_after_save_metrics
                # # )
                # >> loop( task_send_heartbeat )
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer
        role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_HEARTBEAT]
