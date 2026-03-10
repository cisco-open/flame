import logging
from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
import torch
import time
import io
import json
import hashlib
import os
import numpy as np
from datetime import datetime
import ast
from flame.config import TrainerAvailState

from flame.monitor.runtime import FwdLLMStage, timer_decorator
import flame.monitor.runtime
import math

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and PyTorch tensors"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


def _serialize_value(value):
    """Serialize a value to JSON-compatible format"""
    if isinstance(value, torch.Tensor):
        # Convert PyTorch tensor to list
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, np.ndarray):
        # Convert numpy array to list
        return value.tolist()
    elif isinstance(value, np.integer):
        # Convert numpy integer to Python int
        return int(value)
    elif isinstance(value, np.floating):
        # Convert numpy float to Python float
        return float(value)
    elif isinstance(value, (list, tuple)):
        # Recursively serialize lists/tuples
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        # Recursively serialize dictionaries
        return {key: _serialize_value(val) for key, val in value.items()}
    else:
        # For other types, try to convert to string
        return str(value)


def _extract_sample_data(example):
    """Extract standardized data from an example object"""
    # Try to get text fields
    text_a = getattr(example, "text_a", None)
    text_b = getattr(example, "text_b", None)
    text = getattr(example, "text", None)

    # If no specific text field, use string representation
    if text_a is None and text_b is None and text is None:
        text = str(example)
    else:
        # Combine text fields if multiple exist
        text_parts = []
        if text_a:
            # Serialize text_a if it's a tensor/array
            text_a_serialized = _serialize_value(text_a)
            text_parts.append(str(text_a_serialized))
        if text_b:
            # Serialize text_b if it's a tensor/array
            text_b_serialized = _serialize_value(text_b)
            text_parts.append(str(text_b_serialized))
        if text:
            # Serialize text if it's a tensor/array
            text_serialized = _serialize_value(text)
            text_parts.append(str(text_serialized))
        text = " [SEP] ".join(text_parts) if len(text_parts) > 1 else text_parts[0]

    # Get label and serialize it properly
    label = getattr(example, "label", None)
    if label is None:
        label = str(example)
    else:
        # Serialize the label to handle numpy/torch types
        label = _serialize_value(label)

    # Create a hash for easy comparison
    sample_str = f"{text}|{label}"
    sample_hash = hashlib.md5(sample_str.encode("utf-8")).hexdigest()

    return {"text": text, "label": label, "hash": sample_hash}


class FedSGDTrainer(Trainer):

    def __init__(
        self,
        trainer_id,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
        config=None,
        client_index=None,
    ):
        self.trainer = model_trainer
        self.trainer_id = trainer_id
        self.client_index = client_index  # this variable is diff from client_idx because it contains a list of clients. we dont need it. was used by fwdllm

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # NRL most of this is reduntant since our dict is of size=1. But keeping this code for consistency
        logger.debug(
            f"train_data_local_dict keys: {train_data_local_dict.keys()}, client idx: {args.client_idx}, {type(args.client_idx)}"
        )

        self.train_local = [self.train_data_local_dict[args.client_idx]]

        # this will return -1
        self.test_local = self.test_data_local_dict[args.client_idx]

        self.train_local_list = [
            [data for data in self.train_local[i]] for i in range(len(self.train_local))
        ]
        self.dataset_size = len(self.train_local_list)
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.local_sample_number = None
        # logger.info(f"self.device: {self.device}, torch.cuda.device_count(): {torch.cuda.device_count()}")

        # self.train_data_local_dict.to(self.device)

        self.args = args
        self.accumulated_error = None
        self.config = config
        self.device = device

        # abstract attributes
        self.loss_fn = torch.nn.CrossEntropyLoss
        self.dataset_size = None
        self.model = model_trainer.model
        # NRL adding new variables
        self.data_id = None
        logger.info("[GJD] self.data_id is reset to None")
        self.total_data_bins = None
        self.grad_for_var_check = None
        self.data_written_to_file = False  # Flag to prevent writing data multiple times
        setattr(
            self.trainer.model_trainer, "base_trainer", self
        )  # for accessing FedSGDTrainer methods inside model_trainer - stat utility

        # Check if client will emulate delays in training time
        self.training_delay_enabled = self.config.hyperparameters.training_delay_enabled
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)
        self.training_delay_factor = float(self.config.hyperparameters.training_delay_factor)
        self.speedup_factor = 1.0

        self.trainer_start_ts = time.time()
        #TODO (ARM): Fix this to read traces better!
        # Storing synthetic avail traces
        self.avl_events_syn_0 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_0
        )

        self.avl_events_syn_20 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_20
        )

        self.avl_events_syn_50 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_50
        )

        self.avl_events_syn_train_100_eval_0_unavail_0 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_train_100_eval_0_unavail_0
        )

        self.avl_events_syn_train_90_eval_10_unavail_0 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_train_90_eval_10_unavail_0
        )

        self.avl_events_syn_train_50_eval_30_unavail_20 = ast.literal_eval(
            self.config.hyperparameters.avl_events_syn_train_50_eval_30_unavail_20
        )

        self.client_notify = self.config.hyperparameters.client_notify

        if self.client_notify["trace"] == "syn_0":
            self.state_avl_event_ts = self.avl_events_syn_0
            logger.info(f"Set avl_events_syn_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_20":
            self.state_avl_event_ts = self.avl_events_syn_20
            logger.info(f"Set avl_events_syn_20 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_50":
            self.state_avl_event_ts = self.avl_events_syn_50
            logger.info(f"Set avl_events_syn_50 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "avl_events_syn_train_100_eval_0_unavail_0":
            self.state_avl_event_ts = self.avl_events_syn_train_100_eval_0_unavail_0
            logger.info(f"Set avl_events_syn_train_100_eval_0_unavail_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "avl_events_syn_train_90_eval_10_unavail_0":
            self.state_avl_event_ts = self.avl_events_syn_train_90_eval_10_unavail_0
            logger.info(f"Set avl_events_syn_train_90_eval_10_unavail_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "avl_events_syn_train_100_eval_0_unavail_0":
            self.state_avl_event_ts = self.avl_events_syn_train_50_eval_30_unavail_20
            logger.info(f"Set avl_events_syn_train_50_eval_30_unavail_20 for trainer id {self.trainer_id}.")
        else:
            logger.info(
                f"No avl_events set for trainer id {self.trainer_id} since state not specified."
            )

        self.avl_state = TrainerAvailState.AVL_TRAIN
        logger.info(f"Set the available_state for {self.trainer_id} to AVL_TRAIN.")

        # flag to decide whether the trainer upon unavailability will wait or exit
        self.wait_until_next_avl = self.config.hyperparameters.wait_until_next_avl

        logger.info(f"Set the wait_until_next_avl to be {self.wait_until_next_avl}")

    def _write_client_data_to_file(self, client_id, train_data, round_idx=None):
        """Write all training data for a client to a JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = "../../../../../../../client_data_files"
            os.makedirs(output_dir, exist_ok=True)

            # Include round information in filename if provided
            if round_idx is not None:
                filename = os.path.join(
                    output_dir,
                    f"flame_client_{client_id}_round_{round_idx}_training_data.json",
                )
            else:
                filename = os.path.join(
                    output_dir, f"flame_client_{client_id}_training_data.json"
                )

            # Prepare data structure
            client_data = {
                "metadata": {
                    "client_id": int(client_id),  # Ensure it's a Python int
                    "round_idx": int(round_idx) if round_idx is not None else None,
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": int(
                        len(train_data.examples)
                    ),  # Ensure it's a Python int
                    "file_format_version": "1.0",
                },
                "samples": [],
            }

            # Extract all samples
            for i, example in enumerate(train_data.examples):
                try:
                    sample_data = _extract_sample_data(example)
                    sample_data["sample_index"] = int(i)  # Ensure it's a Python int

                    # Verify all values are JSON serializable
                    for key, value in sample_data.items():
                        if isinstance(
                            value, (np.integer, np.floating, np.ndarray, torch.Tensor)
                        ):
                            sample_data[key] = _serialize_value(value)

                    client_data["samples"].append(sample_data)
                except Exception as e:
                    logger.error(
                        f"Failed to extract sample {i} for client {client_id}: {e}"
                    )
                    # Log the problematic example for debugging
                    logger.error(f"Problematic example type: {type(example)}")
                    logger.error(f"Example attributes: {dir(example)}")
                    if hasattr(example, "label"):
                        logger.error(f"Label type: {type(example.label)}")
                        logger.error(f"Label value: {example.label}")
                    raise  # Re-raise to see the full error

            # Write to file using custom encoder
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    client_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder
                )

            logger.info(
                f"Successfully wrote {len(client_data['samples'])} samples to {filename}"
            )

        except Exception as e:
            logger.error(f"Failed to write data for client {client_id}: {e}")
            # Add more detailed error information
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.debug(f"self.device: {self.device}")
        self.total_data_bins = len(self.train_local[0])

        # Write training data to files during initialization (data doesn't change across rounds)
        # Use args.client_idx since that's what's used to set up the training data
        if (
            not self.data_written_to_file
            and hasattr(self.args, "client_idx")
            and self.train_local is not None
        ):
            logger.info(
                f"Writing training data to files during initialization for client {self.args.client_idx}"
            )
            client_id = self.args.client_idx
            if len(self.train_local) > 0:
                # Use None for round_idx since this is initialization, not a specific round
                self._write_client_data_to_file(
                    client_id, self.train_local[0], round_idx=None
                )
                self.data_written_to_file = True  # Mark as written
                logger.info(
                    f"Successfully wrote training data for client {client_id} during initialization"
                )
            else:
                logger.warning(
                    f"No training data available for client {client_id} during initialization"
                )
        elif self.data_written_to_file:
            logger.info(
                "Training data already written to files, skipping initialization write"
            )
        else:
            logger.warning(
                "Cannot write training data during initialization: missing client_idx or train_local"
            )

        # loading data to gpu
        # NRL TODO: This didnt work. Error: expected all tensors to be on the same device. Needed to load them on gpu again during train_model
        for each_train_local in self.train_local[0]:
            train_data = tuple(t for t in each_train_local)
            # logger.info(f"train data: {len(train_data)}")
            train_data[1].to(self.device)
            train_data[4].to(self.device)
        logger.info(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}"
        )
        self.init_oort_variables()  # initialize oort variables for stat_utility calculation (fwdllm)

    def update_model(self, weights):
        # logger.info(f"NRL: Updated model weights: {weights}")
        self.trainer.set_model_params(weights)

    def train(self, round_idx=None):
        logger.info("entered train where weights = params and not grad")
        self.args.round_idx = round_idx

        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    @timer_decorator
    def _check_availability(self):
        if self.avl_state != TrainerAvailState.AVL_TRAIN:
            if self.wait_until_next_avl:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Waiting for it to be available"
                )
                while self.avl_state != TrainerAvailState.AVL_TRAIN:
                    time.sleep(1)
                logger.info(
                    f"Trainer id {self.trainer_id} is back to available to train."
                )
            else:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Exiting training."
                )
                return False
        return True

    @timer_decorator
    def _perform_training(self):
        logger.info(
            f"starting training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )
        logger.info(
            f"train_local_list[0][0]: {len(self.train_local_list[0][0])}, {len(self.train_local_list)}"
        )

        self.reset_stat_utility()  # reset stat_utility for this databin (fwdllm)

        # List Index to be used in case of both sync and async version.
        # In sync model version = round hence, Index = model version
        # In async: Index = model version % round
        # list_index = self._model_version % self._round if self._model_version  > self._round else self._model_version
        list_index = self.data_id # Which data bin to use for training
        logging.info(f"self._model_version: {self._model_version } - list-index/data-id = {list_index}")
        self.trainer.train(
            [self.train_local_list[0][list_index]], self.device, self.args,
            {"round_id": self._round, "data_id": self.data_id, "iteration": self.iteration_per_data_id}
        )
        self.grad_for_var_check = self.trainer.model_trainer.grad_for_var_check

    @timer_decorator
    def _emulate_training_delay(self):
        if self.training_delay_enabled == "True":
            # Eval is 3X faster than training on CPU
            # Eval on NPUs is 10-50X is faster than training on CPUs. We could take 20X if we wanted to consider an all-NPU client cohort for Eval (NPUs don't support training)
            eval_delay = self.training_delay_s / self.training_delay_factor
            time.sleep(eval_delay / self.speedup_factor)
            logger.info(
                f"Delayed eval time for trainer "
                f"{self.trainer_id} by {eval_delay}s. Sleeping for {eval_delay / self.speedup_factor}s."
            )

    @timer_decorator
    def train_with_data_id(self):
        # Create FwdLLMStage for timing/metrics logging
        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id, self.trainer_id
        )

        if self.abort_training == True:
            logger.info(
                f"Aborting training for trainer id: {self.trainer_id} because it has already sent updates for iteration_per_data_id: {self.iteration_per_data_id}"
            )
            return

        if not self._check_availability():
            return
        
        self._perform_training()

        # emulate delays in training (due to compute resource and/or
        # dataset size and/or network latency)
        self._emulate_training_delay()

        logger.info(
            f"completed training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def check_and_sleep(self) -> None:
        pass

    def check_and_update_state_avl(self):
        if hasattr(self, "cm") and self.cm is not None:
            if len(self.state_avl_event_ts) > 0:
                next_event_ts = self.trainer_start_ts + (self.state_avl_event_ts[0][0])
                if time.time() >= next_event_ts:
                    state_to_set = self.state_avl_event_ts.pop(0)[1]
                    old_status = self.avl_state.value
                    try:
                        self.avl_state = TrainerAvailState(state_to_set)
                    except ValueError:
                        logger.error(
                            f"Invalid status encountered: {state_to_set}. Retaining old status {old_status}."
                        )
                        return
                    new_status = self.avl_state.value
                    logger.info(
                        f"Changed the availability status of trainer {self.trainer_id} from {old_status} to {new_status}"
                    )
                    if self.client_notify["enabled"] == "True":
                        logger.info("Trainer trying to notify aggregator")
                        self._perform_channel_state_update(
                            tag="upload",
                            state=self.avl_state,
                            timestamp=str(time.time()),
                        )
            else:
                logger.debug(
                    f"No availability events pending for trainer {self.trainer_id}"
                )
        else:
            logger.info(
                f"Channel manager not set yet for trainer {self.trainer_id}. "
                f"Skipping avail status update. "
                f"Sleep for 20s before checking again."
            )
            time.sleep(20)

    def notify_trainer_avail(self) -> None:
        logger.info("notify_trainer_avail thread running")
        while True:
            time.sleep(1)  # Will check every 1 second
            self.check_and_update_state_avl()
