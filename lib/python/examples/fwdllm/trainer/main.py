import os
import socket
import sys

import psutil
import setproctitle
import torch
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (
    TLMPreprocessor,
)
from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (
    ForwardTextClassificationTrainer,
)
from examples.fwdllm.trainer.model.transformer.model_args import ClassificationArgs
from examples.fwdllm.data_manager.text_classification_data_manager import (
    TextClassificationDataManager,
)
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager
from examples.fwdllm.expts.initializer import (
    add_federated_args,
    set_seed,
    create_model,
)
from examples.fwdllm.trainer.forward_training.fed_trainer_transformer import (
    FedTransformerTrainer,
)
from examples.fwdllm.trainer.forward_training.FedSgdTrainer import FedSGDTrainer

import argparse
import logging
from flame.launch.cli import load_config_from_argv
from flame import telemetry


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


if __name__ == "__main__":
    config = load_config_from_argv()

    # --time_mode is a launcher CLI-only arg (not in config JSON), unconditionally
    # appended by TrainerSpawner.spawn_trainer() to every trainer's argv. FedFwd has
    # no simulated-clock concept (FedSgdTrainer's _emulate_training_delay() always
    # sleeps real wall-clock time) -- parsed here only so the launcher's argv
    # injection doesn't crash with "unrecognized argument"; "simulated" is a
    # documented no-op for now, deferred future work, not retrofitted in this
    # migration. log_level is similarly launcher/manual-run CLI-only.
    _cli_parser = argparse.ArgumentParser(add_help=False)
    _cli_parser.add_argument("--time_mode", default="real")
    _cli_parser.add_argument("--log_level", default="INFO")
    _cli_parser.add_argument("--battery_threshold", default=50)
    _cli_args, _ = _cli_parser.parse_known_args()

    logging.basicConfig(
        level=logging._nameToLevel[_cli_args.log_level],
        format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.debug(config)
    telemetry.configure(role="trainer", end_id=str(config.task_id))
    if _cli_args.time_mode != "real":
        logging.warning(
            f"--time_mode={_cli_args.time_mode!r} requested, but FedFwd has no "
            "simulated-clock support yet; running in real wall-clock mode."
        )
    set_seed(config.hyperparameters.manual_seed)

    # dataset attributes
    attributes = BaseDataManager.load_attributes(config.hyperparameters.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # create the model
    model_args = ClassificationArgs()
    model_args.model_name = config.hyperparameters.model_name
    model_args.model_type = config.hyperparameters.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.client_idx = config.hyperparameters.client_idx
    model_args.update_from_dict(
        {
            "fl_algorithm": config.hyperparameters.fl_algorithm,
            "freeze_layers": config.hyperparameters.freeze_layers,
            "epochs": config.hyperparameters.epochs,
            "learning_rate": config.hyperparameters.learning_rate,
            "gradient_accumulation_steps": config.hyperparameters.gradient_accumulation_steps,
            "do_lower_case": config.hyperparameters.do_lower_case,
            "manual_seed": config.hyperparameters.manual_seed,
            # for ignoring the cache features.
            "reprocess_input_data": False,
            "overwrite_output_dir": True,
            "max_seq_length": config.hyperparameters.max_seq_length,
            "train_batch_size": config.hyperparameters.train_batch_size,
            "eval_batch_size": config.hyperparameters.eval_batch_size,
            "evaluate_during_training": False,  # Disabled for FedAvg.
            "evaluate_during_training_steps": config.hyperparameters.evaluate_during_training_steps,
            "fp16": config.hyperparameters.fp16,
            "data_file_path": config.hyperparameters.data_file_path,
            "partition_file_path": config.hyperparameters.partition_file_path,
            "partition_method": config.hyperparameters.partition_method,
            "dataset": config.hyperparameters.dataset,
            "output_dir": config.hyperparameters.output_dir,
            "is_debug_mode": config.hyperparameters.is_debug_mode,
            "fedprox_mu": config.hyperparameters.fedprox_mu,
            "use_adapter": config.hyperparameters.use_adapter,
            "comm_round": config.hyperparameters.comm_round,
            "peft_method": config.hyperparameters.peft_method,
            "var_control": config.hyperparameters.var_control,
            "perturbation_sampling": config.hyperparameters.perturbation_sampling,
            "select_perturbation_using_jvp": config.hyperparameters.select_perturbation_using_jvp,
        }
    )
    model_args.config["num_labels"] = num_labels
    model_config, client_model, tokenizer = create_model(
        model_args, formulation="classification"
    )

    def bytes_to_mb(num_bytes):
        return num_bytes / (1024 * 1024)

    total_params = trainable_params = frozen_params = 0
    zero_trainable = zero_frozen = 0
    trainable_bytes = frozen_bytes = 0

    for name, param in client_model.named_parameters():
        numel = param.numel()
        param_bytes = numel * param.element_size()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel
            trainable_bytes += param_bytes
            if torch.all(param == 0):
                print(f"[Trainable] {name} is all zeros")
                zero_trainable += numel
        else:
            frozen_params += numel
            frozen_bytes += param_bytes
            if torch.all(param == 0):
                print(f"[Frozen] {name} is all zeros")
                zero_frozen += numel

    logging.info(f"Total parameters: {total_params}")
    logging.info(
        f"  Trainable: {trainable_params} | Zeroed: {zero_trainable} | Size: {bytes_to_mb(trainable_bytes):.2f} MB"
    )
    logging.info(
        f"  Frozen:    {frozen_params} | Zeroed: {zero_frozen} | Size: {bytes_to_mb(frozen_bytes):.2f} MB"
    )
    logging.info(f"Total size: {bytes_to_mb(trainable_bytes + frozen_bytes):.2f} MB")
    logging.info(
        f"Total zero-weighted: {zero_trainable + zero_frozen} ({100 * (zero_trainable + zero_frozen) / total_params:.2f}%)"
    )

    # data manager
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    process_id = 1
    # TODO: data_loader_num_workers is used for client partition sampling in BaseDataManager,
    # not for DataLoader parallelism (all DataLoader calls hardcode num_workers=0). Rename
    # the config field and wire it through to the DataLoader constructors in base_data_manager.py.
    dm = TextClassificationDataManager(
        config.hyperparameters,
        model_args,
        preprocessor,
        process_id,
        config.hyperparameters.data_loader_num_workers,
    )
    (
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_clients,
    ) = dm.load_federated_data(
        process_id=process_id, client_idx=config.hyperparameters.client_idx
    )
    logging.debug(f"NRL Client idx: {config.hyperparameters.client_idx}")
    logging.debug(f"NRL train_data_local_dict: {train_data_local_dict}")
    logging.debug(f"NRL train_data_global: {train_data_global}")
    logging.debug(f"NRL test_data_local_dict: {test_data_local_dict}")
    logging.debug(f"NRL test_data_global: {test_data_global}")
    logging.info(
        f"[Trainer {config.task_id}] PID: {os.getpid()}, Thread: {threading.get_ident()}"
    )
    client_trainer = ForwardTextClassificationTrainer(
        model_args,
        config.hyperparameters.client_idx % 8,
        client_model,
        None,
        None,
        config.task_id,
    )
    fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    # client manager in their code also passes client index which is the list of clients that need to do training
    trainer = FedSGDTrainer(
        config.task_id,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        config.hyperparameters.client_idx % 8,
        config.hyperparameters,
        fed_trainer,
        config,
    )

    if trainer.client_notify["trace"] is not None:
        logging.info(
            f"Will initiate thread to update state of " f"trainer {trainer.trainer_id}"
        )
        if trainer.client_notify["enabled"] == "True":
            logging.info(
                f"Will send avail notifications for trainer {trainer.trainer_id}"
            )
        # Note that even though trainer sends notifications, only
        # async_oort will use it. Other selectors will not use it so
        # it can remain enabled.
        avail_notify_thread = threading.Thread(target=trainer.notify_trainer_avail)
        avail_notify_thread.daemon = True
        avail_notify_thread.start()

    trainer.compose()
    trainer.run()
