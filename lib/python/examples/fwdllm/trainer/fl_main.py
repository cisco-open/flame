import os
import socket
import sys

import psutil
import setproctitle
import torch
import threading

# this is a temporal import, we will refactor FedML as a package installation
# import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

# from training.fed_trainer_transformer import FedTransformerTrainer
from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (
    TLMPreprocessor,
)

# from training.tc_transformer_trainer import TextClassificationTrainer
from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (
    ForwardTextClassificationTrainer,
)

# from training.tc_transformer_trainer_distribute import TextClassificationTrainer as SgdTextClassificationTrainer
from examples.fwdllm.trainer.model.transformer.model_args import ClassificationArgs
from examples.fwdllm.data_manager.text_classification_data_manager import (
    TextClassificationDataManager,
)
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager

# from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
# from FedML.fedml_api.distributed.fedavg_beifen.FedAvgAPI import FedML_init
from examples.fwdllm.expts.initializer import (
    add_federated_args,
    set_seed,
    create_model,
)  # , \
from examples.fwdllm.trainer.forward_training.fed_trainer_transformer import (
    FedTransformerTrainer,
)
from examples.fwdllm.trainer.forward_training.FedSgdTrainer import FedSGDTrainer

# get_fl_algorithm_initializer

import argparse
import logging
from flame.config import Config

# class FwdLLMTrainer(Trainer):


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    # parser = add_federated_args(parser)
    # args = parser.parse_args()
    parser.add_argument("--config", type=str, default="./config.json", required=True)
    args = parser.parse_args()
    config = Config(args.config)

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.debug(config)
    set_seed(config.hyperparameters.manual_seed)

    # # initialize distributed computing (MPI)
    # comm, process_id, worker_number = FedML_init()

    # # customize the process name
    # str_process_name = "FedNLP-" + str(args.dataset) + ":" + str(process_id)
    # setproctitle.setproctitle(str_process_name)

    # hostname = socket.gethostname()
    # logging.info("#############process ID = " + str(process_id) +
    #              ", host name = " + hostname + "########" +
    #              ", process ID = " + str(os.getpid()) +
    #              ", process Name = " + str(psutil.Process(os.getpid())))

    # # logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # # if process_id == 0:
    # #     # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    # #     wandb.init(project="fednlp", entity="automl", name="FedNLP-" + str(args.fl_algorithm) +
    # #                                                        "-TC-" + str(args.dataset) + "-" + str(
    # #         args.model_name) + "-freeze-" + args.freeze_layers if args.freeze_layers else "",
    # #                config=args)

    # # device: check "gpu_mapping.yaml" to see how to define the topology
    # device = mapping_processes_to_gpu_device_from_yaml_file(
    #     process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    # logging.info("process_id = %d, size = %d, device=%s" %
    #              (process_id, worker_number, str(device)))
    # logging.info("torch.cuda.current_device()=" + str(torch.cuda.current_device()))
    # logging.info("torch.cuda.device_count()=" + str(torch.cuda.device_count()))

    # dataset attributes
    attributes = BaseDataManager.load_attributes(config.hyperparameters.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # create the model
    model_args = ClassificationArgs()
    model_args.model_name = config.hyperparameters.model_name
    model_args.model_type = config.hyperparameters.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.client_num_per_round = config.hyperparameters.client_num_per_round
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
    logging.info(f"  Trainable: {trainable_params} | Zeroed: {zero_trainable} | Size: {bytes_to_mb(trainable_bytes):.2f} MB")
    logging.info(f"  Frozen:    {frozen_params} | Zeroed: {zero_frozen} | Size: {bytes_to_mb(frozen_bytes):.2f} MB")
    logging.info(f"Total size: {bytes_to_mb(trainable_bytes + frozen_bytes):.2f} MB")
    logging.info(f"Total zero-weighted: {zero_trainable + zero_frozen} ({100 * (zero_trainable + zero_frozen) / total_params:.2f}%)")

    # trainer

    # if args.fl_algorithm == "FedFwd":
    #     client_trainer = ForwardTextClassificationTrainer(
    # model_args, device, client_model, None, None)
    # elif args.fl_algorithm == "FedSgd":
    #     client_trainer = SgdTextClassificationTrainer(
    #         model_args, device, client_model, None, None)
    # else:
    #     client_trainer = TextClassificationTrainer(
    #         model_args, device, client_model, None, None)
    # fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    # data manager
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    process_id = 1
    dm = TextClassificationDataManager(
        config.hyperparameters,
        model_args,
        preprocessor,
        process_id,
        config.hyperparameters.client_num_per_round,
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
    logging.info(f"[Trainer {config.task_id}] PID: {os.getpid()}, Thread: {threading.get_ident()}")
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
    trainer.compose()
    trainer.run()
    # # start FedAvg algorithm
    # # for distributed algorithm, train_data_gloabl and test_data_global are required
    # if process_id == 0:
    #     client_trainer.test_dl = test_data_global
    # args.client_num_in_total = num_clients
    # args.warmup_ratio = model_args.warmup_ratio
    # # args.client_num_per_round = 500
    # # args.learning_rate = 0.01

    # fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    # fl_algorithm(process_id, worker_number, device, comm, client_model, train_data_num,
    #              train_data_global, test_data_global, train_data_local_num_dict,
    #              train_data_local_dict, test_data_local_dict, args, fed_trainer)
