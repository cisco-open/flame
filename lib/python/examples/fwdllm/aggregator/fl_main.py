"""FwdLLM horizontal FL aggregator for PyTorch."""

import os
import sys
import argparse
import logging

import torch
import wandb

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
from examples.fwdllm.expts.initializer import set_seed, create_model
from examples.fwdllm.trainer.forward_training.fed_trainer_transformer import (
    FedTransformerTrainer,
)
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator

from flame.config import Config

logger = logging.getLogger(__name__)


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # fedbuff
            "server_learning_rate": 40.9,
            "client_learning_rate": 0.000195,
            # oort "client_learning_rate": 0.04,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "async, fedbuff",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            # fedbuff
            "client-concurrency C": 20,
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "First oort no failure run",
        },
    )


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
    # parser = add_federated_args(parser) args = parser.parse_args()
    parser.add_argument("--config", type=str, default="./config.json", required=True)
    parser.add_argument("--log_level", type=str, default="INFO", required=False)
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # customize the log format
    logging.basicConfig(
        level=logging._nameToLevel[args.log_level],
        format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger.info(config)
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
            "client_idx": config.hyperparameters.client_idx,
        }
    )
    model_args.config["num_labels"] = num_labels
    model_config, client_model, tokenizer = create_model(
        model_args, formulation="classification"
    )

    # trainer
    client_trainer = ForwardTextClassificationTrainer(
        model_args, 0, client_model, None, None, config.task_id
    )
    fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    # data manager
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    process_id = 0
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
    logger.info(f"NRL Client idx: {config.hyperparameters.client_idx}")
    logger.info(f"NRL train_data_local_dict: {train_data_local_dict}")
    logger.info(f"NRL train_data_global: {train_data_global}")
    logger.info(f"NRL test_data_local_dict: {test_data_local_dict}")
    logger.info(f"NRL test_data_global: {test_data_global}")
    logger.info(f"NRL fed_trainer: {fed_trainer}")
    logger.info(f"NRL num_clients: {num_clients}")
    config.hyperparameters.client_num_in_total = num_clients
    config.hyperparameters.warmup_ratio = model_args.warmup_ratio

    worker_num = 1
    aggregator = FedSGDAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        config,
        fed_trainer,
        num_labels,
    )

    aggregator.compose()
    aggregator.run()
