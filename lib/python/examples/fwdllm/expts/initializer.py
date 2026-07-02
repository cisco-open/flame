from operator import mod
import random

import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)

# from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_FedAvg_distributed
# from FedML.fedml_api.distributed.fedopt.FedOptAPI import FedML_FedOpt_distributed
# from FedML.fedml_api.distributed.fedprox.FedProxAPI import FedML_FedProx_distributed
# from FedML.fedml_api.distributed.fedsgd.FedSgdAPI import FedML_FedSgd_distributed
# `adapter-transformers` (the old fork that monkeypatched `transformers.adapters`)
# is replaced by the standalone `adapters` add-on, which works on top of mainline
# `transformers`. Call `adapters.init(model)` once before `add_adapter`/
# `train_adapter`, and build bottleneck adapters via `BnConfig`.
import adapters
from adapters import BnConfig, LoRAConfig

import logging


# def get_fl_algorithm_initializer(alg_name):
#     if alg_name == "FedAvg":
#         fl_algorithm = FedML_FedAvg_distributed
#     elif alg_name == "FedSgd" or alg_name == "FedFwd":
#         fl_algorithm = FedML_FedSgd_distributed
#     elif alg_name == "FedOPT":
#         fl_algorithm = FedML_FedOpt_distributed
#     elif alg_name == "FedProx":
#         fl_algorithm = FedML_FedProx_distributed
#     else:
#         raise Exception("please do sanity check for this algorithm.")

#     return fl_algorithm


def create_model(args, formulation="classification"):

    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            ),
            "roberta-large": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizer,
            ),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "deberta": (
                DebertaConfig,
                DebertaForSequenceClassification,
                DebertaTokenizer,
            ),
        },
        "seq_tagging": {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "distilbert": (
                DistilBertConfig,
                DistilBertForTokenClassification,
                DistilBertTokenizer,
            ),
        },
        "span_extraction": {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (
                DistilBertConfig,
                DistilBertForQuestionAnswering,
                DistilBertTokenizer,
            ),
        },
        "seq2seq": {
            "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
        },
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][
        args.model_type
    ]
    # config = config_class.from_pretrained(
    #     args.model_name, num_labels=args.num_labels, **args.config)
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)
    if formulation != "seq2seq":
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name, do_lower_case=args.do_lower_case
        )
    else:
        tokenizer = [None, None]
        tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
        tokenizer[1] = tokenizer[0]
    # print('befor lorabefor lora befor lora befor lora befor lora')
    # print(model)
    logging.info(f"peft_method: {args.peft_method}")
    if args.peft_method == "adapter":
        adapters.init(model)
        adapter_config = {
            "original_ln_before": True,
            "original_ln_after": True,
            "residual_before_ln": True,
            "adapter_residual_before_ln": False,
            "ln_before": False,
            "ln_after": False,
            "mh_adapter": False,
            "output_adapter": True,
            "non_linearity": "relu",
            "reduction_factor": 16,
            "inv_adapter": None,
            "inv_adapter_reduction_factor": None,
            "cross_adapter": False,
            "leave_out": [],
        }
        model.add_adapter("rotten tomato", config=BnConfig(**adapter_config))
        model.train_adapter("rotten tomato")
    elif args.peft_method == "lora":
        adapters.init(model)
        config = LoRAConfig(r=8, alpha=16)
        model.add_adapter("lora_adapter", config=config)
        model.train_adapter("lora_adapter")
    elif args.peft_method == "bitfit":
        for n, p in model.named_parameters():
            if not ("bias" in n or "classifier" in n):
                p.requires_grad = False
    # print("after lora after lora after lora after lora after lora")
    # print(model)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return config, model, tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_federated_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--client_idx", type=int, default=-1)

    parser.add_argument("--is_debug_mode", default=0, type=int, help="is_debug_mode")

    # Data related
    # TODO: list all dataset names:
    parser.add_argument(
        "--dataset",
        type=str,
        default="agnews",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="/home/bill/fednlp_data/data_files/agnews_data.h5",
        help="data h5 file path",
    )

    parser.add_argument(
        "--partition_file_path",
        type=str,
        default="/home/bill/fednlp_data/partition_files/agnews_partition.h5",
        help="partition h5 file path",
    )

    parser.add_argument(
        "--partition_method", type=str, default="uniform", help="partition method"
    )

    # Model related
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        metavar="N",
        help="transformer model type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        metavar="N",
        help="transformer model name",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        metavar="N",
        help="transformer model name",
    )

    # Learning related
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for evaluation (default: 8)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        metavar="N",
        help="maximum sequence length (default: 128)",
    )

    parser.add_argument(
        "--n_gpu", type=int, default=1, metavar="EP", help="how many gpus will be used "
    )

    parser.add_argument(
        "--fp16", default=False, action="store_true", help="if enable fp16 for training"
    )
    parser.add_argument(
        "--manual_seed", type=int, default=42, metavar="N", help="random seed"
    )

    # IO related
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/",
        metavar="N",
        help="path to save the trained results and ckpts",
    )

    # Federated Learning related
    parser.add_argument(
        "--fl_algorithm",
        type=str,
        default="FedAvg",
        help="Algorithm list: FedAvg; FedOPT; FedProx ",
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=0,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=-1,
        metavar="NN",
        help="number of clients in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=4,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        metavar="EP",
        help="how many steps for accumulate the loss.",
    )

    parser.add_argument(
        "--client_optimizer",
        type=str,
        default="adam",
        help="Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate on the client (default: 0.001)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0, metavar="N", help="L2 penalty"
    )

    parser.add_argument(
        "--server_optimizer",
        type=str,
        default="sgd",
        help="Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.1,
        help="server learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--server_momentum", type=float, default=0, help="server momentum (default: 0)"
    )

    parser.add_argument(
        "--fedprox_mu", type=float, default=1, help="server momentum (default: 1)"
    )
    parser.add_argument(
        "--evaluate_during_training_steps",
        type=int,
        default=100,
        metavar="EP",
        help="the frequency of the evaluation during training",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=1,
        help="the frequency of the algorithms",
    )

    # GPU device management
    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    # cached related
    parser.add_argument(
        "--reprocess_input_data", action="store_true", help="whether generate features"
    )

    # freeze related
    parser.add_argument(
        "--freeze_layers", type=str, default="", metavar="N", help="freeze which layers"
    )

    parser.add_argument(
        "--use_adapter", type=bool, default=False, metavar="N", help="use_adapter"
    )

    parser.add_argument(
        "--forward_mode", action="store_true", help="whether forward_mode"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning_rate"
    )
    parser.add_argument("--worker_num", type=int, default=1, help="worker_num")
    parser.add_argument("--peft_method", type=str, default="full", help="peft_method")

    parser.add_argument(
        "--var_control", action="store_true", help="whether var_control"
    )

    parser.add_argument(
        "--perturbation_sampling",
        action="store_true",
        help="whether perturbation_sampling",
    )

    parser.add_argument(
        "--select_perturbation_using_jvp", type=bool, default=False, metavar="N", help="use_jvp_to_select_best_perturbation"
    )

    return parser
