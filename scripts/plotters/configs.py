from datetime import datetime
import re
from typing import List, Dict, Any, Callable, Optional, Tuple


# --- Config related handlers ---
def handle_stat_utility(parser: "LogParser", match: re.Match) -> Dict[str, Any]:
    trainer_id = match.group("trainer_id")
    trainer_state = parser.keyed_state[trainer_id]
    round_num = trainer_state.get("round")
    data_id = trainer_state.get("data_id")
    iteration_key = (trainer_id, round_num, data_id)
    parser.iteration_tracker[iteration_key] += 1
    iteration = parser.iteration_tracker[iteration_key]
    return {"round": round_num, "data_id": data_id, "iteration": iteration}


# --- Configs ---
# Note: Patterns are checked in order & only the first match is used
LOG_CONFIG = {
    "flame_fwdllm_aggregator": [
        {
            "name": "first_distribute_weights",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*fwdllm_aggregator\.py.*_distribute_weights.*sending weights"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
        {
            "name": "extract_stat_utility",
            "regex": re.compile(
                r"stat_utility for trainerId: (?P<trainer_id>\w+) is (?P<stat>[\d\.]+), loss: (?P<loss>[\d\.]+)"
            ),
            "type": "EXTRACT",
            # 'extract_key_group': 'trainer_id',
            "group_to_columns": {
                "trainer_id": ("trainer_id", str),
                "stat": ("stat_utility", float),
                "loss": ("loss", float),
            },
            "handler": handle_stat_utility,
        },
        {
            "name": "eval_model",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*"
                r"'acc':\s*(?P<acc>[\d.]+).*'data_id_iterations':\s*(?P<data_id_iterations>\d+)"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "acc": ("accuracy", lambda x: float(x) * 100),
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
                ),
                "data_id_iterations": ("data_id_iterations", int),
            },
        },
        {
            "name": "var",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*"
                r"self\.var\s*=\s*(?P<self_var>[\d.]+), self.var_threshold"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "self_var": ("var", float),
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
                ),
            },
        },
        {
            'name': 'distribute',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"Runtime of distribute is (?P<runtime>[\d\.]+)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('distribute_latency', float)
            }
        },
        {
            'name': 'aggregate',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"Runtime of aggregate is (?P<runtime>[\d\.]+)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('aggregate_latency', float)
            }
        },
        {
            'name': 'eval_model_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of eval_model:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('eval_model_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'collect_and_accumulate_grads_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of collect_and_accumulate_grads:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('collect_and_accumulate_grads_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'aggregate_runtime',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of aggregate:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('aggregate_runtime', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'aggregate_grads_sync_latency',     # As compared to aggregate_grads, it contains iterationId
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _aggregate_grads_sync:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('aggregate_grads_sync_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'process_single_trainer_message_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _process_single_trainer_message:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('process_single_trainer_message_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'process_aggregation_goal_met_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _process_aggregation_goal_met:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('process_aggregation_goal_met_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'agg_force_cuda_memory_cleanup_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _force_cuda_memory_cleanup:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('agg_force_cuda_memory_cleanup_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'prepare_distribution_payload_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _prepare_distribution_payload:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('prepare_distribution_payload_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'distribute_weights_sync_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _distribute_weights_sync:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('distribute_weights_sync_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
        {
            'name': 'distribute_weights_async_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _distribute_weights_async:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('distribute_weights_async_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', None)
            }
        },
    ],
    "flame_fwdllm_trainer": [
        {
            'name': 'decorator_train_time',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of train_with_data_id:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('train_time_sec', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', str)
            }
        },
        # The train_time pattern matches even witht the decorator_train_time pattern & the parser just chooses the first one & moves on.
        # {
        #     'name': 'train_time',
        #     'regex': re.compile(
        #         r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
        #         r"Runtime of train_with_data_id:\s(?P<runtime>[\d\.]+)s\s"
        #         r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
        #     ),
        #     'type': 'EXTRACT',
        #     'group_to_columns': {
        #         'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
        #         'runtime': ('train_time_sec', float),
        #         'round_id': ('round_id', lambda round_id: int(round_id) - 1),
        #         'data_id': ('data_id', int),
        #         'iter_id': ('iteration_id', int),
        #         'trainer_id': ('trainer_id', str)
        #     }
        # },
        {
            "name": "recv_weights_time",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                r"Runtime of recv_wrapper:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
                "runtime": ("recv_weights_time", float),
                "round_id": ("round_id", int),
                "data_id": ("data_id", int),
                "iter_id": ("iteration_id", int),
                "trainer_id": ("trainer_id", str),
            },
        },
        {
            'name': 'send_grads',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _send_grads:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('send_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', str)
            }
        },
        {
            'name': 'fetch_weights',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _fetch_weights:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('recv_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'pause_execution',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of pause_execution:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('pause_latency', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', str)
            }
        },
        {
            'name': 'make_model_functional_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _make_model_functional:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('make_model_functional_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'select_optimal_perturbations_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _select_optimal_perturbations:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('select_optimal_perturbations_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'compute_batch_stat_utility_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _compute_batch_stat_utility:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('compute_batch_stat_utility_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'prepare_perturbation_tensors_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _prepare_perturbation_tensors:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('prepare_perturbation_tensors_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'compute_forward_jvp_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _compute_forward_jvp:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('compute_forward_jvp_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'accumulate_and_extract_grads_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _accumulate_and_extract_grads:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('accumulate_and_extract_grads_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
        {
            'name': 'force_cuda_memory_cleanup_latency',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*?"
                r"\[decorator\]\sRuntime of _force_cuda_memory_cleanup:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>None|\d+),\sDataId=(?P<data_id>None|\d+),\sIter=(?P<iter_id>None|\d+),\sTrainerId=(?P<trainer_id>\w+|None)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('force_cuda_memory_cleanup_latency', float),
                'round_id': ('round_id', lambda x: None if x == 'None' else int(x)),
                'data_id': ('data_id', lambda x: None if x == 'None' else int(x)),
                'iter_id': ('iteration_id', lambda x: None if x == 'None' else int(x)),
                'trainer_id': ('trainer_id', lambda x: None if x == 'None' else str(x))
            }
        },
    ],
    "flame_fwdllm_trainer_old": [
        {
            "name": "train_time",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                # r"Runtime of train is \s(?P<runtime>[\d\.]+)$"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
                # 'runtime': ('train_time_sec', float)
            },
        },
        {
            "name": "first_distribute_weights",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                r"New message received for trainer_id 505f9fc483cf4df68a2409257b5fad7d3c580372"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
    ],
    # --- NEW CONFIG GROUP ---
    "Async-Cifar-10": [
        {
            "name": "agg_train_sent",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*sending weights.*task: train"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
        {
            "name": "agg_eval_sent",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*sending weights.*task: eval"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
        {
            "name": "agg_weight_recv",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*Received weights.*trained on model version"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
        {
            "name": "agg_eval_recv",
            "regex": re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*received eval message"
            ),
            "type": "EXTRACT",
            "group_to_columns": {
                "timestamp": (
                    "timestamp",
                    lambda ts_str: datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"),
                ),
            },
        },
    ],
}

CONSTANTS = {
    'file_prefix': '',
}

EXPORT_CONFIG = {
    # TODO: Change all names to use the common prefix
    'flame_fwdllm_aggregator': {
        'evaluation_metrics': {
            # 'default_output_filename': '19Oct_slow_straggler_evaluation_metrics.csv',
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-evaluation_metrics.csv',
            # 'default_output_filename': f'evaluation_metrics.csv',
            'log_names': ['eval_model'],
            'columns': ['timestamp', 'time_since_start', 'round_id', 'data_id', 'accuracy', 'data_id_iterations']
        },
        # 'trainer_performance': {
        #     'default_output_filename': 'trainer_performance.csv',
        #     'log_names': ['extract_stat_utility'],
        #     'columns': ['timestamp', 'trainer_id', 'trainer_num', 'loss', 'stat_utility']
        # },
        'distribute_latency': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-distribute_latency.csv',
            'log_names': ['distribute'],
            'columns': ['timestamp', 'distribute_latency']
        },
        'aggregate_latency': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-aggregate_latency.csv',
            'log_names': ['aggregate'],
            'columns': ['timestamp', 'aggregate_latency']
        },
        'eval_model_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-eval_model_latency.csv',
            'log_names': ['eval_model_latency'],
            'columns': ['timestamp', 'eval_model_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'collect_and_accumulate_grads_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-collect_and_accumulate_grads_latency.csv',
            'log_names': ['collect_and_accumulate_grads_latency'],
            'columns': ['timestamp', 'collect_and_accumulate_grads_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'aggregate_runtime_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-aggregate_runtime_latency.csv',
            'log_names': ['aggregate_runtime'],
            'columns': ['timestamp', 'aggregate_runtime', 'round_id', 'data_id', 'iteration_id']
        },
        'aggregate_grads_sync_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-aggregate_grads_sync_latency.csv',
            'log_names': ['aggregate_grads_sync_latency'],
            'columns': ['timestamp', 'aggregate_grads_sync_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'process_single_trainer_message_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-process_single_trainer_message_latency.csv',
            'log_names': ['process_single_trainer_message_latency'],
            'columns': ['timestamp', 'process_single_trainer_message_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'process_aggregation_goal_met_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-process_aggregation_goal_met_latency.csv',
            'log_names': ['process_aggregation_goal_met_latency'],
            'columns': ['timestamp', 'process_aggregation_goal_met_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'agg_force_cuda_memory_cleanup_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-agg_force_cuda_memory_cleanup_latency.csv',
            'log_names': ['agg_force_cuda_memory_cleanup_latency'],
            'columns': ['timestamp', 'agg_force_cuda_memory_cleanup_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'prepare_distribution_payload_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-prepare_distribution_payload_latency.csv',
            'log_names': ['prepare_distribution_payload_latency'],
            'columns': ['timestamp', 'prepare_distribution_payload_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'distribute_weights_sync_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-distribute_weights_sync_latency.csv',
            'log_names': ['distribute_weights_sync_latency'],
            'columns': ['timestamp', 'distribute_weights_sync_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'distribute_weights_async_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-distribute_weights_async_latency.csv',
            'log_names': ['distribute_weights_async_latency'],
            'columns': ['timestamp', 'distribute_weights_async_latency', 'round_id', 'data_id', 'iteration_id']
        },
    },
    'flame_fwdllm_trainer': {
        # 'train_times': {
        #     # 'default_output_filename': 'train_times_delay_slow_6hr.csv',
        #     # 'default_output_filename': 'train_times_noDelay_slow_4hr.csv',
        #     # 'default_output_filename': 'train_times_delayBy20_3hrs.csv',
        #     'default_output_filename': 'train_times_delayBy3_8hrs.csv',
        #     'log_names': ['recv_weights_time'],
        #     'columns': ['timestamp', 'round_id', 'data_id', 'iteration_id', 'train_time_sec',
        #                 # 'cumulative_train_time_sec', 'mean:cumulative_train_time_sec',
        #                 'cumulative_recv_weights_time', 'mean:cumulative_recv_weights_time', 
        #                 'time_since_start',
        #                 'trainer_num', '_model_version',
        #                 # 'sum:recv_weights_time', 'mean:sum:recv_weights_time',
        #                 # 'sum:train_time_sec', 'mean:sum:train_time_sec',
        #                 # 'trainer_id',
        #                 ]
        # },
        'train_latency': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-train_latency.csv',
            'log_names': ['decorator_train_time'],
            'columns': ['timestamp', 'trainer_id', 'train_time_sec', 'round_id', 'data_id', 'iteration_id']
        },
        'send_latency': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-send_latency.csv',
            'log_names': ['send_grads'],
            'columns': ['timestamp', 'trainer_id', 'send_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'recv_latency': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-recv_latency.csv',
            'log_names': ['fetch_weights'],
            'columns': ['timestamp', 'trainer_id', 'recv_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'pause_execution': {
            'default_output_filename': lambda: f'{CONSTANTS['file_prefix']}-pause_execution.csv',
            'log_names': ['pause_execution'],
            'columns': ['timestamp', 'trainer_id', 'pause_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'make_model_functional_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-make_model_functional_latency.csv',
            'log_names': ['make_model_functional_latency'],
            'columns': ['timestamp', 'trainer_id', 'make_model_functional_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'select_optimal_perturbations_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-select_optimal_perturbations_latency.csv',
            'log_names': ['select_optimal_perturbations_latency'],
            'columns': ['timestamp', 'trainer_id', 'select_optimal_perturbations_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'compute_batch_stat_utility_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-compute_batch_stat_utility_latency.csv',
            'log_names': ['compute_batch_stat_utility_latency'],
            'columns': ['timestamp', 'trainer_id', 'compute_batch_stat_utility_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'prepare_perturbation_tensors_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-prepare_perturbation_tensors_latency.csv',
            'log_names': ['prepare_perturbation_tensors_latency'],
            'columns': ['timestamp', 'trainer_id', 'prepare_perturbation_tensors_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'compute_forward_jvp_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-compute_forward_jvp_latency.csv',
            'log_names': ['compute_forward_jvp_latency'],
            'columns': ['timestamp', 'trainer_id', 'compute_forward_jvp_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'accumulate_and_extract_grads_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-accumulate_and_extract_grads_latency.csv',
            'log_names': ['accumulate_and_extract_grads_latency'],
            'columns': ['timestamp', 'trainer_id', 'accumulate_and_extract_grads_latency', 'round_id', 'data_id', 'iteration_id']
        },
        'force_cuda_memory_cleanup_latency': {
            'default_output_filename': lambda: f'{CONSTANTS["file_prefix"]}-force_cuda_memory_cleanup_latency.csv',
            'log_names': ['force_cuda_memory_cleanup_latency'],
            'columns': ['timestamp', 'trainer_id', 'force_cuda_memory_cleanup_latency', 'round_id', 'data_id', 'iteration_id']
        }
    },
    "flame_fwdllm_trainer_old": {
        "train_times": {
            "default_output_filename": "old_train_times.csv",
            "log_names": ["train_time", "first_distribute_weights"],
            "columns": [
                "timestamp",
                "round_id",
                "data_id",
                "iteration_id",
                "train_time_sec",
                # 'mean:train_time_sec', 'sum:train_time_sec',
                "cumulative_train_time_sec",
                "mean:cumulative_train_time_sec",
                "time_since_start",
                "trainer_num",
                # 'trainer_id',
            ],
        }
    },
    "Async-Cifar-10": {
        "communication_raw": {
            "output_filename": "communication_raw.csv",
            "log_names": [
                "agg_train_sent",
                "agg_eval_sent",
                "agg_weight_recv",
                "agg_eval_recv",
            ],
            "columns": ["log_name", "timestamp"],
        },
        "communication_summary": {
            "output_filename": f"communication_summary.csv",  # todo: make it easier to declartively add suffix here
            "log_names": [
                "agg_train_sent",
                "agg_eval_sent",
                "agg_weight_recv",
                "agg_eval_recv",
            ],
            "columns": ["log_name", "count:timestamp"],
        },
    },
}
