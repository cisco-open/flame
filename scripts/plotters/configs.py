from datetime import datetime
import re
from typing import List, Dict, Any, Callable, Optional, Tuple

# --- Config related handlers ---
def handle_stat_utility(parser: 'LogParser', match: re.Match) -> Dict[str, Any]:
    trainer_id = match.group('trainer_id')
    trainer_state = parser.keyed_state[trainer_id]
    round_num = trainer_state.get('round')
    data_id = trainer_state.get('data_id')
    iteration_key = (trainer_id, round_num, data_id)
    parser.iteration_tracker[iteration_key] += 1
    iteration = parser.iteration_tracker[iteration_key]
    return {'round': round_num, 'data_id': data_id, 'iteration': iteration}

# --- Configs ---
LOG_CONFIG = {
    'flame_fwdllm_aggregator': [
        {
            'name': 'first_distribute_weights',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*fwdllm_aggregator\.py.*_distribute_weights.*sending weights"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
        {
            'name': 'extract_stat_utility',
            'regex': re.compile(r"stat_utility for trainerId: (?P<trainer_id>\w+) is (?P<stat>[\d\.]+), loss: (?P<loss>[\d\.]+)"),
            'type': 'EXTRACT',
            # 'extract_key_group': 'trainer_id',
            'group_to_columns': {'trainer_id': ('trainer_id', str), 'stat': ('stat_utility', float), 'loss': ('loss', float)},
            'handler': handle_stat_utility
        },
        {
            'name': 'eval_model',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*"
                r"'acc':\s*(?P<acc>[\d.]+).*'data_id_iterations':\s*(?P<data_id_iterations>\d+)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'acc': ('accuracy', lambda x: float(x) * 100),
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')),
                'data_id_iterations': ('data_id_iterations', int)
            }
        },
        {
            'name': 'var',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*"
                r"self\.var\s*=\s*(?P<self_var>[\d.]+), self.var_threshold"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'self_var': ('var', float),
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S'))
            }
        },
    ],
    'flame_fwdllm_trainer': [
        {
            'name': 'train_time',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                r"Runtime of train_with_data_id:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('train_time_sec', float),
                'round_id': ('round_id', lambda round_id: int(round_id) - 1),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', str)
            }
        },
        {
            'name': 'recv_weights_time',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                r"Runtime of recv_wrapper:\s(?P<runtime>[\d\.]+)s\s"
                r"\(Round=(?P<round_id>\d+),\sDataId=(?P<data_id>\d+),\sIter=(?P<iter_id>\d+),\sTrainerId=(?P<trainer_id>\w+)\)"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                'runtime': ('recv_weights_time', float),
                'round_id': ('round_id', int),
                'data_id': ('data_id', int),
                'iter_id': ('iteration_id', int),
                'trainer_id': ('trainer_id', str)
            }
        },
    ],
    'flame_fwdllm_trainer_old': [
        {
            'name': 'train_time',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                # r"Runtime of train is \s(?P<runtime>[\d\.]+)$"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
                # 'runtime': ('train_time_sec', float)
            }
        },
        {
            'name': 'first_distribute_weights',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*"
                r"New message received for trainer_id 505f9fc483cf4df68a2409257b5fad7d3c580372"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
    ],
    # --- NEW CONFIG GROUP ---
    'Async-Cifar-10': [
        {
            'name': 'agg_train_sent',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*sending weights.*task: train"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
        {
            'name': 'agg_eval_sent',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*sending weights.*task: eval"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
        {
            'name': 'agg_weight_recv',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*Received weights.*trained on model version"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
        {
            'name': 'agg_eval_recv',
            'regex': re.compile(
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}).*received eval message"
            ),
            'type': 'EXTRACT',
            'group_to_columns': {
                'timestamp': ('timestamp', lambda ts_str: datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')),
            }
        },
    ]
}

EXPORT_CONFIG = {
    'flame_fwdllm_aggregator': {
        'evaluation_metrics': {
            # 'default_output_filename': '19Oct_slow_straggler_evaluation_metrics.csv',
            'default_output_filename': 'eval_agg_k10_n50_rnd1_acc70_delayBy20_19_10_05_18.csv',
            'log_names': ['eval_model'],
            'columns': ['timestamp', 'time_since_start', 'round_id', 'data_id', 'accuracy']
        },
        # 'trainer_performance': {
        #     'default_output_filename': 'trainer_performance.csv',
        #     'log_names': ['extract_stat_utility'],
        #     'columns': ['timestamp', 'trainer_id', 'trainer_num', 'loss', 'stat_utility']
        # },
    },
    'flame_fwdllm_trainer': {
        'train_times': {
            # 'default_output_filename': 'train_times_delay_slow_6hr.csv',
            # 'default_output_filename': 'train_times_noDelay_slow_4hr.csv',
            # 'default_output_filename': 'train_times_delayBy20_3hrs.csv',
            'default_output_filename': 'train_times_delayBy3_8hrs.csv',
            'log_names': ['recv_weights_time'],
            'columns': ['timestamp', 'round_id', 'data_id', 'iteration_id', 'train_time_sec',
                        # 'cumulative_train_time_sec', 'mean:cumulative_train_time_sec',
                        'cumulative_recv_weights_time', 'mean:cumulative_recv_weights_time', 
                        'time_since_start',
                        'trainer_num', '_model_version',
                        # 'sum:recv_weights_time', 'mean:sum:recv_weights_time',
                        # 'sum:train_time_sec', 'mean:sum:train_time_sec',
                        # 'trainer_id',
                        ]
        }
    },
    'flame_fwdllm_trainer_old': {
        'train_times': {
            'default_output_filename': 'old_train_times.csv',
            'log_names': ['train_time', 'first_distribute_weights'],
            'columns': ['timestamp', 'round_id', 'data_id', 'iteration_id',
                        'train_time_sec',
                        # 'mean:train_time_sec', 'sum:train_time_sec',
                        'cumulative_train_time_sec', 'mean:cumulative_train_time_sec',
                        'time_since_start',
                        'trainer_num',
                        # 'trainer_id',
                        ]
        }
    },
    'Async-Cifar-10': {
        'communication_raw': {
            'output_filename': 'communication_raw.csv',
            'log_names': ['agg_train_sent', 'agg_eval_sent', 'agg_weight_recv', 'agg_eval_recv'],
            'columns': ['log_name', 'timestamp']
        },
        'communication_summary': {
            'output_filename': f'communication_summary.csv',                # todo: make it easier to declartively add suffix here
            'log_names': ['agg_train_sent', 'agg_eval_sent', 'agg_weight_recv', 'agg_eval_recv'],
            'columns': ['log_name', 'count:timestamp']
        }
    },
}