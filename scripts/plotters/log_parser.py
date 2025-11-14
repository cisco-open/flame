import re, os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

# --- Row-wise Processors ---


def create_numeric_id_processor(source_col: str, dest_col: str) -> Callable:
    """
    Factory for a row-processor that assigns a unique, sequential integer ID
    to unique values in a source column.
    """
    def process(row: pd.Series, state: dict) -> pd.Series:
        # State stores the mapping and the next available ID
        id_map = state.setdefault(f'numeric_id_map_{source_col}', {})
        next_id = state.setdefault(f'numeric_id_next_{source_col}', 0)

        source_value = row[source_col]
        if pd.notna(source_value):
            if source_value not in id_map:
                id_map[source_value] = next_id
                state[f'numeric_id_next_{source_col}'] += 1
            row[dest_col] = id_map[source_value]
        else:
            row[dest_col] = pd.NA
        return row
    return process


def create_sequential_id_processor(eval_log_name: str, iter_log_name: str) -> Callable:
    """Factory to create a row-processor for sequential and iterative IDs."""
    def process(row: pd.Series, state: dict) -> pd.Series:
        seq_counter = state.setdefault('sequential_id_counter', 0)
        iter_counter = state.setdefault('iteration_id_counter', 0)

        # Retrieve the last round/data ID from state to carry forward
        round_id = seq_counter // 150
        data_id = seq_counter % 150

        if row['log_name'] == eval_log_name:
            row['round_id'], row['data_id'], row['iteration_id'] = round_id, data_id, iter_counter
            state['sequential_id_counter'] += 1
            state['iteration_id_counter'] = 0
            state['current_round_id'], state['current_data_id'] = round_id, data_id
        elif row['log_name'] == iter_log_name:
            # Use the stored round/data ID from the last eval_log
            row['round_id'] = round_id
            row['data_id'] = data_id
            # Assign current iteration count
            row['iteration_id'] = iter_counter

            # Update state for the next 'var' log
            state['iteration_id_counter'] += 1

        else:
            row['round_id'], row['data_id'], row['iteration_id'] = pd.NA, pd.NA, pd.NA
        return row
    return process


def create_time_calculator_processor(start_log_name: str) -> Callable:
    """Factory to create a row-processor that calculates time since a start event."""
    def process(row: pd.Series, state: dict) -> pd.Series:
        # Set the start time in state when the specific log is encountered.
        # setdefault ensures this is only set on the first occurrence.
        if row['log_name'] == start_log_name and pd.notna(row['timestamp']):
            state.setdefault('training_start_time', row['timestamp'])

        # Calculate time difference if start time is set and current row has a timestamp
        if 'training_start_time' in state and pd.notna(row['timestamp']):
            datetime_diff = row['timestamp'] - state['training_start_time']
            row['time_since_start'] = datetime_diff.total_seconds()
        else:
            row['time_since_start'] = pd.NaT
        return row
    return process


def create_cumulative_sum_processor(group_key_col: str, target_cols: List[str]) -> Callable:
    """
    Factory for a row-processor that calculates cumulative sums for target columns,
    grouped by a specific key (e.g., 'trainer_id').
    """
    def process(row: pd.Series, state: dict) -> pd.Series:
        # Initialize a nested dictionary in state to hold sums if it doesn't exist
        # e.g., {'cumulative_sums': {'trainer1': {'train_time': 10, 'stall_time': 2}, ...}}
        sums_cache = state.setdefault(
            'cumulative_sums', defaultdict(lambda: defaultdict(float)))

        group_key = row[group_key_col]
        if pd.isna(group_key):
            return row  # Cannot process if the group key is missing

        for col in target_cols:
            if pd.notna(row[col]):
                sums_cache[group_key][col] += row[col]
            # Add the new cumulative column to the row
            row[f'cumulative_{col}'] = sums_cache[group_key][col]

        return row
    return process

# --- DataFrame Processors ---


def create_broadcast_aggregator(group_by_cols: List[str], aggregations: Dict[str, List[str]]) -> Callable:
    """
    Factory for a DataFrame-processor that performs groupby aggregations and
    broadcasts the results back to the original rows.
    """
    def process(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not all(col in df.columns for col in group_by_cols):
            return df

        print(
            f"\nApplying broadcast aggregation grouped by {group_by_cols}...")

        # Make a copy to avoid SettingWithCopyWarning
        df_out = df.copy()

        # Group the dataframe
        grouped = df_out.groupby(group_by_cols)

        # Iterate through the desired aggregations
        for col, agg_funcs in aggregations.items():
            if col not in df_out.columns:
                print(
                    f"  - Warning: Column '{col}' not found for aggregation. Skipping.")
                continue

            for agg_func in agg_funcs:
                if agg_func not in ['mean', 'sum', 'max', 'min', 'count']:
                    print(f"  - Warning: Aggregation '{agg_func}' not supported for broadcasting. Skipping.")
                    continue
                
                new_col_name = f"{agg_func}:{col}"
                print(f"  - Calculating '{new_col_name}'...")

                # Use transform to calculate aggregation and align it back to the original df index
                df_out[new_col_name] = grouped[col].transform(agg_func)

        return df_out
    return process


def create_summarization_processor(group_by_col: str, aggregations: Dict[str, Any]) -> Callable:
    """
    Factory for a DataFrame-processor that performs groupby 'log_name'
    and appends the results as new summary rows with new log_names.
    """
    def process(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or group_by_col not in df.columns:
            return df
        
        print(f"\nApplying summarization grouped by '{group_by_col}'...")
        
        # --- START CHANGE ---
        # Perform the aggregation. This creates a new, small DataFrame.
        # e.g., df.groupby('log_name').agg({'timestamp': 'count'})
        summary_df = df.groupby(group_by_col).agg(aggregations)
        
        # Rename columns for clarity, e.g., 'timestamp' -> 'count:timestamp'
        summary_df.columns = [f"{agg_func}:{col}" for col, agg_func_list in aggregations.items() for agg_func in (agg_func_list if isinstance(agg_func_list, list) else [agg_func_list])]

        # Reset the index so the group_by column (e.g., 'log_name') becomes a regular column
        summary_df.reset_index(inplace=True)
        
        # This is your requested fix:
        # Create a new log_name based on the grouping key
        # e.g., 'agg_train_sent' -> 'summary:agg_train_sent'
        summary_df['log_name'] = summary_df['log_name'].apply(lambda x: f'{x}')
        
        print(f"  - Generated {len(summary_df)} summary rows with 'summary:' prefix.")
        print(summary_df)
        # print(df)
        # --- END CHANGE ---

        # Concatenate the original DataFrame with the new summary rows
        # return pd.concat([df, summary_df], ignore_index=True)             # todo: deug why this was returning 4 rows instead of 2
        return summary_df
    
    return process


def apply_oort_comm_fix(group_by_col: str, concurrency: int = 13) -> Callable:
    """
    Factory for a DataFrame-processor that applies the special OORT
    communication counting logic.
    """
    def process(df: pd.DataFrame) -> pd.DataFrame:
        print(f"\nApplying OORT communication fix...")
        
        # --- START CHANGE ---
        # Reconstruct the names based on the convention from create_summarization_processor
        summary_log_name = group_by_col
        # group_key_col = f'{group_key_col_base}:{group_by_col}'
        
        # Extract the summary rows
        summary_rows = df[df['log_name'] == summary_log_name].copy()
        if summary_rows.empty:
            print(f"  - Warning: No summary rows found ('{summary_log_name}'). Skipping fix.")
            return df
            
        # Find the counts for train_sent and weight_recv
        try:
            train_sent_count = summary_rows.loc[summary_rows['log_name'] == 'agg_train_sent', 'count:timestamp'].iloc[0]
            weight_recv_count = summary_rows.loc[summary_rows['log_name'] == 'agg_weight_recv', 'count:timestamp'].iloc[0]

            # Apply the logic ONLY if weight_recv_count is 0
            if weight_recv_count == 0:
                print(f"  - Found agg_weight_recv=0. Applying fix...")
                new_weight_recv_count = train_sent_count - concurrency
                
                # Update the count in the main DataFrame
                df.loc[
                    (df['log_name'] == summary_log_name) & (df['log_name'] == 'agg_weight_recv'),
                    'count:timestamp'
                ] = new_weight_recv_count
            else:
                print("  - agg_weight_recv is non-zero. No fix needed.")

        except (IndexError, KeyError):
            print("  - Warning: Could not find 'agg_train_sent' or 'agg_weight_recv' rows in summary. Skipping fix.")
        
        return df
    return process


# --- Handlers & Config ---


def handle_stat_utility(parser: 'LogParser', match: re.Match) -> Dict[str, Any]:
    trainer_id = match.group('trainer_id')
    trainer_state = parser.keyed_state[trainer_id]
    round_num = trainer_state.get('round')
    data_id = trainer_state.get('data_id')
    iteration_key = (trainer_id, round_num, data_id)
    parser.iteration_tracker[iteration_key] += 1
    iteration = parser.iteration_tracker[iteration_key]
    return {'round': round_num, 'data_id': data_id, 'iteration': iteration}


# Declaration
LOG_CONFIG = {}
EXPORT_CONFIG = {}


class LogParser:
    def __init__(self, patterns: List[Dict], row_processors: Optional[List[Callable]] = None, dataframe_processors: Optional[List[Callable]] = None, export_configs: Optional[Dict] = None):
        self.patterns = patterns
        self.row_processors = row_processors or []
        self.dataframe_processors = dataframe_processors or []
        self.records: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}
        self.keyed_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.iteration_tracker: Dict[Tuple, int] = defaultdict(int)
        self.export_configs = export_configs

    def parse_log_file(self, log_filepath: Path):
        self._reset_state()
        with open(log_filepath, 'r') as f:
            for line in f:
                for pattern in self.patterns:
                    match = pattern['regex'].search(line)
                    if not match:
                        continue
                    processed_data = {}
                    if 'group_to_columns' in pattern:
                        for group, (col, type_fn) in pattern['group_to_columns'].items():
                            try:
                                processed_data[col] = type_fn(
                                    match.group(group))
                            except (IndexError, TypeError, ValueError):
                                continue

                    if 'handler' in pattern:
                        handler_data = pattern['handler'](self, match)
                        processed_data.update(handler_data)

                    if pattern['type'] == 'STATE_UPDATE':
                        # (State update logic is unchanged)
                        pass

                    elif pattern['type'] == 'EXTRACT':
                        key = match.group(
                            pattern['extract_key_group']) if 'extract_key_group' in pattern else None

                        record = {
                            'log_name': pattern['name'],  # <-- ADDED LOG_NAME
                            # **self.global_state,
                            # **(self.keyed_state[key] if key else {}),
                            **processed_data
                        }
                        self.records.append(record)

                    break  # Ensures only the first matching regex is used

    def _apply_row_processors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a chain of row-wise processors in a single pass over the DataFrame.
        """
        if not self.row_processors or df.empty:
            return df

        # A single state dictionary is created and shared across all processors for this run.
        shared_state = {}

        def runner(row):
            for func in self.row_processors:
                row = func(row, shared_state)
            return row
        return df.apply(runner, axis=1)

    def _apply_dataframe_processors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies a chain of processors that operate on the entire DataFrame."""
        if not self.dataframe_processors or df.empty:
            return df
        # Processors are applied sequentially
        for func in self.dataframe_processors:
            df = func(df)
        return df

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)

        # Apply the new row-wise post-processing logic
        df = self._apply_row_processors(df)
        df = self._apply_dataframe_processors(df)
        return df

    def export_to_configured_csvs(self, output_dir: Path):
        """
        Generates multiple CSV files based on the declarative EXPORT_CONFIG.
        """
        main_df = self.to_dataframe()
        if main_df.empty or not self.export_configs:
            print("ℹ️ No data or export configurations to process.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n--- Exporting DataFrames based on Configuration ---")

        for name, config in self.export_configs.items():
            try:
                # Filter rows based on log_name
                df_filtered = main_df[main_df['log_name'].isin(
                    config['log_names'])].copy()
                if df_filtered.empty:
                    print(f"⚠️ No records found for '{name}'. Skipping.")
                    continue
                
                # Get list of columns that actually exist in the filtered df
                existing_cols = [col for col in config['columns'] if col in df_filtered.columns]
                
                # todo (Gaurav): understand why we even need this
                # Drop duplicates for summary CSVs
                if name == 'iteration_timing' or name == 'communication_summary':
                    df_filtered.drop_duplicates(subset=existing_cols, inplace=True)

                output_path = output_dir / config['output_filename']
                
                # Select only the requested (and existing) columns
                df_filtered[existing_cols].to_csv(output_path, index=False)
                print(
                    f"✅ Successfully wrote {len(df_filtered)} records for '{name}' to {os.path.abspath(output_path)}")

            except KeyError as e:
                print(f"❌ Error in export config '{name}': Missing key {e}")

    def _reset_state(self):
        self.records = []
        self.global_state = {}
        self.keyed_state = defaultdict(dict)
        self.iteration_tracker = defaultdict(int)

    def to_csv(self, output_filepath: Path):
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(output_filepath, index=False)
            print(
                f"✅ Successfully wrote {len(df)} records to {output_filepath}")

# --- Configuration ---


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
            'output_filename': '19Oct_slow_straggler_evaluation_metrics.csv',
            'log_names': ['eval_model'],
            'columns': ['timestamp', 'time_since_start', 'round_id', 'data_id', 'accuracy']
        },
        # 'trainer_performance': {
        #     'output_filename': 'trainer_performance.csv',
        #     'log_names': ['extract_stat_utility'],
        #     'columns': ['timestamp', 'trainer_id', 'trainer_num', 'loss', 'stat_utility']
        # },
    },
    'flame_fwdllm_trainer': {
        'train_times': {
            # 'output_filename': 'train_times_delay_slow_6hr.csv',
            'output_filename': 'train_times_noDelay_slow_4hr.csv',
            # 'output_filename': 'train_times_delayBy20_3hrs.csv',
            'log_names': ['recv_weights_time'],
            'columns': ['timestamp', 'round_id', 'data_id', 'iteration_id', 'train_time_sec',
                        # 'cumulative_train_time_sec', 'mean:cumulative_train_time_sec',
                        'cumulative_recv_weights_time', 'mean:cumulative_recv_weights_time', 
                        'time_since_start',
                        'trainer_num',
                        # 'trainer_id',
                        ]
        }
    },
    'flame_fwdllm_trainer_old': {
        'train_times': {
            'output_filename': 'old_train_times.csv',
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

if __name__ == '__main__':
    # global EXPORT_CONFIG
    output_dir = Path("output/")

    # log_file_type = 'flame_fwdllm_aggregator'
    # log_file = Path("../logs/agg_2000.log")
    # row_proc_steps = [
    #     create_sequential_id_processor(eval_log_name='eval_model', iter_log_name='var'),
    #     create_time_calculator_processor(start_log_name='first_distribute_weights'),
    #     # create_numeric_id_processor(source_col='trainer_id', dest_col='trainer_num'),
    #     # create_cumulative_sum_processor(group_key_col='trainer_id', target_cols=['train_time', 'stall_time'])
    #     # Example of a simple lambda processor
    #     # lambda row, state: row.assign(accuracy_plus_one=row['accuracy'] + 1 if pd.notna(row['accuracy']) else pd.NA)
    # ]

    ################# Aggregator

    ################# 1 hour runs
    # log_file_type = "flame_fwdllm_trainer"
    # log_file = Path(
    #     "../logs/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_19_10_01_39.log")
    # row_proc_steps = [
    #     create_time_calculator_processor(start_log_name='train_time'),
    #     create_numeric_id_processor(
    #         source_col='trainer_id', dest_col='trainer_num'),
    #     create_cumulative_sum_processor(
    #         group_key_col='trainer_id', target_cols=['train_time_sec'])
    # ]

    ################# 2.5-3 hour runs
    log_file_type = "flame_fwdllm_trainer"
    # log_file = Path(
    #     "../logs/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_19_10_18_03.log")
    log_file = Path(
        "../logs/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_19_10_20_02.log")
    # log_file = Path(
    #     "../logs/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_19_10_05_18.log")
    row_proc_steps = [
        create_time_calculator_processor(start_log_name='train_time'),
        create_numeric_id_processor(
            source_col='trainer_id', dest_col='trainer_num'),
        create_cumulative_sum_processor(
            group_key_col='trainer_id', target_cols=['recv_weights_time'])
    ]

    # log_file_type = "flame_fwdllm_trainer_old"
    # log_file = Path(
    #     "../logs/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_30_09_05_22.log")
    # row_proc_steps = [
        # create_time_calculator_processor(start_log_name='train_time'),
        # create_sequential_id_processor(
        #     eval_log_name='first_fetch_weights', iter_log_name='train_time'),
        # create_numeric_id_processor(
        #     source_col='iteration_id', dest_col='trainer_num'),             # hack
        # create_cumulative_sum_processor(
        #     group_key_col='trainer_num', target_cols=['train_time_sec'])
    # ]

    # Define DataFrame Processors (Whole-dataframe, group-wise operations)
    df_proc_steps = [
        create_broadcast_aggregator(
            group_by_cols=['round_id', 'data_id', 'iteration_id'],
            aggregations={
                'train_time_sec': ['mean', 'sum'],
                'cumulative_recv_weights_time': ['mean', 'sum'],
            }
        )
    ]

    # Define DataFrame Processors (Whole-dataframe, group-wise operations)
    df_proc_steps = [
        create_broadcast_aggregator(
            group_by_cols=['round_id', 'data_id', 'iteration_id'],
            aggregations={
                'train_time_sec': ['mean', 'sum'],
                'cumulative_recv_weights_time': ['mean', 'sum'],
            }
        )
    ]

    ################### Aggregator
    log_file_type = "flame_fwdllm_aggregator"
    log_file = Path(
        "/home/dgarg39/shreya/flame/lib/python/examples/fwdllm/expts/run_tc_expts/log/new/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_05_11_22_44.log")
    suffix = "async"
    EXPORT_CONFIG['flame_fwdllm_aggregator']['evaluation_metrics']['output_filename'] = f'reject_stale_k5_c15_n100-{suffix}.csv'

    row_proc_steps = [
        create_sequential_id_processor(eval_log_name='eval_model', iter_log_name='var'),
        create_time_calculator_processor(start_log_name='first_distribute_weights'),
    ]

    df_proc_steps = []
    
    ############## Async-Cifar-10
    # log_file_type = "Async-Cifar-10"

    # log_file_dir = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs"

    # oort_syn0_comm = f"{log_file_dir}/agg_sheph_14_05_12_52_alpha0.1_cifar_70acc_fedavg_oort_unaware_syn_50.log"
    # oort_oracular_syn0_comm = f"{log_file_dir}/agg_wash_11_05_02_42_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn0.log"
    # oort_async_syn0_comm = f"{log_file_dir}/agg_wash_15_05_12_46_alpha0.1_cifar_70acc_fedbuff_async_oort_unaware_syn_0.log"
    # oort_async_oracular_syn0_comm = f"{log_file_dir}/agg_sheph_11_05_02_42_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn0.log"
    # felix_syn0_comm = f"{log_file_dir}/agg_sheph_13_05_01_50_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_0.log"

    # oort_syn0_comm_replacement = f"{log_file_dir}/agg_sheph_15_05_12_46_alpha0.1_cifar_70acc_fedbuff_async_oort_unaware_syn_50.log"
    # suffix = "oort_async_oracular"
    # EXPORT_CONFIG['Async-Cifar-10']['communication_summary']['output_filename'] = f'communication_summary-{suffix}.csv'

    # log_file = Path(oort_async_oracular_syn0_comm)
    # row_proc_steps = []
    # df_proc_steps = [
    #     # Step 1: Generate the summary counts
    #     create_summarization_processor(
    #         group_by_col='log_name',
    #         aggregations={
    #             'timestamp': 'count',
    #             # 'timestamp': 'max',           # todo: Fix output for multiple aggregations
    #         },
    #     ),
    #     # Step 2: Apply the custom OORT logic
    #     apply_oort_comm_fix(
    #         group_by_col='log_name', # <-- Pass the col that was grouped on
    #         concurrency=13
    #     ),
    # ]
    
    parser = LogParser(
        patterns=LOG_CONFIG[log_file_type],
        row_processors=row_proc_steps,
        dataframe_processors=df_proc_steps,
        export_configs=EXPORT_CONFIG[log_file_type]
    )
    parser.parse_log_file(log_file)

    # Export the data to multiple CSV files as configured
    parser.export_to_configured_csvs(output_dir)

    # df_result = parser.to_dataframe().iloc[:5]

    # print("\n--- Parsed DataFrame with All Features ---")
    # print(df_result.to_string())

    # parser.to_csv(Path("agg_2000_parsed.csv"))
