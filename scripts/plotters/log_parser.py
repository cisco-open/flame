import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Callable, Optional, Tuple
from configs import LOG_CONFIG, EXPORT_CONFIG, CONSTANTS

# def configure():
#     output_dir = Path("output/")

#     log_file_type = "flame_fwdllm_aggregator"
#     # log_file = Path("/Users/gaurav/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/SysML_experiment_logs/logs/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_14_11_11_02.log")
#     # suffix = "reject_stale"
#     log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_07_01_21_52.log")
#     suffix = "baseline"
#     EXPORT_CONFIG['flame_fwdllm_aggregator']['evaluation_metrics']['default_output_filename'] = f'sync_k5_c5_n5-{suffix}.csv'

#     ## Post processors on the parsed data
#     row_proc_steps = [
#         create_sequential_id_processor(eval_log_name='eval_model', iter_log_name='var'),
#         create_time_calculator_processor(start_log_name='first_distribute_weights'),
#     ]

#     df_proc_steps = []

#     return log_file_type, row_proc_steps, df_proc_steps, log_file, output_dir


def configure():
    output_dir = Path("output/")

    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_07_01_21_52.log")
    # log_file_type = "flame_fwdllm_aggregator"
    # CONSTANTS['file_prefix'] = "sync_nck_5_baseline_agg"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_08_01_02_53.log")
    # log_file_type = "flame_fwdllm_aggregator"
    # CONSTANTS['file_prefix'] = "sync_n100_c30_k10_baseline_agg"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_07_01_21_52.log")
    # log_file_type = "flame_fwdllm_trainer"
    # CONSTANTS['file_prefix'] = "sync_nck_5_baseline_trainer"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_08_01_02_53.log")
    # log_file_type = "flame_fwdllm_trainer"
    # CONSTANTS['file_prefix'] = "sync_n100_c30_k10_baseline_trainer"


    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_27_01_19_57.log")
    # log_file_type = "flame_fwdllm_aggregator"
    # CONSTANTS['file_prefix'] = "sync_nck_5_eval256_agg"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_27_01_19_57.log")
    # log_file_type = "flame_fwdllm_trainer"
    # CONSTANTS['file_prefix'] = "sync_nck_5_eval256_trainer"


    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_28_01_01_16.log")
    # log_file_type = "flame_fwdllm_aggregator"
    # CONSTANTS['file_prefix'] = "sync_n100_c30_k10_eval32_agg"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_28_01_01_16.log")
    # log_file_type = "flame_fwdllm_trainer"
    # CONSTANTS['file_prefix'] = "sync_n100_c30_k10_eval32_trainer"
    log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_28_01_01_56.log")
    log_file_type = "flame_fwdllm_aggregator"
    CONSTANTS['file_prefix'] = "sync_n100_c30_k10_eval256_agg"
    # log_file = Path("/Users/gaurav/Projects/flame/scripts/logs/runtime_optimizations/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_28_01_01_56.log")
    # log_file_type = "flame_fwdllm_trainer"
    # CONSTANTS['file_prefix'] = "sync_n100_c30_k10_eval256_trainer"

    ## Post processors on the parsed data
    row_proc_steps = [
        create_sequential_id_processor(eval_log_name="eval_model", iter_log_name="var"),
        create_time_calculator_processor(start_log_name="first_distribute_weights"),
    ]

    df_proc_steps = []

    return log_file_type, row_proc_steps, df_proc_steps, log_file, output_dir


def create_numeric_id_processor(source_col: str, dest_col: str) -> Callable:
    """
    Factory for a row-processor that assigns a unique, sequential integer ID
    to unique values in a source column.
    """

    def process(row: pd.Series, state: dict) -> pd.Series:
        id_map = state.setdefault(f"numeric_id_map_{source_col}", {})
        next_id = state.setdefault(f"numeric_id_next_{source_col}", 0)

        source_value = row[source_col]
        if pd.notna(source_value):
            if source_value not in id_map:
                id_map[source_value] = next_id
                state[f"numeric_id_next_{source_col}"] += 1
            row[dest_col] = id_map[source_value]
        else:
            row[dest_col] = pd.NA
        return row

    return process


def populate_model_version() -> Callable:
    def process(row: pd.Series, state: dict) -> pd.Series:
        row["_model_version"] = row["round_id"] * row["data_id"]
        return row

    return process


def create_sequential_id_processor(eval_log_name: str, iter_log_name: str) -> Callable:
    """Factory to create a row-processor for sequential and iterative IDs."""

    def process(row: pd.Series, state: dict) -> pd.Series:
        seq_counter = state.setdefault("sequential_id_counter", 0)
        iter_counter = state.setdefault("iteration_id_counter", 0)

        # TODO: This is only for the current AGNews dataset. Need to parametrize this for other datasets.
        round_id = seq_counter // 150
        data_id = seq_counter % 150

        # Don't override if any of row['iteration_id'], row['data_id'] and row['round_id'] are present
        has_ids = any(pd.notna(row.get(k)) for k in ['iteration_id', 'data_id', 'round_id'])

        if row['log_name'] == eval_log_name:
            if not has_ids:
                row['round_id'], row['data_id'] = round_id, data_id
                row['iteration_id'] = iter_counter
            
            if pd.isna(row.get('_model_version')):
                row['_model_version'] = seq_counter

            state['sequential_id_counter'] += 1
            state['iteration_id_counter'] = 0
            state['current_round_id'], state['current_data_id'] = round_id, data_id
        elif row['log_name'] == iter_log_name:
            if not has_ids:
                row['round_id'], row['data_id'] = round_id, data_id
                row['iteration_id'] = iter_counter

            if pd.isna(row.get('_model_version')):
                row['_model_version'] = seq_counter

            state['iteration_id_counter'] += 1
        else:
            if not has_ids:
                row['round_id'], row['data_id'] = pd.NA, pd.NA 
                row['iteration_id'] = pd.NA
        return row

    return process

# def time_per_iteration_processor(start_log_name: str) -> Callable:
#     def process(row: pd.Series, state: dict) -> pd.Series:


def create_time_calculator_processor(start_log_name: str) -> Callable:
    """Factory to create a row-processor that calculates time since a start event."""

    def process(row: pd.Series, state: dict) -> pd.Series:
        if row["log_name"] == start_log_name and pd.notna(row["timestamp"]):
            state.setdefault("training_start_time", row["timestamp"])

        if "training_start_time" in state and pd.notna(row["timestamp"]):
            datetime_diff = row["timestamp"] - state["training_start_time"]
            row["time_since_start"] = datetime_diff.total_seconds()
        else:
            row["time_since_start"] = pd.NaT
        return row

    return process


def create_cumulative_sum_processor(
    group_key_col: str, target_cols: List[str]
) -> Callable:
    """
    Factory for a row-processor that calculates cumulative sums for target columns,
    grouped by a specific key (e.g., 'trainer_id').
    """

    def process(row: pd.Series, state: dict) -> pd.Series:
        sums_cache = state.setdefault(
            "cumulative_sums", defaultdict(lambda: defaultdict(float))
        )

        group_key = row[group_key_col]
        if pd.isna(group_key):
            return row

        for col in target_cols:
            if pd.notna(row[col]):
                sums_cache[group_key][col] += row[col]
            row[f"cumulative_{col}"] = sums_cache[group_key][col]

        return row

    return process


def create_broadcast_aggregator(
    group_by_cols: List[str], aggregations: Dict[str, List[str]]
) -> Callable:
    """
    Factory for a DataFrame-processor that performs groupby aggregations and
    broadcasts the results back to the original rows.
    """

    def process(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not all(col in df.columns for col in group_by_cols):
            return df

        print(f"\nApplying broadcast aggregation grouped by {group_by_cols}...")

        df_out = df.copy()
        grouped = df_out.groupby(group_by_cols)

        for col, agg_funcs in aggregations.items():
            if col not in df_out.columns:
                print(
                    f"  - Warning: Column '{col}' not found for aggregation. Skipping."
                )
                continue

            for agg_func in agg_funcs:
                if agg_func not in ["mean", "sum", "max", "min", "count"]:
                    print(
                        f"  - Warning: Aggregation '{agg_func}' not supported for broadcasting. Skipping."
                    )
                    continue

                new_col_name = f"{agg_func}:{col}"
                print(f"  - Calculating '{new_col_name}'...")
                df_out[new_col_name] = grouped[col].transform(agg_func)

        return df_out

    return process


def create_summarization_processor(
    group_by_col: str, aggregations: Dict[str, Any]
) -> Callable:
    """
    Factory for a DataFrame-processor that performs groupby 'log_name'
    and appends the results as new summary rows with new log_names.
    """

    def process(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or group_by_col not in df.columns:
            return df

        print(f"\nApplying summarization grouped by '{group_by_col}'...")

        summary_df = df.groupby(group_by_col).agg(aggregations)
        summary_df.columns = [
            f"{agg_func}:{col}"
            for col, agg_func_list in aggregations.items()
            for agg_func in (
                agg_func_list if isinstance(agg_func_list, list) else [agg_func_list]
            )
        ]
        summary_df.reset_index(inplace=True)
        summary_df["log_name"] = summary_df["log_name"].apply(lambda x: f"{x}")

        print(f"  - Generated {len(summary_df)} summary rows.")
        print(summary_df)

        return summary_df

    return process


def apply_oort_comm_fix(group_by_col: str, concurrency: int = 13) -> Callable:
    """
    Factory for a DataFrame-processor that applies the special OORT
    communication counting logic.
    """

    def process(df: pd.DataFrame) -> pd.DataFrame:
        print(f"\nApplying OORT communication fix...")

        summary_log_name = group_by_col
        summary_rows = df[df["log_name"] == summary_log_name].copy()
        if summary_rows.empty:
            print(
                f"  - Warning: No summary rows found ('{summary_log_name}'). Skipping fix."
            )
            return df

        try:
            train_sent_count = summary_rows.loc[
                summary_rows["log_name"] == "agg_train_sent", "count:timestamp"
            ].iloc[0]
            weight_recv_count = summary_rows.loc[
                summary_rows["log_name"] == "agg_weight_recv", "count:timestamp"
            ].iloc[0]

            if weight_recv_count == 0:
                print(f"  - Found agg_weight_recv=0. Applying fix...")
                new_weight_recv_count = train_sent_count - concurrency
                df.loc[
                    (df["log_name"] == summary_log_name)
                    & (df["log_name"] == "agg_weight_recv"),
                    "count:timestamp",
                ] = new_weight_recv_count
            else:
                print("  - agg_weight_recv is non-zero. No fix needed.")

        except (IndexError, KeyError):
            print(
                "  - Warning: Could not find 'agg_train_sent' or 'agg_weight_recv' rows in summary. Skipping fix."
            )

        return df

    return process


class LogParser:
    def __init__(
        self,
        patterns: List[Dict],
        row_processors: Optional[List[Callable]] = None,
        dataframe_processors: Optional[List[Callable]] = None,
        export_configs: Optional[Dict] = None,
    ):
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
        with open(log_filepath, "r") as f:
            for line in f:
                for pattern in self.patterns:
                    match = pattern["regex"].search(line)
                    if not match:
                        continue
                    processed_data = {}
                    if "group_to_columns" in pattern:
                        for group, (col, type_fn) in pattern[
                            "group_to_columns"
                        ].items():
                            try:
                                processed_data[col] = type_fn(match.group(group))
                            except (IndexError, TypeError, ValueError):
                                continue

                    if "handler" in pattern:
                        handler_data = pattern["handler"](self, match)
                        processed_data.update(handler_data)

                    if pattern["type"] == "STATE_UPDATE":
                        pass
                    elif pattern["type"] == "EXTRACT":
                        key = (
                            match.group(pattern["extract_key_group"])
                            if "extract_key_group" in pattern
                            else None
                        )

                        record = {"log_name": pattern["name"], **processed_data}
                        self.records.append(record)

                    break  # Ensures only the first matching regex is used

    def _apply_row_processors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a chain of row-wise processors in a single pass over the DataFrame.
        """
        if not self.row_processors or df.empty:
            return df

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
        for func in self.dataframe_processors:
            df = func(df)
        return df

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        df = self._apply_row_processors(df)
        df = self._apply_dataframe_processors(df)
        return df

    def export_to_configured_csvs(self, output_dir: Path):
        """
        Generates multiple CSV files based on the declarative EXPORT_CONFIG.
        """
        main_df = self.to_dataframe()
        if main_df.empty or not self.export_configs:
            print(f"ℹ️ No data or export configurations to process. Double check if the `log_file_type` config is correctly set e.g: {list[str](LOG_CONFIG.keys())[:2]}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n--- Exporting DataFrames based on Configuration ---")

        for name, config in self.export_configs.items():
            try:
                df_filtered = main_df[
                    main_df["log_name"].isin(config["log_names"])
                ].copy()
                if df_filtered.empty:
                    print(f"⚠️ No records found for '{name}'. Skipping.")
                    continue

                existing_cols = [
                    col for col in config["columns"] if col in df_filtered.columns
                ]

                if name == "iteration_timing" or name == "communication_summary":
                    df_filtered.drop_duplicates(subset=existing_cols, inplace=True)

                output_path = output_dir / config['default_output_filename']()
                df_filtered[existing_cols].to_csv(output_path, index=False)
                print(
                    f"✅ Successfully wrote {len(df_filtered)} records for '{name}' to {os.path.abspath(output_path)}"
                )

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
            print(f"✅ Successfully wrote {len(df)} records to {output_filepath}")


if __name__ == "__main__":
    log_file_type, row_proc_steps, df_proc_steps, log_file, output_dir = configure()

    parser = LogParser(
        patterns=LOG_CONFIG[log_file_type],
        row_processors=row_proc_steps,
        dataframe_processors=df_proc_steps,
        export_configs=EXPORT_CONFIG[log_file_type],
    )
    parser.parse_log_file(log_file)

    parser.export_to_configured_csvs(output_dir)
