import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback
from pathlib import Path

CONSTANTS = {
    'file_prefix': '',
}
TRAINER_PLOTTING_CONFIG = {
    'cdfs': [
        {
            'latency_type': 'train_latency',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-train_latency.csv',
            'latency_column': 'train_time_sec',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-train_latency_cdf.png',
            'x_label': 'Overall Train Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Train Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'send_latency',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-send_latency.csv',
            'latency_column': 'send_latency',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-send_latency_cdf.png',
            'x_label': 'Send Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Send Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'recv_latency',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-recv_latency.csv',
            'latency_column': 'recv_latency',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-recv_latency_cdf.png',
            'x_label': 'Receive Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Receive Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'pause_execution',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-pause_execution.csv',
            'latency_column': 'pause_latency',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-pause_execution_cdf.png',
            'x_label': 'Pause Execution Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Pause Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'make_model_functional_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-make_model_functional_latency.csv',
            'latency_column': 'make_model_functional_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-make_model_functional_latency_cdf.png',
            'x_label': 'Make Model Functional Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Make Model Functional Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'select_optimal_perturbations_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-select_optimal_perturbations_latency.csv',
            'latency_column': 'select_optimal_perturbations_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-select_optimal_perturbations_latency_cdf.png',
            'x_label': 'Select Optimal Perturbations Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Select Optimal Perturbations Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'compute_batch_stat_utility_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-compute_batch_stat_utility_latency.csv',
            'latency_column': 'compute_batch_stat_utility_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-compute_batch_stat_utility_latency_cdf.png',
            'x_label': 'Compute Batch Stat Utility Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Compute Batch Stat Utility Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'prepare_perturbation_tensors_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-prepare_perturbation_tensors_latency.csv',
            'latency_column': 'prepare_perturbation_tensors_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-prepare_perturbation_tensors_latency_cdf.png',
            'x_label': 'Prepare Perturbation Tensors Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Prepare Perturbation Tensors Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'compute_forward_jvp_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-compute_forward_jvp_latency.csv',
            'latency_column': 'compute_forward_jvp_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-compute_forward_jvp_latency_cdf.png',
            'x_label': 'Compute Forward JVP Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Compute Forward JVP Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'accumulate_and_extract_grads_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-accumulate_and_extract_grads_latency.csv',
            'latency_column': 'accumulate_and_extract_grads_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-accumulate_and_extract_grads_latency_cdf.png',
            'x_label': 'Accumulate and Extract Grads Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Accumulate and Extract Grads Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'force_cuda_memory_cleanup_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-force_cuda_memory_cleanup_latency.csv',
            'latency_column': 'force_cuda_memory_cleanup_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-force_cuda_memory_cleanup_latency_cdf.png',
            'x_label': 'Force CUDA Memory Cleanup Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Force CUDA Memory Cleanup Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'setup_training_state_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-setup_training_state_latency.csv',
            'latency_column': 'setup_training_state_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-setup_training_state_latency_cdf.png',
            'x_label': 'Setup Training State Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Setup Training State Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'train_one_batch_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-train_one_batch_latency.csv',
            'latency_column': 'train_one_batch_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-train_one_batch_latency_cdf.png',
            'x_label': 'Train One Batch Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Train One Batch Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'training_loop_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-training_loop_latency.csv',
            'latency_column': 'training_loop_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-training_loop_latency_cdf.png',
            'x_label': 'Training Loop Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Training Loop Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'finalize_training_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-finalize_training_latency.csv',
            'latency_column': 'finalize_training_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-finalize_training_latency_cdf.png',
            'x_label': 'Finalize Training Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Finalize Training Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'trainer_train_model_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-trainer_train_model_latency.csv',
            'latency_column': 'trainer_train_model_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-trainer_train_model_latency_cdf.png',
            'x_label': 'Trainer Train Model Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Trainer Train Model Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'trainer_eval_model_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-trainer_eval_model_latency.csv',
            'latency_column': 'trainer_eval_model_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-trainer_eval_model_latency_cdf.png',
            'x_label': 'Trainer Eval Model Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Trainer Eval Model Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'check_availability_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-check_availability_latency.csv',
            'latency_column': 'check_availability_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-check_availability_latency_cdf.png',
            'x_label': 'Check Availability Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Check Availability Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'perform_training_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-perform_training_latency.csv',
            'latency_column': 'perform_training_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-perform_training_latency_cdf.png',
            'x_label': 'Perform Training Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Perform Training Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'emulate_training_delay_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-emulate_training_delay_latency.csv',
            'latency_column': 'emulate_training_delay_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-emulate_training_delay_latency_cdf.png',
            'x_label': 'Emulate Training Delay Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Emulate Training Delay Latency ({CONSTANTS["file_prefix"]})'
        }
    ],
    'directory_name': 'plots/',
    'x_lim_left': 0,
    'y_lim_bottom': 0
}

AGGREGATOR_PLOTTING_CONFIG = {
    'cdfs': [
        {
            'latency_type': 'distribute_latency',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-distribute_latency.csv',
            'latency_column': 'distribute_latency',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-distribute_latency_cdf.png',
            'x_label': 'Distribute Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Distribute Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'aggregate_latency',
            'input_csv': lambda: f'output/{CONSTANTS['file_prefix']}-aggregate_latency.csv',
            'latency_column': 'aggregate_latency',
            'output_filename': lambda: f'{CONSTANTS['file_prefix']}-aggregate_latency_cdf.png',
            'x_label': 'Aggregate Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Aggregate Latency ({CONSTANTS['file_prefix']})'
        },
        {
            'latency_type': 'eval_model_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-eval_model_latency.csv',
            'latency_column': 'eval_model_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-eval_model_latency_cdf.png',
            'x_label': 'Eval Model Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Eval Model Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'sync_collect_and_accumulate_grads_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-sync_collect_and_accumulate_grads_latency.csv',
            'latency_column': 'sync_collect_and_accumulate_grads_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-sync_collect_and_accumulate_grads_latency_cdf.png',
            'x_label': 'sync_collect_and_accumulate_grads latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'sync_collect_and_accumulate_grads latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'aggregate_runtime_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-aggregate_runtime_latency.csv',
            'latency_column': 'aggregate_runtime',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-aggregate_runtime_latency_cdf.png',
            'x_label': 'Aggregate Runtime Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'Aggregate Runtime Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'aggregate_grads_sync_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-aggregate_grads_sync_latency.csv',
            'latency_column': 'aggregate_grads_sync_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-aggregate_grads_sync_latency_cdf.png',
            'x_label': '_aggregate_grads_sync Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_aggregate_grads_sync Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'process_single_trainer_message_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-process_single_trainer_message_latency.csv',
            'latency_column': 'process_single_trainer_message_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-process_single_trainer_message_latency_cdf.png',
            'x_label': '_process_single_trainer_message Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_process_single_trainer_message Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'process_aggregation_goal_met_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-process_aggregation_goal_met_latency.csv',
            'latency_column': 'process_aggregation_goal_met_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-process_aggregation_goal_met_latency_cdf.png',
            'x_label': '_process_aggregation_goal_met Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_process_aggregation_goal_met Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'agg_force_cuda_memory_cleanup_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-agg_force_cuda_memory_cleanup_latency.csv',
            'latency_column': 'agg_force_cuda_memory_cleanup_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-agg_force_cuda_memory_cleanup_latency_cdf.png',
            'x_label': 'agg_force_cuda_memory_cleanup Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'agg_force_cuda_memory_cleanup Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'prepare_distribution_payload_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-prepare_distribution_payload_latency.csv',
            'latency_column': 'prepare_distribution_payload_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-prepare_distribution_payload_latency_cdf.png',
            'x_label': '_prepare_distribution_payload Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_prepare_distribution_payload Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'distribute_weights_sync_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-distribute_weights_sync_latency.csv',
            'latency_column': 'distribute_weights_sync_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-distribute_weights_sync_latency_cdf.png',
            'x_label': '_distribute_weights_sync Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_distribute_weights_sync Latency ({CONSTANTS["file_prefix"]})'
        },
        {
            'latency_type': 'distribute_weights_async_latency',
            'input_csv': lambda: f'output/{CONSTANTS["file_prefix"]}-distribute_weights_async_latency.csv',
            'latency_column': 'distribute_weights_async_latency',
            'output_filename': lambda: f'{CONSTANTS["file_prefix"]}-distribute_weights_async_latency_cdf.png',
            'x_label': '_distribute_weights_async Latency (seconds)',
            'y_label': 'CDF',
            'title': lambda: f'_distribute_weights_async Latency ({CONSTANTS["file_prefix"]})'
        },
    ],
    'directory_name': 'plots/',
    'x_lim_left': 0,
    'y_lim_bottom': 0
}

def load_latency_values(csv_filepath, latency_column):
    """
    Load latency values from a CSV file.
    Returns a list of latency values (aggregating across all trainers).
    """
    try:
        df = pd.read_csv(csv_filepath)
        
        if latency_column not in df.columns:
            print(f"Error: Column '{latency_column}' not found in '{csv_filepath}'")
            return None
        
        # Extract latency values, dropping NaN values
        latency_values = df[latency_column].dropna().tolist()
        
        if not latency_values:
            print(f"Warning: No valid latency values found in '{csv_filepath}'")
            return None
        
        return latency_values
        
    except FileNotFoundError:
        print(f"Error: File not found '{csv_filepath}'")
        return None
    except Exception as e:
        print(f"Error loading '{csv_filepath}':\n{traceback.format_exc()}")
        return None

def generateCDF(
    list_vals,
    x_label,
    y_label,
    title,
    file_name,
    directory_name,
    x_lim_left=0,
    y_lim_bottom=0,
):
    """
    Generate a CDF plot from a list of values with sample count annotation.
    """
    if not list_vals:
        print(f"Error: No values provided for CDF generation")
        return False
    
    try:
        list_vals = np.array(list_vals)
        count1, bins_count1 = np.histogram(list_vals, bins=5000)
        pdf1 = count1 / sum(count1)
        cdf1 = np.cumsum(pdf1)

        p50_1 = round(np.quantile(list_vals, 0.50), 3)
        p90_1 = round(np.quantile(list_vals, 0.90), 3)
        p99_1 = round(np.quantile(list_vals, 0.99), 3)

        color = "tab:red"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel(y_label, color=color, fontweight="bold")
        ax.set_title(title, fontweight="bold")

        ax.plot(bins_count1[1:], cdf1, color=color, linewidth=2)

        percentiles = [
            (p50_1, 0.5, 'P50', '^'), 
            (p90_1, 0.9, 'P90', 'o'), 
            (p99_1, 0.99, 'P99', 'x')
        ]

        for x_val, y_val, label, marker in percentiles:
            bin_idx = np.searchsorted(bins_count1[1:], x_val, side='right')
            if bin_idx >= len(cdf1):
                bin_idx = len(cdf1) - 1
            
            actual_y = cdf1[min(bin_idx, len(cdf1) - 1)] if bin_idx >= 0 else y_val
            
            ax.plot(x_val, actual_y, marker=marker, markersize=10, color='black', 
                   markeredgewidth=2, label=f'{label}: {x_val:.3f}s')
            
            ax.annotate(
                "{:.3f}".format(x_val),
                (x_val, actual_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="black",
                size=11,
                fontweight="bold"
            )
        
        # Position legend slightly higher to leave room for the count text
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.12), fontsize=10)
        
        # Add Sample Count below the legend
        # (0.98, 0.02) places it in the bottom right corner of the plot area
        ax.text(0.98, 0.02, f'# of samples: {len(list_vals)}', 
                transform=ax.transAxes, 
                fontsize=11, 
                fontweight='bold', 
                ha='right', 
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        ax.set_yscale("linear")
        ax.set_xlim(left=x_lim_left)
        ax.set_ylim(bottom=y_lim_bottom)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(directory_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / file_name
        plt.savefig(output_path, facecolor="white")
        plt.close()
        
        print(f"Saved CDF plot to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error generating CDF: {traceback.format_exc()}")
        return False

def process_cdf_configs(config, config_name):
    """Process a set of CDF configurations."""
    print(f"\n{'='*60}")
    print(f"Processing {config_name} CDFs...")
    print(f"{'='*60}")
    
    for cdf_config in config['cdfs']:
        print(f"\nProcessing {cdf_config['latency_type']}...")
        print(f"  Reading from: {cdf_config['input_csv']()}")
        
        # Load latency values from CSV
        latency_values = load_latency_values(
            cdf_config['input_csv'](),
            cdf_config['latency_column']
        )
        
        if latency_values is None:
            print(f"  Skipping {cdf_config['latency_type']} due to error.")
            continue
        
        print(f"  Found {len(latency_values)} latency values")
        
        # Generate CDF plot
        success = generateCDF(
            latency_values,
            cdf_config['x_label'],
            cdf_config['y_label'],
            cdf_config['title'](),
            cdf_config['output_filename'](),
            config['directory_name'],
            config['x_lim_left'],
            config['y_lim_bottom']
        )
        
        if success:
            print(f"  ✓ Successfully generated CDF for {cdf_config['latency_type']}")
        else:
            print(f"  ✗ Failed to generate CDF for {cdf_config['latency_type']}")

if __name__ == "__main__":
    print("Starting CDF plot generation...")
    
    CONSTANTS['file_prefix'] = 'sync_n100_c30_k10_eval256_trainer'
    process_cdf_configs(TRAINER_PLOTTING_CONFIG, "Trainer")
    CONSTANTS['file_prefix'] = 'sync_n100_c30_k10_eval256_agg'
    process_cdf_configs(AGGREGATOR_PLOTTING_CONFIG, "Aggregator")
    
    print("\n" + "="*60)
    print("CDF plot generation complete.")
    print("="*60)