import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Global Constants for Dynamic Plot Configuration ---
# Default ratio to determine interpolation points if not provided.
# E.g., if there are 100 max unique data points, NUM_INTERPOLATION_POINTS = 100 * INTERP_RATIO (200)
INTERP_RATIO = 1 
DEFAULT_X_TICK_COUNT = 10
DEFAULT_Y_TICK_COUNT = 5
BATCHES_PER_EPOCH = 150
LINE_WIDTH = 1.5

MIN_MAX_DISABLED = True

# SYSTEM1_FILES = ["flame_run1.csv", "flame_run2.csv", "flame_run3.csv"]
# SYSTEM1_NAME = "Flame"
# SYSTEM2_FILES = ["fwdllm_run1.csv", "fwdllm_run2.csv", "fwdllm_run3.csv"]
# SYSTEM2_NAMES = "FwdLLM"

SYSTEM1_FILES = ["/home/dgarg39/shreya/flame/scripts/plotters/output/reject_stale_k5_c15_n100-async.csv"]
SYSTEM1_NAME = "AsyncFL reject stale updates (no stragglers)"
SYSTEM2_FILES = ["/home/dgarg39/shreya/flame/scripts/plotters/output/reject_stale_k5_c15_n100-async.csv"]
SYSTEM2_NAMES = "AsyncFL aggregate stale updates (no stragglers)"
# -----------------------------------------------------

def round_nice_ticks(data_min, data_max, num_ticks_target):
    """
    Calculates tick positions that are 'nice' round numbers (steps of 1, 2, or 5 
    times a power of 10) based on the data range and a target number of ticks.
    This fixed logic reliably prevents non-clean steps like 333 or 999.
    """
    data_range = data_max - data_min
    if data_range <= 1e-9:  # Handle near-zero range
        return np.array([data_min])
    
    # 1. Determine target step size (used only to estimate the magnitude)
    target_step = data_range / max(1, num_ticks_target - 1)
    
    # 2. Find the 'nicest' step size (1, 2, or 5 times 10^E)
    
    # Calculate the exponent (magnitude)
    exponent = np.floor(np.log10(target_step))
    
    # Calculate the base unit (e.g., if exponent=2, step_unit=100)
    step_unit = 10**exponent
    
    # Test multipliers (1, 2, 5, 10) to find the smallest step that is large enough
    # and results in a clean step (e.g., 100, 200, 500, 1000)
    nice_multipliers = [1, 2, 5, 10]
    best_nice_step = step_unit * 10 
    
    # Iterate to find the smallest nice step >= target_step
    for mult in nice_multipliers:
        current_step = mult * step_unit
        # Use a small tolerance for comparison to handle floating point issues
        if current_step >= target_step * 0.999: 
            best_nice_step = current_step
            break
            
    nice_step = best_nice_step
    
    # 3. Determine the start and end of the ticks based on the nice step
    # This ensures ticks start and end on clean multiples of the nice_step.
    tick_start = np.floor(data_min / nice_step) * nice_step
    tick_end = np.ceil(data_max / nice_step) * nice_step
    
    # 4. Generate the actual ticks
    # Use a small tolerance (+ nice_step / 2) to ensure the floating point array generation
    # includes the 'tick_end' value if it's supposed to be included.
    ticks = np.arange(tick_start, tick_end + nice_step / 2, nice_step)
    
    # Apply rounding to ensure ticks are clean integers/decimals (removes 199.99999999999997)
    # The degree of rounding depends on the step magnitude.
    if nice_step >= 1:
        # Round to the nearest integer for large steps
        ticks = np.round(ticks).astype(int)
    else:
        # Round based on the magnitude of the step for small steps (e.g., 0.1, 0.2)
        precision = int(-exponent) + 2
        ticks = np.round(ticks, precision)
    
    # Filter out negative ticks if min is near zero (relevant for X-axis)
    if data_min >= 0 and ticks[0] < 0:
        ticks = ticks[ticks >= 0]
        
    # The tick count may vary slightly from num_ticks_target, but the ticks will be clean.
    return ticks

def parse_time_string(time_str):
    """Converts a time string (HH:MM:SS or MM:SS) to total seconds."""
    parts = time_str.split(':')
    # Handle both HH:MM:SS (3 parts) and M:SS/MM:SS (2 parts)
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, s = map(int, parts)
    else:
        raise ValueError(f"Time format not recognized: {time_str}")
    return h * 3600 + m * 60 + s

def load_and_preprocess_data(file_list):
    """
    Loads all files, handling two CSV formats by using headers for data access.
    1. New Format: Uses the header from the CSV file.
    2. Old Format: Programmatically assigns headers before processing.
    
    It standardizes the data into a DataFrame with 'Accuracy', 'Time_Since_Start',
    and 'Unique_Mini_Batch_ID' columns.
    """
    all_runs_dfs = []
    for file_path in file_list:
        try:
            # --- Format Detection: Peek at the header line ---
            with open(file_path, 'r') as f:
                header = f.readline().strip().lower()
            
            # Check for column names from the new format
            is_new_format = 'time_since_start' in header and 'round_id' in header

            if is_new_format:
                print(f"Detected new format (with header) for: {file_path}")
                df = pd.read_csv(file_path)
                # Standardize column names to lowercase for consistency
                df.columns = map(str.lower, df.columns) 

                # 1. Process Accuracy (already a float, needs scaling)
                df['Accuracy'] = df['accuracy'] / 100.0

                # 2. Use Time_Since_Start directly (already in seconds)
                df['Time_Since_Start'] = df['time_since_start']

                # 3. Create Unique_Mini_Batch_ID from round_id and data_id
                df['Unique_Mini_Batch_ID'] = df['round_id'] * BATCHES_PER_EPOCH + df['data_id']
            
            else:
                print(f"Detected old format (no header) for: {file_path}")
                df = pd.read_csv(file_path, header=None, skipinitialspace=True)
                # Programmatically assign headers for old format
                # Assumes structure: Epoch, Mini_Batch_ID, Accuracy%, Time_String, ...
                base_headers = ['epoch_id', 'mini_batch_id', 'accuracy_str', 'time_str']
                # Create remaining column names to avoid errors
                extra_headers = [f'col_{i}' for i in range(len(base_headers), df.shape[1])]
                df.columns = base_headers + extra_headers

                # 1. Convert Accuracy from string 'XX%' to float using its new name
                df['Accuracy'] = df['accuracy_str'].astype(str).str.rstrip('%').astype(float) / 100

                # 2. Convert Time string to Total Seconds using its new name
                df['Time_Since_Start'] = df['time_str'].astype(str).apply(parse_time_string)
                
                # 3. Create Unique_Mini_Batch_ID from named columns
                df['Unique_Mini_Batch_ID'] = df['epoch_id'].astype(int) * BATCHES_PER_EPOCH + df['mini_batch_id'].astype(int)

            all_runs_dfs.append(df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    return all_runs_dfs

def plot_comparison_chart(
    system1_files,
    system1_label,
    system2_files,
    system2_label,
    plot_type='time', # 'time', 'batch', or 'time_vs_batch'
    x_tick_count=None,
    y_tick_count=None,
    explicit_interpolation_points=None
):
    """
    Generates a comparative plot for two systems, handling interpolation 
    and dynamic tick generation.
    """
    # Set plot colors explicitly
    color1 = 'C0' # Explicit Blue for System 1 (Flame)
    color2 = 'red' # Explicit Red for System 2 (FwdLLM)
    
    # 1. Load and Preprocess Data
    system1_runs = load_and_preprocess_data(system1_files)
    system2_runs = load_and_preprocess_data(system2_files)

    if not system1_runs or not system2_runs:
        print("Not enough valid run data to compare.")
        return
    print('Data loaded successfully')

    # 2. Determine Axis Keys, Labels, and Interpolation Settings
    if plot_type == 'time':
        x_data_key, y_data_key = 'Time_Since_Start', 'Accuracy'
        x_label, y_label = 'Time Since Start (Minutes)', 'Test Accuracy'
        interpolate = True
    elif plot_type == 'batch':
        x_data_key, y_data_key = 'Unique_Mini_Batch_ID', 'Accuracy'
        x_label, y_label = 'Mini-Batch ID/ Model Version', 'Test Accuracy'
        interpolate = False
    elif plot_type == 'time_vs_batch':
        x_data_key, y_data_key = 'Time_Since_Start', 'Unique_Mini_Batch_ID'
        x_label, y_label = 'Time Since Start (Minutes)', 'Mini-Batch ID/ Model Version'
        interpolate = True
    else:
        raise ValueError("plot_type must be 'time', 'batch', or 'time_vs_batch'.")

    # Combine all x-axis data to determine the plot range
    all_x_data = [t for df in system1_runs + system2_runs for t in df[x_data_key].tolist()]
    max_x = np.max(all_x_data) if all_x_data else 0
    
    # Determine interpolation points (if required)
    if interpolate:
        # Use provided value or calculate dynamically based on max data points
        max_unique_points = len(np.unique(all_x_data))
        default_interp = max_unique_points * INTERP_RATIO
        num_interpolation_points = explicit_interpolation_points if explicit_interpolation_points is not None else default_interp
        
        common_x_grid = np.linspace(0, max_x, num_interpolation_points)
    else:
        # For 'batch' plot, use the union of all unique batch IDs
        all_batches = sorted(list(set(all_x_data)))
        common_x_grid = np.array(all_batches)

    # 3. Process runs to get Y-values corresponding to the common X-grid
    def process_runs(runs_list):
        processed_y_values = []
        for df in runs_list:
            if df[x_data_key].empty:
                continue

            if interpolate:
                # Interpolate Y-values onto the dense time grid
                # Sort by the x-axis key for correct interpolation
                sorted_df = df.sort_values(by=x_data_key).reset_index(drop=True)
                y_processed = np.interp(
                    common_x_grid,
                    sorted_df[x_data_key].values,
                    sorted_df[y_data_key].values
                )
            else:
                # Align data to the common batch ID set (using previous value if missing)
                df_temp = df.set_index(x_data_key)[y_data_key].reindex(common_x_grid).ffill()
                y_processed = df_temp.values
            
            processed_y_values.append(y_processed)
        return np.array(processed_y_values)

    # 4. Calculate Mean and Min/Max for the Y-axis data
    s1_processed_y = process_runs(system1_runs)
    s1_mean = np.mean(s1_processed_y, axis=0)
    s1_upper_bound = np.max(s1_processed_y, axis=0)
    s1_lower_bound = np.min(s1_processed_y, axis=0)

    # System 2
    s2_processed_y = process_runs(system2_runs)
    s2_mean = np.mean(s2_processed_y, axis=0)
    s2_upper_bound = np.max(s2_processed_y, axis=0)
    s2_lower_bound = np.min(s2_processed_y, axis=0)

    # 5. Plotting (Using default style for better custom axis control)
    plt.style.use('default')
    fig, axes = plt.subplots(figsize=(10, 6))

    
    # System 1: Line (Mean) and Fill (Min/Max)
    axes.plot(common_x_grid, s1_mean, label=system1_label, color=color1, linewidth=LINE_WIDTH)
    if not MIN_MAX_DISABLED:
        axes.fill_between(
            common_x_grid, s1_lower_bound, s1_upper_bound,
            color=color1, alpha=0.25,
            label=f'Min/Max of {system1_label}'
            # label=f'$\\pm 1$ StdDev ({system1_label})'
        )

    # System 2: Line (Mean) and Fill (Min/Max)
    axes.plot(common_x_grid, s2_mean, label=system2_label, color=color2, linewidth=LINE_WIDTH)
    if not MIN_MAX_DISABLED:
        axes.fill_between(
            common_x_grid, s2_lower_bound, s2_upper_bound,
            color=color2, alpha=0.25,
            label=f'Min/Max of {system2_label}'
        )

    # 6. Dynamic and Human-Readable Ticks
    
    # Determine tick counts (use provided or default)
    x_tick_count = x_tick_count if x_tick_count is not None else DEFAULT_X_TICK_COUNT
    y_tick_count = y_tick_count if y_tick_count is not None else DEFAULT_Y_TICK_COUNT
    
    # Y-axis range (find global min/max across all runs/systems)
    global_min_y = min(np.min(s1_lower_bound), np.min(s2_lower_bound))
    global_max_y = max(np.max(s1_upper_bound), np.max(s2_upper_bound))

    # Add a small buffer (5% padding) to the Y range
    y_range = global_max_y - global_min_y
    buffer = y_range * 0.05 if y_range > 0 else 1
    min_y = max(0, global_min_y - buffer)
    max_y = global_max_y + buffer
    
    # Calculate round number ticks using the fixed nice tick logic
    x_ticks = round_nice_ticks(0, max_x, x_tick_count)
    y_ticks = round_nice_ticks(min_y, max_y, y_tick_count)

    # Set axis limits and ticks
    axes.set_xlim(x_ticks[0], x_ticks[-1])
    axes.set_ylim(min_y, max_y)
    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks)

    # Format tick labels
    if x_data_key == 'Time_Since_Start':
        # Time axis label format (seconds -> minutes)
        x_ticks_min = [f'{int(t/60)}m' for t in x_ticks]
        axes.set_xticklabels(x_ticks_min)
    else:
        # Mini-batch ID axis label (integer format)
        axes.set_xticklabels([f'{int(t)}' for t in x_ticks])
        
    if y_data_key == 'Accuracy':
        # Accuracy axis label format (float -> %)
        y_ticks_formatted = [f'${y*100:.0f}\\%$' for y in y_ticks] 
    else:
        # Other y-axis labels (integer format)
        y_ticks_formatted = [f'{int(t)}' for t in y_ticks]
    axes.set_yticklabels(y_ticks_formatted)
    
    # 7. Apply Solid Black Axis Lines and styling
    for spine in ['bottom', 'left']:
        axes.spines[spine].set_color('black')
        axes.spines[spine].set_linewidth(1.5)
    for spine in ['top', 'right']:
        axes.spines[spine].set_visible(False)

    # Configure major ticks (no negative X ticks because xlim starts at 0)
    axes.tick_params(axis='both', which='major', length=6, width=1.5, color='black')
    axes.grid(True, linestyle='--', alpha=0.6, color='lightgray')

    axes.set_title(f'Performance Comparison: {system1_label} vs {system2_label}')
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend(loc='upper right')
    plt.tight_layout()

    file_name = f'plots/{plot_type}_comparison.png'
    plt.savefig(file_name)
    # plt.show() # Disabled for production environment

    print(f'Plot saved to {file_name}')

if __name__ == "__main__":
    
    # Example usage for the original 'time' plot:
    plot_comparison_chart(
        system1_files=SYSTEM1_FILES,
        system1_label=SYSTEM1_NAME,
        system2_files=SYSTEM2_FILES,
        system2_label=SYSTEM2_NAMES,
        plot_type='time',
        x_tick_count=10,
        y_tick_count=5
    )

    # Example usage for the new 'batch' plot (no interpolation):
    plot_comparison_chart(
        system1_files=SYSTEM1_FILES,
        system1_label=SYSTEM1_NAME,
        system2_files=SYSTEM2_FILES,
        system2_label=SYSTEM2_NAMES,
        plot_type='batch',
        x_tick_count=10
    )

    # Example usage for the new 'time_vs_batch' plot:
    plot_comparison_chart(
        system1_files=SYSTEM1_FILES,
        system1_label=SYSTEM1_NAME,
        system2_files=SYSTEM2_FILES,
        system2_label=SYSTEM2_NAMES,
        plot_type='time_vs_batch',
        x_tick_count=10,
        y_tick_count=5
    )