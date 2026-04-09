import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

# --- Global Constants for Dynamic Plot Configuration ---
# Default ratio to determine interpolation points if not provided.
# E.g., if there are 100 max unique data points, NUM_INTERPOLATION_POINTS = 100 * INTERP_RATIO (200)
INTERP_RATIO = 0.5
DEFAULT_X_TICK_COUNT = 10
DEFAULT_Y_TICK_COUNT = 3
BATCHES_PER_EPOCH = 150
LINE_WIDTH = 2

MIN_MAX_DISABLED = True
STOP_LINE_AT_MISSING_DATA = (
    True  # If True, lines stop at missing data instead of forward-filling
)
X_AXIS_END_AT_SHORTEST = False  # If True, x-axis ends at shortest system's max x-value; if False, extends to longest system's max x-value
X_AXIS_MAX = 15  # Configurable maximum for X-axis scaling (e.g., 15 for 15 hours). Set to None for auto.
Y_AXIS_MIN = 0.20  # Configurable minimum for Y-axis scaling (e.g., 0.20 for 20%). Set to None for auto.
Y_AXIS_MAX = 0.85  # Configurable maximum for Y-axis scaling (e.g., 0.85 for 85%). Set to None for auto.

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.figsize": [6, 3],  # Adjusted to match the standard paper aspect ratio
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.color": "gainsboro",
        "axes.grid": True,
        "axes.axisbelow": True,
        # "axes.labelweight": "bold",  # Bold axis labels like your reference
    }
)
plt.margins(0.5, 0.5)
plt.tight_layout(pad=0)

SYSTEM_COLORS = ["royalblue", "red", "green", "purple", "goldenrod", "maroon"]
base_folder = "/Users/gaurav/Library/CloudStorage/GoogleDrive-curiouscreature97@gmail.com/My Drive/fwdllm_experiments"


# --- System-Agnostic Configuration ---

# Define all systems and their corresponding run files in this dictionary.
# Add as many systems as you need.
SYSTEMS_DATA = {
    # 100-avail
    # "Sync": [
    #     f"{base_folder}/output/nsdi/nsdi_sync_n_100_c_13_k_10_alpha_1_a_100_e_0_u_0-evaluation_metrics.csv"
    # ],
    # "Random": [
    #     f"{base_folder}/output/nsdi/nsdi_async_n_100_c_30_k_10_alpha_1_a_100_e_0_u_0-evaluation_metrics.csv"
    # ],
    # "OORT": [
    #     f"{base_folder}/output/nsdi/nsdi_felix_n_100_c_30_k_10_alpha_1_a_100_e_0_u_0-evaluation_metrics.csv"
    # ],
    # 90-avail-10-eval-0-unavail
    "Sync": [
        f"{base_folder}/output/nsdi_sync_n_100_c_13_k_10_alpha_1_a_90_e_10_u_0-evaluation_metrics.csv"
    ],
    "Random": [
        f"{base_folder}/output/async_n100_c30_k10_alpha1_avail90_agg-evaluation_metrics.csv"
    ],
    "OORT": [
        f"{base_folder}/output/async_n100_c30_k10_opts_alpha1_oort_durations_fixed_agg-evaluation_metrics.csv"
    ],
    "OORT++": [
        f"{base_folder}/output/async_n100_c30_k10_opts_alpha1_oort++_agg-evaluation_metrics.csv"
    ],
    # "OORT Greedy": [
    #     f"{base_folder}/output/nsdi_felix_deter_n_100_c_30_k_10_alpha_1_a_90_e_10_u_0-evaluation_metrics.csv"
    # ],
}

# Define colors for the system lines.
# Colors will be assigned in the order systems are defined in SYSTEMS_DATA.
# If there are more systems than colors, the list will wrap around.

# -----------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    parts = time_str.split(":")
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
            with open(file_path, "r") as f:
                header = f.readline().strip().lower()

            # Check for column names from the new format
            is_new_format = "time_since_start" in header and "round_id" in header

            if is_new_format:
                print(f"Detected new format (with header) for: {file_path}")
                df = pd.read_csv(file_path)
                # Standardize column names to lowercase for consistency
                df.columns = map(str.lower, df.columns)

                # 1. Process Accuracy (already a float, needs scaling)
                df["Accuracy"] = df["accuracy"] / 100.0

                # 2. Use Time_Since_Start directly (Convert seconds to hours)
                df["Time_Since_Start"] = df["time_since_start"] / 3600.0

                # 3. Create Unique_Mini_Batch_ID from round_id and data_id
                df["Unique_Mini_Batch_ID"] = (
                    df["round_id"] * BATCHES_PER_EPOCH + df["data_id"]
                )

            else:
                print(f"Detected old format (no header) for: {file_path}")
                df = pd.read_csv(file_path, header=None, skipinitialspace=True)
                # Programmatically assign headers for old format
                # Assumes structure: Epoch, Mini_Batch_ID, Accuracy%, Time_String, ...
                base_headers = ["epoch_id", "mini_batch_id", "accuracy_str", "time_str"]
                # Create remaining column names to avoid errors
                extra_headers = [
                    f"col_{i}" for i in range(len(base_headers), df.shape[1])
                ]
                df.columns = base_headers + extra_headers

                # 1. Convert Accuracy from string 'XX%' to float using its new name
                df["Accuracy"] = (
                    df["accuracy_str"].astype(str).str.rstrip("%").astype(float) / 100
                )

                # 2. Convert Time string to total hours using its new name
                df["Time_Since_Start"] = (
                    df["time_str"].astype(str).apply(parse_time_string)
                ) / 3600.0

                # 3. Create Unique_Mini_Batch_ID from named columns
                df["Unique_Mini_Batch_ID"] = df["epoch_id"].astype(
                    int
                ) * BATCHES_PER_EPOCH + df["mini_batch_id"].astype(int)

            all_runs_dfs.append(df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    return all_runs_dfs


def plot_comparison_chart(
    systems_data,
    plot_type="time",  # 'time', 'batch', or 'time_vs_batch'
    x_tick_count=None,
    y_tick_count=None,
    explicit_interpolation_points=None,
    smoothing_window=15,
    legend_loc="lower right",
    legend_bbox_to_anchor=None,
    legend_ncol=1,
):
    """
    Generates a comparative plot for N systems, handling interpolation
    and dynamic tick generation.
    """

    # 1. Load and Preprocess Data for all systems
    all_systems_runs = {}
    all_runs_list = []

    for system_name, file_list in systems_data.items():
        runs = load_and_preprocess_data(file_list)
        if not runs:
            print(
                f"Warning: No valid data loaded for system '{system_name}'. Skipping."
            )
            continue
        all_systems_runs[system_name] = runs
        all_runs_list.extend(runs)

    if not all_systems_runs:
        print("No valid run data found for any system. Aborting plot.")
        return
    print("Data loaded successfully")

    # 2. Determine Axis Keys, Labels, and Interpolation Settings
    if plot_type == "time":
        x_data_key, y_data_key = "Time_Since_Start", "Accuracy"
        x_label, y_label = "Train Time (Hours)", "Test Accuracy (%)"
        interpolate = True
    elif plot_type == "batch":
        x_data_key, y_data_key = "Unique_Mini_Batch_ID", "Accuracy"
        x_label, y_label = "Model Version (Mini-Batch ID)", "Test Accuracy (%)"
        interpolate = False
    elif plot_type == "time_vs_batch":
        x_data_key, y_data_key = "Time_Since_Start", "Unique_Mini_Batch_ID"
        x_label, y_label = "Train Time (Hours)", "Model Version (Mini-Batch ID)"
        interpolate = True
    else:
        raise ValueError("plot_type must be 'time', 'batch', or 'time_vs_batch'.")

    # Determine the plot range based on per-system maximums
    system_max_xs = []
    for system_name, runs in all_systems_runs.items():
        max_x_for_system = max(
            [df[x_data_key].max() for df in runs if not df[x_data_key].empty], default=0
        )
        system_max_xs.append(max_x_for_system)

    if not system_max_xs:
        print("No x-data found.")
        return

    min_x = min(system_max_xs)
    max_x = max(system_max_xs)

    # Use min or max of per-system maximums based on X_AXIS_END_AT_SHORTEST
    if X_AXIS_END_AT_SHORTEST:
        if max_x > (min_x * 1.05):
            max_x = min_x * 1.05  # Add 5% extra to show that the run ended early
        else:
            max_x = max_x

    if X_AXIS_MAX is not None:
        max_x = X_AXIS_MAX

    # Also keep all_x_data for other calculations (e.g., interpolation points)
    all_x_data = [t for df in all_runs_list for t in df[x_data_key].tolist()]

    # Determine interpolation points (if required)
    if interpolate:
        # Use provided value or calculate dynamically based on max data points
        max_unique_points = len(np.unique(all_x_data))
        default_interp = max_unique_points * INTERP_RATIO
        num_interpolation_points = int(
            explicit_interpolation_points
            if explicit_interpolation_points is not None
            else default_interp
        )
        print(f"Calculating {num_interpolation_points} interpolated points on X-axis")

        common_x_grid = np.linspace(0, max_x, num_interpolation_points)
    else:
        # For 'batch' plot, use the union of all unique batch IDs
        all_batches = sorted(list(set(all_x_data)))
        # Filter to only include batch IDs up to max_x (which respects X_AXIS_END_AT_SHORTEST)
        all_batches = [b for b in all_batches if b <= max_x]
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
                # Only interpolate within the range of observed x. Beyond that, forward fill
                x_obs = sorted_df[x_data_key].values
                y_obs = sorted_df[y_data_key].values
                max_x_obs = np.max(x_obs)  # Maximum x-value for this specific run

                # Determine right boundary behavior based on STOP_LINE_AT_MISSING_DATA
                if STOP_LINE_AT_MISSING_DATA:
                    right_val = np.nan  # Stop the line at missing data
                else:
                    right_val = y_obs[-1]  # Forward fill with last value

                y_interp = np.interp(
                    common_x_grid, x_obs, y_obs, left=0, right=right_val
                )

                # When STOP_LINE_AT_MISSING_DATA is True, set values to NaN for points beyond this run's max x-value
                if STOP_LINE_AT_MISSING_DATA:
                    y_interp[common_x_grid > max_x_obs] = np.nan

                y_processed = y_interp
            else:
                # Align data to the common batch ID set
                df_temp = df.set_index(x_data_key)[y_data_key].reindex(common_x_grid)
                if not STOP_LINE_AT_MISSING_DATA:
                    # Forward fill missing values to extend the line
                    df_temp = df_temp.ffill()
                y_processed = df_temp.values

            processed_y_values.append(y_processed)
        return np.array(processed_y_values)

    # 4. Calculate Mean and Min/Max for the Y-axis data for all systems
    processed_data = {}

    for system_name, runs in all_systems_runs.items():
        processed_y = process_runs(runs)

        if processed_y.size == 0:
            print(f"Warning: No processed Y-data for system '{system_name}'.")
            processed_data[system_name] = {
                "mean": np.array([]),
                "lower": np.array([]),
                "upper": np.array([]),
            }
            continue

        if STOP_LINE_AT_MISSING_DATA:
            mean = np.nanmean(processed_y, axis=0)
            upper = np.nanmax(processed_y, axis=0)
            lower = np.nanmin(processed_y, axis=0)
        else:
            mean = np.mean(processed_y, axis=0)
            upper = np.max(processed_y, axis=0)
            lower = np.min(processed_y, axis=0)

        processed_data[system_name] = {"mean": mean, "lower": lower, "upper": upper}

    # 5. Determine Ranges and Ticks Before Plotting
    # Determine tick counts (use provided or default)
    x_tick_count = x_tick_count if x_tick_count is not None else DEFAULT_X_TICK_COUNT
    y_tick_count = y_tick_count if y_tick_count is not None else DEFAULT_Y_TICK_COUNT

    # Y-axis range (find global min/max across all systems)
    all_lower_bounds = [
        data["lower"] for data in processed_data.values() if data["lower"].size > 0
    ]
    all_upper_bounds = [
        data["upper"] for data in processed_data.values() if data["upper"].size > 0
    ]

    if not all_lower_bounds or not all_upper_bounds:
        print("Warning: No Y-data to determine range. Using default [0, 1].")
        global_min_y = 0
        global_max_y = 1
    else:
        if STOP_LINE_AT_MISSING_DATA:
            global_min_y = np.nanmin([np.nanmin(b) for b in all_lower_bounds])
            global_max_y = np.nanmax([np.nanmax(b) for b in all_upper_bounds])
        else:
            global_min_y = np.min([np.min(b) for b in all_lower_bounds])
            global_max_y = np.max([np.max(b) for b in all_upper_bounds])

    # Add a small buffer (5% padding) to the Y range
    # Handle NaN values (can occur when STOP_LINE_AT_MISSING_DATA is True and all data is missing)
    if np.isnan(global_min_y) or np.isnan(global_max_y):
        print("Warning: All data points are missing. Using default Y-axis range.")
        global_min_y = 0
        global_max_y = 1

    y_range = global_max_y - global_min_y
    buffer = y_range * 0.05 if y_range > 0 else 1
    min_y = max(0, global_min_y - buffer) if Y_AXIS_MIN is None else Y_AXIS_MIN
    max_y = global_max_y + buffer if Y_AXIS_MAX is None else Y_AXIS_MAX

    # Calculate round number ticks using the fixed nice tick logic
    x_ticks = round_nice_ticks(0, max_x, x_tick_count)
    y_ticks = round_nice_ticks(min_y, max_y, y_tick_count)

    if X_AXIS_MAX is not None:
        x_ticks = x_ticks[x_ticks <= max_x]

    # Filter ticks to strictly adhere to the defined limits
    y_ticks = y_ticks[(y_ticks >= min_y) & (y_ticks <= max_y)]

    # 6. Plotting
    fig, axes = plt.subplots()

    # Enforce a 4-sided black box around the plot area
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    axes.margins(x=0.02, y=0.02)
    axes.tick_params(
        which="both", direction="out", length=4, width=1.2, color="black", labelsize=18
    )

    # Assign colors
    system_names = list(all_systems_runs.keys())
    colors = {}
    for i, name in enumerate(system_names):
        colors[name] = SYSTEM_COLORS[i % len(SYSTEM_COLORS)]

    # Loop through and plot each system
    for system_name, data in processed_data.items():
        if data["mean"].size == 0:
            continue  # Skip systems with no data

        color = colors[system_name]

        # Apply smoothing
        if smoothing_window > 1 and len(data["mean"]) > smoothing_window:
            smoothed_mean = (
                pd.Series(data["mean"])
                .rolling(window=smoothing_window, min_periods=1, center=True)
                .mean()
                .values
            )
            smoothed_lower = (
                pd.Series(data["lower"])
                .rolling(window=smoothing_window, min_periods=1, center=True)
                .mean()
                .values
            )
            smoothed_upper = (
                pd.Series(data["upper"])
                .rolling(window=smoothing_window, min_periods=1, center=True)
                .mean()
                .values
            )
        else:
            smoothed_mean = data["mean"]
            smoothed_lower = data["lower"]
            smoothed_upper = data["upper"]

        # Plot Line (Mean)
        axes.plot(
            common_x_grid,
            smoothed_mean,
            label=system_name,
            color=color,
            linewidth=LINE_WIDTH,
        )

        # Draw opaque dashed line from last point to x-axis
        valid_indices = np.where(~np.isnan(smoothed_mean))[0]
        if len(valid_indices) > 0:
            last_idx = valid_indices[-1]
            last_x = common_x_grid[last_idx]
            last_y = smoothed_mean[last_idx]
            axes.plot(
                [last_x, last_x],
                [min_y, last_y],
                color=color,
                linestyle="--",
                alpha=0.7,
            )

        # Plot Fill (Min/Max)
        if not MIN_MAX_DISABLED:
            axes.fill_between(
                common_x_grid,
                smoothed_lower,
                smoothed_upper,
                color=color,
                alpha=0.25,
                label=f"Min/Max of {system_name}",
            )

    # Set axis limits and ticks
    xlim_max = max_x if X_AXIS_MAX is not None else x_ticks[-1]
    axes.set_xlim(x_ticks[0], xlim_max)
    axes.set_ylim(min_y, max_y)
    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks)

    # Format tick labels
    if x_data_key == "Time_Since_Start":
        # Time axis label format (hours)
        x_ticks_hour = [f"{t:g}h" for t in x_ticks]
        axes.set_xticklabels(x_ticks_hour)
    else:
        # Mini-batch ID axis label (integer format)
        axes.set_xticklabels([f"{int(t)}" for t in x_ticks])

    if y_data_key == "Accuracy":
        # Accuracy axis label format (float -> %)
        y_ticks_formatted = [f"${y*100:.0f}$" for y in y_ticks]
    else:
        # Other y-axis labels (integer format)
        y_ticks_formatted = [f"{int(t)}" for t in y_ticks]
    axes.set_yticklabels(y_ticks_formatted)

    # Configure major ticks (no negative X ticks because xlim starts at 0)
    axes.tick_params(
        axis="both", which="major", length=6, width=1.5, color="black", labelsize=16
    )
    axes.grid(True, linestyle="--", alpha=0.6, color="lightgray", zorder=1)

    axes.set_xlabel(x_label, fontsize=16)
    axes.set_ylabel(y_label, fontsize=16)
    axes.legend(
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=14,
    )
    plt.tight_layout(pad=0.2)

    output_dir = Path(
        f"{base_folder}/plots/90-avail-10-eval-0-unavail",
        # f"{base_folder}/plots/100_0_0"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_dir / f"{plot_type}_comparison.png"
    plt.savefig(file_name)
    # plt.show() # Disabled for production environment

    print(f"Plot saved to {file_name}")


if __name__ == "__main__":

    # Example usage for the original 'time' plot:
    # We shift the legend slightly up (e.g., bbox_to_anchor=(1, 0.1)) to prevent overlap.
    # Alternatively, for a horizontal legend, use: legend_ncol=3, legend_loc="upper right" (and check spacing).
    plot_comparison_chart(
        systems_data=SYSTEMS_DATA,
        plot_type="time",
        x_tick_count=10,
        y_tick_count=10,
        legend_bbox_to_anchor=(1, 0.3),  # Shifts the legend up slightly for this graph
    )

    # Example usage for the new 'batch' plot (no interpolation):
    plot_comparison_chart(
        systems_data=SYSTEMS_DATA, plot_type="batch", x_tick_count=10, y_tick_count=10
    )

    # Example usage for the new 'time_vs_batch' plot:
    plot_comparison_chart(
        systems_data=SYSTEMS_DATA,
        plot_type="time_vs_batch",
        x_tick_count=10,
        y_tick_count=4,
    )
