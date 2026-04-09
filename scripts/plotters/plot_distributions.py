import os
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# --- Configuration ---
PLOT_PERCENTILE_LOWER = "p20"
PLOT_PERCENTILE_UPPER = "p99"
PLOT_PERCENTILE_MID = "p50"


def parse_logs(log_paths):
    """
    Parses specific multiline blocks containing percentiles of train duration
    and stat utilities from the given log files.
    """
    system_data = {}

    for system_name, log_path in log_paths.items():
        if not os.path.exists(log_path):
            print(f"File not found: {log_path}")
            continue

        times = []
        durations_lower = []
        durations_mid = []
        durations_upper = []

        utils_lower = []
        utils_mid = []
        utils_upper = []

        start_time = None

        with open(log_path, "r") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            line = lines[i]

            # Start of the block
            if "==== Model version incremented" in line:
                # Extract timestamp
                time_match = re.search(
                    r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})", line
                )
                if not time_match:
                    continue

                timestamp = datetime.strptime(
                    time_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                )
                if start_time is None:
                    start_time = timestamp

                time_since_start_hours = (
                    timestamp - start_time
                ).total_seconds() / 3600.0

                try:
                    # Line i+1: "p20, p30, p50, p75, p90, p99 of train duration"
                    header_line_dur = lines[i + 1]
                    percentiles_list_dur = [
                        p.strip() for p in header_line_dur.split(" of ")[0].split(",")
                    ]

                    try:
                        dur_lower_idx = percentiles_list_dur.index(
                            PLOT_PERCENTILE_LOWER
                        )
                        dur_mid_idx = percentiles_list_dur.index(PLOT_PERCENTILE_MID)
                        dur_upper_idx = percentiles_list_dur.index(
                            PLOT_PERCENTILE_UPPER
                        )
                    except ValueError:
                        print(
                            f"Warning: Missing required percentiles in duration header: {header_line_dur.strip()}"
                        )
                        continue

                    # Line i+2: "3.015, 3.173, 3.501, 3.733, 4.204, 4.779"
                    duration_vals = [float(x.strip()) for x in lines[i + 2].split(",")]

                    # Line i+3: "p20, p30, p50, p75, p90, p99 of partial stat utilities"
                    header_line_util = lines[i + 3]
                    percentiles_list_util = [
                        p.strip() for p in header_line_util.split(" of ")[0].split(",")
                    ]

                    try:
                        util_lower_idx = percentiles_list_util.index(
                            PLOT_PERCENTILE_LOWER
                        )
                        util_mid_idx = percentiles_list_util.index(PLOT_PERCENTILE_MID)
                        util_upper_idx = percentiles_list_util.index(
                            PLOT_PERCENTILE_UPPER
                        )
                    except ValueError:
                        print(
                            f"Warning: Missing required percentiles in stat util header: {header_line_util.strip()}"
                        )
                        continue

                    # Line i+4: "11.5882, 11.5238, 11.2377, 10.8116, 10.4945, 10.4124"
                    util_vals = [float(x.strip()) for x in lines[i + 4].split(",")]

                    times.append(time_since_start_hours)

                    durations_lower.append(duration_vals[dur_lower_idx])
                    durations_mid.append(duration_vals[dur_mid_idx])
                    durations_upper.append(duration_vals[dur_upper_idx])

                    # Note: for stat utilities, the snippet might show decreasing values
                    # Let's just grab endpoints for bands.
                    utils_lower.append(util_vals[util_lower_idx])
                    utils_mid.append(util_vals[util_mid_idx])
                    utils_upper.append(util_vals[util_upper_idx])

                except (IndexError, ValueError) as e:
                    # If lines don't match exactly or we hit end of file, skip
                    continue

        system_data[system_name] = {
            "time": times,
            "duration_lower": durations_lower,
            "duration_mid": durations_mid,
            "duration_upper": durations_upper,
            "util_lower": utils_lower,
            "util_mid": utils_mid,
            "util_upper": utils_upper,
        }

    return system_data


def plot_distributions(system_data):
    """
    Creates the comparative timeline plots for stat utility and train duration (speed).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    colors = ["C0", "C1", "C2", "C3"]

    span_label = f"{PLOT_PERCENTILE_LOWER}-{PLOT_PERCENTILE_UPPER} span"
    mid_label = f"{PLOT_PERCENTILE_MID}"

    for idx, (system_name, data) in enumerate(system_data.items()):
        color = colors[idx % len(colors)]
        times = data["time"]

        if not times:
            continue

        # Plot 1: Stat Util vs Time
        ax1.plot(
            times,
            data["util_mid"],
            label=f"{system_name} ({mid_label})",
            color=color,
            linewidth=2,
        )
        ax1.fill_between(
            times,
            [min(l, u) for l, u in zip(data["util_lower"], data["util_upper"])],
            [max(l, u) for l, u in zip(data["util_lower"], data["util_upper"])],
            color=color,
            alpha=0.2,
            label=f"{system_name} ({span_label})",
        )

        # Plot 2: Speed (Train Duration) vs Time
        ax2.plot(
            times,
            data["duration_mid"],
            label=f"{system_name} ({mid_label})",
            color=color,
            linewidth=2,
        )
        ax2.fill_between(
            times,
            data["duration_lower"],
            data["duration_upper"],
            color=color,
            alpha=0.2,
            label=f"{system_name} ({span_label})",
        )

    # Styling for Stat Util plot
    ax1.set_title("Stat Utility Over Time", fontsize=14)
    ax1.set_xlabel("Time Since Start (Hours)", fontsize=12)
    ax1.set_ylabel("Stat Utility", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    # Apply solid black axis lines
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color("black")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # Styling for Train Duration plot
    # Note: If you prefer "Speed" (1/duration) as sketched, you can invert these arrays!
    ax2.set_title("Train Duration (Speed) Over Time", fontsize=14)
    ax2.set_xlabel("Time Since Start (Hours)", fontsize=12)
    ax2.set_ylabel("Train Duration (Seconds)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    # Apply solid black axis lines
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("black")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()

    output_dir = Path("plots/")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "distribution_over_time.png"
    plt.savefig(out_path)
    print(f"Plot saved successfully to {out_path}")


if __name__ == "__main__":
    # Add your files here to compare different systems
    logs_to_compare = {
        "Oort": "/Users/gaurav/Projects/flame_clone/scripts/logs/nsdi_poster/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_16_03_21_48.log",
        # Uncomment and add another file like Random to compare!
        "Random": "/Users/gaurav/Projects/flame_clone/scripts/logs/nsdi_poster/test_agg_fedFwd_distilbert_agnews_lr0.01_client_num_10_numerical_16_03_22_00.log",
    }

    parsed_data = parse_logs(logs_to_compare)
    plot_distributions(parsed_data)
