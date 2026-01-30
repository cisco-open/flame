import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

### CDF of agg time diff


def extract_and_plot_time_diffs(log_path, suffix, syn_percent):
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*aggregation finished for round"
    )
    timestamps = []

    with open(log_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                timestamps.append(ts)

    if not timestamps:
        print(f"[{suffix}] No valid timestamps found.")
        return

    # Skip first 10 minutes
    start_ts = timestamps[0]
    filtered = [ts for ts in timestamps if (ts - start_ts).total_seconds() > 600]
    if len(filtered) < 2:
        print(f"[{suffix}] Not enough data after 10 minutes.")
        return

    diffs = [
        (filtered[i] - filtered[i - 1]).total_seconds() for i in range(1, len(filtered))
    ]

    # Compute metrics
    sorted_diffs = np.sort(diffs)
    cdf = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    p50_val = np.percentile(sorted_diffs, 50)
    p90_val = np.percentile(sorted_diffs, 90)
    avg = np.mean(diffs)
    num_rounds = len(diffs)
    est_total_time = avg * num_rounds
    actual_total_time = (filtered[-1] - filtered[0]).total_seconds()

    print(f"[{suffix}] Rounds: {num_rounds}")
    print(f"[{suffix}] Avg * Rounds = {est_total_time:.2f}s")
    print(f"[{suffix}] Actual Duration = {actual_total_time:.2f}s")
    print(f"[{suffix}] Delta = {abs(est_total_time - actual_total_time):.2f}s\n")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_diffs, cdf, linestyle="-")

    # plt.axvline(p50_val, color="black", linestyle="--", alpha=0.6)
    # plt.text(p50_val, 0.52, f"p50: {p50_val:.2f}", color="black", fontsize=10, rotation=0, va="bottom", ha="center")

    # plt.axvline(p90_val, color="black", linestyle="--", alpha=0.6)
    # plt.text(p90_val, 0.92, f"p90: {p90_val:.2f}", color="black", fontsize=10, rotation=0, va="bottom", ha="center")

    # P50 and P90 annotations
    plt.axvline(p50_val, color="black", linestyle="--")
    plt.text(
        p50_val + 1,
        0.5,
        f"P50: {p50_val:.2f}s",
        verticalalignment="center",
        color="black",
    )

    plt.axvline(p90_val, color="black", linestyle="--")
    plt.text(
        p90_val + 1,
        0.8,
        f"P90: {p90_val:.2f}s",
        verticalalignment="center",
        color="black",
    )

    plt.xlabel("Inter-round interval (s)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Aggregation Time Diffs - {suffix} {syn_percent}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rounds_{suffix}_syn{syn_percent}.png", bbox_inches="tight")
    plt.close()


# File paths
log_path_j = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_jayne_11_05_11_51_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log"
log_path_s = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log"

# Run for both logs
extract_and_plot_time_diffs(log_path_j, "Felix", "UNAVL(20%)")
extract_and_plot_time_diffs(log_path_s, "OORT+Async", "UNAVL(20%)")

log_path_s2 = "agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50_truncated.log"
log_path_j2 = "agg_sheph_12_05_02_18_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50_copy_truncate.log"

extract_and_plot_time_diffs(log_path_j2, "Felix", "UNAVL(50%)")
extract_and_plot_time_diffs(log_path_s2, "OORT+Async", "UNAVL(50%)")
