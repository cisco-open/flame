import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_and_plot_p50(log_file, suffix, syn_percent):
    p50_staleness = []
    p50_utility = []
    p50_speed = []
    x_ticks = []

    pattern = re.compile(r"_agg_training_stats: ({.*})")

    with open(log_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                stats = eval(match.group(1))
                p50_staleness.append(stats["staleness"]["p50"])
                p50_utility.append(stats["stat_utility"]["p50"])
                p50_speed.append(stats["trainer_speed"]["p50"])
                x_ticks.append(len(x_ticks) * 5)

    # Smooth data
    p50_staleness = gaussian_filter1d(np.array(p50_staleness), sigma=9)
    p50_utility = gaussian_filter1d(np.array(p50_utility), sigma=9)
    p50_speed = gaussian_filter1d(np.array(p50_speed), sigma=5)

    # Plot and save each
    def plot_metric(data, color, metric, filename_prefix):
        plt.figure(figsize=(6, 3))
        plt.plot(x_ticks, data, color=color)
        plt.xlabel("Round")
        plt.ylabel("p50 Value")
        plt.title(f"{metric} Over Rounds {syn_percent}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{filename_prefix}_{suffix}_syn{syn_percent}.png", bbox_inches="tight"
        )
        plt.close()

    plot_metric(p50_staleness, "red", "Staleness", "staleness_p50")
    plot_metric(p50_utility, "blue", "Utility", "utility_p50")
    plot_metric(p50_speed, "green", "Trainer Speed", "speed_p50")


# Run for both logs
log_file_j = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_jayne_11_05_11_51_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log"
log_file_s = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log"

extract_and_plot_p50(log_file_j, "felix", "UNAVL(20%)")
extract_and_plot_p50(log_file_s, "OORT+Async", "UNAVL(20%)")

log_file_s2 = "agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50_truncated.log"
log_file_j2 = "agg_sheph_12_05_02_18_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50_copy_truncate.log"

extract_and_plot_p50(log_file_j2, "felix", "UNAVL(50%)")
extract_and_plot_p50(log_file_s2, "OORT+Async", "UNAVL(50%)")
