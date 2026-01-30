import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_selector_p50(log_file, log_line, suffix, syn_percent):
    x_ticks = []
    stat_util_50, stat_util_100, stat_util_200 = [], [], []
    speed_50, speed_100, speed_200 = [], [], []
    round_50, round_100, round_200 = [], [], []

    if log_line == "Train":
        pattern = re.compile(r"Train selector stats summary: ({.*})")
    else:
        pattern = re.compile(r"Eval selector stats summary: ({.*})")

    with open(log_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                stats = eval(match.group(1))
                if all(stats[f]["p50"] is not None for f in stats):
                    stat_util_50.append(stats["stat_util_last_50"]["p50"])
                    stat_util_100.append(stats["stat_util_last_100"]["p50"])
                    stat_util_200.append(stats["stat_util_last_200"]["p50"])
                    speed_50.append(stats["speed_last_50"]["p50"])
                    speed_100.append(stats["speed_last_100"]["p50"])
                    speed_200.append(stats["speed_last_200"]["p50"])
                    round_50.append(stats["round_last_50"]["p50"])
                    round_100.append(stats["round_last_100"]["p50"])
                    round_200.append(stats["round_last_200"]["p50"])
                    x_ticks.append(len(x_ticks) * 5)

    def smooth(data):
        return gaussian_filter1d(np.array(data), sigma=75)

    def plot_group(d50, d100, d200, ylabel, filename, colors):
        plt.figure(figsize=(6, 3))
        plt.plot(x_ticks, smooth(d50), label="last_50", color=colors[0])
        plt.plot(x_ticks, smooth(d100), label="last_100", color=colors[1])
        plt.plot(x_ticks, smooth(d200), label="last_200", color=colors[2])
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Selector p50 {syn_percent}")
        plt.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_cdf(data_list, labels, title, filename, colors):
        plt.figure(figsize=(6, 3))
        for i, (data, label, color) in enumerate(zip(data_list, labels, colors)):
            sorted_data = np.sort(data)
            cdf = np.linspace(0, 1, len(sorted_data))
            plt.plot(sorted_data, cdf, label=label, color=color)

            if "100" in label:
                p50_idx = int(0.5 * len(sorted_data))
                p90_idx = int(0.9 * len(sorted_data))
                p50_val = sorted_data[p50_idx]
                p90_val = sorted_data[p90_idx]

                plt.axvline(p50_val, color="black", linestyle="--", alpha=0.6)
                plt.text(
                    p50_val,
                    0.52,
                    f"p50: {p50_val:.2f}",
                    color="black",
                    fontsize=10,
                    rotation=0,
                    va="bottom",
                    ha="center",
                )

                plt.axvline(p90_val, color="black", linestyle="--", alpha=0.6)
                plt.text(
                    p90_val,
                    0.92,
                    f"p90: {p90_val:.2f}",
                    color="black",
                    fontsize=10,
                    rotation=0,
                    va="bottom",
                    ha="center",
                )

        plt.xlabel(title)
        plt.ylabel("CDF")
        plt.title(f"CDF of {title} {syn_percent}")
        plt.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    # Line plots
    plot_group(
        stat_util_50,
        stat_util_100,
        stat_util_200,
        f"{suffix} Stat Utility",
        f"{log_line}_selector_stat_util_{suffix}_syn{syn_percent}.png",
        ["blue", "purple", "cyan"],
    )

    plot_group(
        speed_50,
        speed_100,
        speed_200,
        f"{suffix} Speed",
        f"{log_line}_selector_speed_{suffix}_syn{syn_percent}.png",
        ["green", "orange", "brown"],
    )

    plot_group(
        round_50,
        round_100,
        round_200,
        f"{suffix} Round",
        f"{log_line}_selector_round_{suffix}_syn{syn_percent}.png",
        ["red", "magenta", "gray"],
    )

    # CDF plots
    plot_cdf(
        [stat_util_50, stat_util_100, stat_util_200],
        labels=["last_50", "last_100", "last_200"],
        title=f"{suffix} Stat Utility",
        filename=f"{log_line}_selector_stat_util_cdf_{suffix}_syn{syn_percent}.png",
        colors=["blue", "purple", "cyan"],
    )

    plot_cdf(
        [speed_50, speed_100, speed_200],
        labels=["last_50", "last_100", "last_200"],
        title=f"{suffix} Speed",
        filename=f"{log_line}_selector_speed_cdf_{suffix}_syn{syn_percent}.png",
        colors=["green", "orange", "brown"],
    )


# 🧪 Usage Example
log_file_j = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_13_05_00_40_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log"
log_file_s = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20_truncated2.log"
# log_file_j = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_12_05_23_00_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20copy.log"
# log_file_s = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20_truncated.log"
# log_file_j = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_jayne_11_05_11_51_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log"
# log_file_s = "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log"


log_file_s2 = "agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50_truncated.log"
log_file_j2 = "agg_sheph_12_05_02_18_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50_copy_truncate.log"

extract_selector_p50(log_file_j, "Train", "Felix", "UNAVL(20%)")
extract_selector_p50(log_file_s, "Train", "OORT+Async", "UNAVL(20%)")
extract_selector_p50(log_file_j, "Eval", "Felix", "UNAVL(20%)")

# extract_selector_p50(log_file_j2, "Train", "Felix", "UNAVL(50%)")
# extract_selector_p50(log_file_s2, "Train", "OORT+Async", "UNAVL(50%)")
# extract_selector_p50(log_file_j2, "Eval", "Felix", "UNAVL(50%)")

# extract_selector_p50(log_file_s, "eval", "async_oort")
