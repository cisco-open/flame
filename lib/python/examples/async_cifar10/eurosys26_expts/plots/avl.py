import os
import json
import ast
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.figsize": [6, 3],
        "legend.fontsize": 14,
        "legend.columnspacing": 2,
        "legend.handletextpad": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

#### plotting the synthetic trace distribution

FOLDER = "/home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir0.1_num300_traceFail_6d_3state_oort/"
SECONDS_IN_DAY = 86400
keys = ["avl_events_syn_0", "avl_events_syn_20", "avl_events_syn_50"]
timelines_by_key = {key: [] for key in keys}

for filename in os.listdir(FOLDER):
    if (
        filename.startswith("trainer_")
        and filename.endswith(".json")
        and "_test" not in filename
    ):
        with open(os.path.join(FOLDER, filename), "r") as f:
            data = json.load(f)
            for key in keys:
                events_str = data["hyperparameters"][key]
                events = ast.literal_eval(events_str)
                events.append((SECONDS_IN_DAY, "UN_AVL"))
                timeline = [0] * SECONDS_IN_DAY
                for i in range(len(events) - 1):
                    start, state = events[i]
                    end = events[i + 1][0]
                    if state == "AVL_TRAIN":
                        for t in range(start, end):
                            timeline[t] = 1
                timelines_by_key[key].append(timeline)

# Aggregate and smooth
window_size = 600
kernel = np.ones(window_size) / window_size
aggregated_smoothed = {}

for key in keys:
    # aggregated = np.array([sum(t[i] for t in timelines_by_key[key]) for i in range(SECONDS_IN_DAY)])
    num_trainers = len(timelines_by_key[key])
    aggregated = np.array(
        [sum(t[i] for t in timelines_by_key[key]) for i in range(SECONDS_IN_DAY)]
    )
    percent_availability = (aggregated / num_trainers) * 100
    smoothed = np.convolve(percent_availability, kernel, mode="same")
    aggregated_smoothed[key] = smoothed

# Plot
colors = {
    "avl_events_syn_0": "darkorange",
    "avl_events_syn_20": "crimson",
    "avl_events_syn_50": "teal",
}
labels = {
    "avl_events_syn_0": "UNAVL (0%)",
    "avl_events_syn_20": "UNAVL (10%)",
    "avl_events_syn_50": "UNAVL (50%)",
}

for key in keys:
    plt.plot(
        range(SECONDS_IN_DAY),
        aggregated_smoothed[key],
        label=labels[key],
        color=colors[key],
        zorder=3,
    )

plt.grid(True, color="gainsboro", zorder=1)
# plt.xlabel("Time Elapsed (s)")
hour_ticks = np.arange(0, 86401, 10800)  # every 3 hours
plt.xticks(hour_ticks, [str(int(t // 3600)) for t in hour_ticks])
plt.xlabel("Time Elapsed (hours)")
plt.ylabel("% Availability")
plt.xlim(1350, 86000)
plt.ylim(40, 102)
# plt.legend(ncol=2)
# Legend styling (improved)
legend_location = "lower right"
legend_fontsize = None
plot_legend = True
set_legend_fontsize = (
    plt.rcParams.get("legend.fontsize") if legend_fontsize is None else legend_fontsize
)
if plot_legend:
    # plt.legend(ncol=2, loc=legend_location, fontsize=set_legend_fontsize, frameon=True)
    plt.legend(
        ncol=2,
        loc="upper center",
        fontsize=set_legend_fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, 0.68),  # ⬅️ Adjust this Y value as needed
    )

# plt.savefig("avl_train_plot_combined.png", bbox_inches='tight')
plt.savefig("avl_train_plot_combined.pdf", bbox_inches="tight")
plt.show()
