import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


def analyze_comm_cost(log_paths, labels):
    send_train_costs, send_eval_costs = [], []
    recv_train_costs, recv_eval_costs = [], []

    for log_path in log_paths:
        send_train = send_eval = recv_train = recv_eval = 0
        start_time = None

        with open(log_path, "r") as file:
            for line in file:
                timestamp = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line
                )
                if timestamp:
                    current_time = datetime.strptime(
                        timestamp.group(1), "%Y-%m-%d %H:%M:%S,%f"
                    )
                    if not start_time:
                        start_time = current_time
                    if current_time - start_time < timedelta(minutes=10):
                        continue
                    if "sending weights" in line and "train" in line:
                        send_train += 1
                    if "sending weights" in line and "eval" in line:
                        send_eval += 1
                    if "Received weights from" in line:
                        recv_train += 1
                    if "received eval message" in line:
                        recv_eval += 1

        # MB to GB
        send_train_costs.append(send_train / 1024)
        send_eval_costs.append(send_eval / 1024)
        recv_train_costs.append(recv_train / 1024)
        recv_eval_costs.append((recv_eval * 0.1) / 1024)

    N = len(log_paths)
    width = 0.35
    group_gap = 1.0
    group_spacing = 0.6  # tighter than previous 1.0 + N
    send_x = np.arange(N)
    recv_x = send_x + group_spacing
    # send_x = np.arange(N)
    # recv_x = send_x + N + group_gap

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars
    ax.bar(send_x, send_train_costs, width, label="Send Train")
    ax.bar(send_x, send_eval_costs, width, bottom=send_train_costs, label="Send Eval")

    ax.bar(recv_x, recv_train_costs, width, label="Recv Train")
    ax.bar(recv_x, recv_eval_costs, width, bottom=recv_train_costs, label="Recv Eval")

    # Annotations
    for i in range(N):
        st, se = send_train_costs[i], send_eval_costs[i]
        rt, rev = recv_train_costs[i], recv_eval_costs[i]

        send_total = st + se
        recv_total = rt + rev

        ax.text(
            send_x[i],
            send_total + 0.05,
            f"Total: {send_total:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            recv_x[i],
            recv_total + 0.05,
            f"Total: {recv_total:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        ax.text(send_x[i], st / 2, f"{st:.2f}", ha="center", va="center", fontsize=8)
        ax.text(
            send_x[i], st + se / 2, f"{se:.2f}", ha="center", va="center", fontsize=8
        )

        ax.text(recv_x[i], rt / 2, f"{rt:.2f}", ha="center", va="center", fontsize=8)
        ax.text(
            recv_x[i], rt + rev / 2, f"{rev:.2f}", ha="center", va="center", fontsize=8
        )

    # Axis setup
    all_totals = [
        a + b
        for a, b in zip(
            send_train_costs + recv_train_costs, send_eval_costs + recv_eval_costs
        )
    ]
    ax.set_ylim(0, max(all_totals) * 1.2)

    ax.set_ylabel("Communication Cost (GB)")
    ax.set_title("Send and Receive Communication Cost per Log File")
    ax.set_xticks(list(send_x) + list(recv_x))
    ax.set_xticklabels(labels * 2, rotation=15)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("combined_comm_cost.png", bbox_inches="tight")
    plt.close()


# Example usage
log_paths = [
    "agg_sheph_13_05_08_29_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log",
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_jayne_11_05_15_20_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log",
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs/agg_wash_11_05_11_49_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn20.log",
]
labels = ["Felix", "OORT+Async", "OORT+Trace"]

analyze_comm_cost(log_paths, labels)
