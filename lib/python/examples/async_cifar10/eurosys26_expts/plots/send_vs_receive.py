import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

#### print count of msgs sent/received


def analyze_multiple_logs(log_paths, labels):
    send_train_counts, send_eval_counts = [], []
    recv_train_counts, recv_eval_counts = [], []

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

        send_train_counts.append(send_train)
        send_eval_counts.append(send_eval)
        recv_train_counts.append(recv_train)
        recv_eval_counts.append(recv_eval)

    N = len(log_paths)
    width = 0.35
    group_gap = 1.0  # space between send and receive groups
    send_x = np.arange(N)
    recv_x = send_x + N + group_gap

    fig, ax = plt.subplots(figsize=(10, 6))

    # Send bars
    ax.bar(send_x, send_train_counts, width, label="Send Train")
    ax.bar(send_x, send_eval_counts, width, bottom=send_train_counts, label="Send Eval")

    # Receive bars
    ax.bar(recv_x, recv_train_counts, width, label="Recv Train")
    ax.bar(recv_x, recv_eval_counts, width, bottom=recv_train_counts, label="Recv Eval")

    # Total annotations
    for i in range(N):
        send_total = send_train_counts[i] + send_eval_counts[i]
        recv_total = recv_train_counts[i] + recv_eval_counts[i]
        ax.text(
            send_x[i],
            send_total + 2,
            f"Total: {send_total}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            recv_x[i],
            recv_total + 2,
            f"Total: {recv_total}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        ax.text(
            send_x[i],
            send_train_counts[i] / 2,
            str(send_train_counts[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            send_x[i],
            send_train_counts[i] + send_eval_counts[i] / 2,
            str(send_eval_counts[i]),
            ha="center",
            va="center",
            fontsize=8,
        )

        ax.text(
            recv_x[i],
            recv_train_counts[i] / 2,
            str(recv_train_counts[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            recv_x[i],
            recv_train_counts[i] + recv_eval_counts[i] / 2,
            str(recv_eval_counts[i]),
            ha="center",
            va="center",
            fontsize=8,
        )

    # Axes setup
    all_totals = [
        a + b
        for a, b in zip(
            send_train_counts + recv_train_counts, send_eval_counts + recv_eval_counts
        )
    ]
    ax.set_ylim(0, max(all_totals) * 1.2)

    ax.set_ylabel("Message Count")
    ax.set_title("Send and Receive Messages per Log File (Stacked Train/Eval)")
    ax.set_xticks(list(send_x) + list(recv_x))
    ax.set_xticklabels(labels * 2, rotation=15)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("combined_send_recv_msgs.png", bbox_inches="tight")
    plt.close()


# Example usage
log_paths = [
    "agg_sheph_13_05_08_29_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log",
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_jayne_11_05_15_20_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log",
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs/agg_wash_11_05_11_49_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn20.log",
]
labels = ["Felix_20", "OORT+Async_20", "OORT+Trace_20"]

analyze_multiple_logs(log_paths, labels)
