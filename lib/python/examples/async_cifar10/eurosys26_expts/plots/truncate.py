import re
from datetime import datetime, timedelta


## code to truncate logs for the first n mins


def truncate_log_after_420_min(log_path, output_path):
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    start_time = None
    truncated_lines = []

    with open(log_path, "r") as file:
        for line in file:
            match = timestamp_pattern.search(line)
            if match:
                current_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if not start_time:
                    start_time = current_time
                if current_time - start_time > timedelta(minutes=33):
                    break
            truncated_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(truncated_lines)

    print(f"✅ Truncated log saved to: {output_path}")


# truncate_log_after_420_min("agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50.log","agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50_truncated.log")
# truncate_log_after_420_min("agg_sheph_12_05_02_18_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50_copy.log","agg_sheph_12_05_02_18_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50_copy_truncate.log")
truncate_log_after_420_min(
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log",
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/plots/agg_sheph_11_05_11_36_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20_truncated2.log",
)
