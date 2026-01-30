import json
import sys
import os

# --- Configuration ---
# The specific keys we want to copy from the source's "hyperparameters"
KEYS_TO_COPY = [
    "avl_events_mobiperf_3st_50",
    "avl_events_mobiperf_3st_75",
    "avl_events_mobiperf_2st",
    "client_notify",
    "wait_until_next_avl",
    "avl_events_syn_0",
    "avl_events_syn_20",
    "avl_events_syn_50",
]
# ---------------------


def merge_keys(src_file_path, dst_file_path):
    try:
        # Read the source file to get the data
        with open(src_file_path, "r") as f:
            src_data = json.load(f)

        # Read the destination file to update it
        with open(dst_file_path, "r") as f:
            dst_data = json.load(f)

        # Check if 'hyperparameters' exists in both
        if "hyperparameters" not in src_data:
            print(
                f"Error: 'hyperparameters' key missing in source {src_file_path}",
                file=sys.stderr,
            )
            return

        if "hyperparameters" not in dst_data:
            print(
                f"Warning: 'hyperparameters' key missing in dest {dst_file_path}. Creating it.",
                file=sys.stderr,
            )
            dst_data["hyperparameters"] = {}

        # Copy each specified key from source to destination
        src_params = src_data["hyperparameters"]
        dst_params = dst_data["hyperparameters"]

        for key in KEYS_TO_COPY:
            if key in src_params:
                dst_params[key] = src_params[key]
            else:
                print(
                    f"Warning: Key '{key}' not found in source {src_file_path}",
                    file=sys.stderr,
                )

        # Write the modified data back to the destination file
        # We write to a temporary file first, then rename it for an "atomic" save.
        # This prevents data loss if the script is interrupted.
        tmp_file_path = dst_file_path + ".tmp"
        with open(tmp_file_path, "w") as f:
            # Use indent=4 for readable JSON, remove it for minified files
            json.dump(dst_data, f, indent=4)

        # Replace the original destination file with the new one
        os.rename(tmp_file_path, dst_file_path)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {e.filename}: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python3 {sys.argv[0]} <source_json_file> <destination_json_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    src_file = sys.argv[1]
    dst_file = sys.argv[2]

    merge_keys(src_file, dst_file)
