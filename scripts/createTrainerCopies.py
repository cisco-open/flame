"""
Script to create sequential copies of a trainer JSON file with modifications.

Usage:
- Edit SRC_DIR to set the source directory.
- Edit INPUT_FILE to specify the source file (e.g., "trainer_99.json").
- Edit END_TRAINER_NUM to specify the last trainer number to create.
- Run the script. It will create copies from trainer_{START+1}.json to trainer_{END}.json,
  where START is extracted from the INPUT_FILE name.

For each copy, the script:
1. Reads the taskid from the source file, extracts the last 4 characters,
   treats them as an integer, increments by one for each new file, and
   replaces the last 4 characters with the incremented value (zero-padded)
2. Sets hyperparameters.client_idx to match the trainer_id (file suffix)
3. Removes hyperparameters.training_delay_s field
"""

import json
import os
import re

# ---- CONFIG ----
SRC_DIR = "../lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts"
INPUT_FILE = "trainer_99.json"  # Source file to copy from
END_TRAINER_NUM = 149  # Last trainer number to create (inclusive)


def extract_trainer_num(filename):
    """Extract trainer number from filename like 'trainer_99.json'."""
    match = re.search(r"trainer_(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract trainer number from filename: {filename}")


def create_trainer_copy(src_path, dst_path, trainer_id, task_id):
    """Create a copy of the source JSON with modifications."""
    # Read source JSON
    with open(src_path, "r") as f:
        data = json.load(f)

    # 1. Update taskid with the incremented value
    data["taskid"] = task_id

    # 2. Set hyperparameters.client_idx to match trainer_id
    if "hyperparameters" not in data:
        data["hyperparameters"] = {}
    data["hyperparameters"]["client_idx"] = trainer_id

    # 3. Remove hyperparameters.training_delay_s if it exists
    if "hyperparameters" in data and "training_delay_s" in data["hyperparameters"]:
        del data["hyperparameters"]["training_delay_s"]

    # Write to destination file
    with open(dst_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Created: {dst_path} (trainer_id={trainer_id}, taskid={task_id})")


def main():
    # Extract start trainer number from input filename
    start_trainer_num = extract_trainer_num(INPUT_FILE)

    # Build source path
    src_path = os.path.join(SRC_DIR, INPUT_FILE)

    if not os.path.exists(src_path):
        print(f"Error: Source file not found: {src_path}")
        return

    # Validate END_TRAINER_NUM
    if END_TRAINER_NUM <= start_trainer_num:
        print(
            f"Error: END_TRAINER_NUM ({END_TRAINER_NUM}) must be greater than start trainer number ({start_trainer_num})"
        )
        return

    # Read source file to get initial taskid
    with open(src_path, "r") as f:
        src_data = json.load(f)

    # Extract last 4 characters of taskid and convert to integer
    try:
        original_taskid = src_data["taskid"]
        if len(original_taskid) < 4:
            print(
                f"Error: taskid is too short (less than 4 characters). taskid='{original_taskid}'"
            )
            return

        # Get prefix (all but last 4 characters) and last 4 characters
        taskid_prefix = original_taskid[:-4]
        last_four_chars = original_taskid[-4:]

        # Convert last 4 characters to integer
        last_four_int = int(last_four_chars)
    except (ValueError, KeyError) as e:
        print(
            f"Error: Could not parse taskid. taskid='{src_data.get('taskid', 'NOT FOUND')}', error={e}"
        )
        return

    # Create copies from start_trainer_num+1 to END_TRAINER_NUM
    for trainer_id in range(start_trainer_num + 1, END_TRAINER_NUM + 1):
        dst_file = f"trainer_{trainer_id}.json"
        dst_path = os.path.join(SRC_DIR, dst_file)

        # Skip if file already exists (safety check)
        if os.path.exists(dst_path):
            print(f"Warning: {dst_path} already exists, skipping...")
            last_four_int += 1  # Still increment counter even if skipping
            continue

        # Increment the last 4 digits and format back as 4-digit zero-padded string
        last_four_int += 1
        new_last_four = f"{last_four_int:04d}"  # Zero-pad to 4 digits
        new_taskid = taskid_prefix + new_last_four

        create_trainer_copy(src_path, dst_path, trainer_id, new_taskid)

    print(
        f"\nCompleted: Created trainer files from {start_trainer_num + 1} to {END_TRAINER_NUM}"
    )


if __name__ == "__main__":
    main()
