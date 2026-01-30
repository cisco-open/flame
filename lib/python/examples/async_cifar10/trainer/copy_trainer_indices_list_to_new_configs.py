import os
import json
import sys


def copy_trainer_indices_list(alpha):
    # Define source and target directories based on the alpha value
    source_dir = f"config_{alpha}_num300_traceFail_48h"
    target_dir = f"config_{alpha}_num300_traceFail_6d_3state_oort"

    # Check if directories exist
    if not os.path.isdir(source_dir) or not os.path.isdir(target_dir):
        print(
            f"One or both of the directories {source_dir} and {target_dir} do not exist."
        )
        return

    # Iterate through 1 to 300 to access each trainer_X.json
    for i in range(1, 301):
        source_file = os.path.join(source_dir, f"trainer_{i}.json")
        target_file = os.path.join(target_dir, f"trainer_{i}.json")

        # Ensure the source and target files exist
        if not os.path.isfile(source_file) or not os.path.isfile(target_file):
            print(
                f"One or both of the files {source_file} and {target_file} do not exist."
            )
            continue

        # Load data from source and target files
        with open(source_file, "r") as src:
            source_data = json.load(src)
        with open(target_file, "r") as tgt:
            target_data = json.load(tgt)

        # Copy the trainer_indices_list from source to target
        if (
            "hyperparameters" in source_data
            and "trainer_indices_list" in source_data["hyperparameters"]
        ):
            target_data["hyperparameters"]["trainer_indices_list"] = source_data[
                "hyperparameters"
            ]["trainer_indices_list"]
        else:
            print(f"trainer_indices_list not found in {source_file}")
            continue

        # Save the updated target file
        with open(target_file, "w") as tgt:
            json.dump(target_data, tgt, indent=4)

        print(f"Updated trainer_indices_list in {target_file}")


# Run the script with an alpha value
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python copy_trainer_indices_list_to_new_configs.py <alpha>")
    else:
        alpha = f"dir{sys.argv[1]}"
        copy_trainer_indices_list(alpha)
