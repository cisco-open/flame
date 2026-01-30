import os
import json

dir1 = "/home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir10_num300_traceFail_6d_3state_oort"
dir2 = "/home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir0.1_num300_traceFail_6d_3state_oort"
properties_to_copy = ["avl_events_syn_0", "avl_events_syn_20", "avl_events_syn_50"]

for i in range(1, 301):
    file_name = f"trainer_{i}.json"
    file1_path = os.path.join(dir1, file_name)
    file2_path = os.path.join(dir2, file_name)

    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print(f"Skipping {file_name}, file missing in one of the directories")
        continue

    with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    if "hyperparameters" in data1 and "hyperparameters" in data2:
        for prop in properties_to_copy:
            if prop in data1["hyperparameters"]:
                data2["hyperparameters"][prop] = data1["hyperparameters"][prop]

    with open(file2_path, "w") as f2:
        json.dump(data2, f2, indent=4)

    print(f"Updated {file_name}")
