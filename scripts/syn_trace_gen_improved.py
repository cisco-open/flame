import numpy as np
import json
import os
import glob
import re
import matplotlib.pyplot as plt

def batch_inject_and_plot(folder_path='.', max_trainers=100, train_p=0.90, eval_p=0.05, 
                          unavail_p=0.05, total_minutes=3000, interval=10, stickiness=0.95):
    states = ['UN_AVL', 'AVL_EVAL', 'AVL_TRAIN']
    target_dist = np.array([unavail_p, eval_p, train_p])
    total_rounds = total_minutes // interval
    
    # 1. ROBUST TRANSITION MATRIX (Detailed Balance)
    n = len(target_dist)
    trans_matrix = np.zeros((n, n))
    
    # We define a base transition rate 'alpha'
    # High stickiness means alpha is small
    alpha = 1.0 - stickiness 
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # The probability of moving i -> j depends on the target density of j
                # This ensures the Markov chain is 'pulled' toward the target distribution
                trans_matrix[i, j] = alpha * target_dist[j]
        
        # The diagonal (staying put) is 1 minus the sum of moving elsewhere
        trans_matrix[i, i] = 1.0 - np.sum(trans_matrix[i, :])

    # 2. File and Key Setup
    key_name = f"avl_events_syn_train_{int(train_p*100)}_eval_{int(eval_p*100)}_unavail_{int(unavail_p*100)}"
    search_pattern = os.path.join(folder_path, "trainer_*.json")
    files = glob.glob(search_pattern)
    files.sort(key=lambda f: int(re.sub('\D', '', os.path.basename(f)) or 0))
    files_to_process = files[:max_trainers]

    if not files_to_process:
        print(f"No files found in {folder_path}")
        return

    all_states_history = np.zeros((total_rounds, len(files_to_process)))

    for node_idx, file_path in enumerate(files_to_process):
        with open(file_path, 'r') as f:
            data = json.load(f)

        node_trace = []
        # INITIALIZE BASED ON TARGET
        curr_state_idx = np.random.choice([0, 1, 2], p=target_dist)
        all_states_history[0, node_idx] = curr_state_idx
        last_state_name = states[curr_state_idx]
        node_trace.append((0, last_state_name))
        
        for r in range(1, total_rounds):
            curr_state_idx = np.random.choice([0, 1, 2], p=trans_matrix[curr_state_idx])
            all_states_history[r, node_idx] = curr_state_idx
            curr_state_name = states[curr_state_idx]
            
            if curr_state_name != last_state_name:
                node_trace.append((r * interval * 60, curr_state_name))
                last_state_name = curr_state_name

        if "hyperparameters" in data:
            data["hyperparameters"][key_name] = str(node_trace)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

    # 3. Updated Two-Line Validation Plot
    time_axis = np.arange(total_rounds) * interval * 60
    
    # Calculate Residency Percentages
    train_pct = np.mean(all_states_history == 2, axis=1) * 100
    # Total Available = Anything NOT Unavail (state index 0)
    combined_pct = np.mean(all_states_history > 0, axis=1) * 100

    plt.figure(figsize=(14, 6))
    
    # Actual Data Lines
    plt.plot(time_axis, combined_pct, label='Total Available (TRAIN + EVAL)', 
             color='green', linewidth=1.8, alpha=0.9)
    plt.plot(time_axis, train_pct, label='Actual AVL_TRAIN', 
             color='blue', linewidth=1.2, alpha=0.7)
    
    # Expectation Guide Lines
    expected_total = (train_p + eval_p) * 100
    expected_train = train_p * 100
    
    plt.axhline(y=expected_total, color='green', linestyle='--', 
                alpha=0.4, label=f'Total Expectation ({int(expected_total)}%)')
    plt.axhline(y=expected_train, color='blue', linestyle=':', 
                alpha=0.5, label=f'Train Expectation ({int(expected_train)}%)')

    plt.title(f"System Availability: {int(train_p*100)}/{int(eval_p*100)}/{int(unavail_p*100)} (Stickiness {stickiness})")
    plt.ylabel("% of Total Trainers")
    plt.xlabel("Time (seconds)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    filename = f'availability_two_lines_{int(train_p*100)}_{int(eval_p*100)}_{int(unavail_p*100)}_stickiness_{stickiness}.png'
    plt.savefig(filename)
    plt.show()

    return all_states_history


import pandas as pd
import seaborn as sns

def plot_trainer_distribution(history_matrix, train_p, eval_p, 
                          unavail_p, stickiness, states=['UN_AVL', 'AVL_EVAL', 'AVL_TRAIN']):
    """
    history_matrix: (total_rounds, num_trainers)
    """
    total_rounds = history_matrix.shape[0]
    num_trainers = history_matrix.shape[1]
    
    # 1. Calculate the % of time each trainer spent in each state
    # Result: A list of 100 values for each state
    trainer_stats = {
        'UN_AVL': np.mean(history_matrix == 0, axis=0) * 100,
        'AVL_EVAL': np.mean(history_matrix == 1, axis=0) * 100,
        'AVL_TRAIN': np.mean(history_matrix == 2, axis=0) * 100
    }

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    colors = {'UN_AVL': 'red', 'AVL_EVAL': 'blue', 'AVL_TRAIN': 'green'}
    
    for state in states:
        # We use a histogram with a Kernel Density Estimate (KDE) for smoothness
        sns.histplot(trainer_stats[state], bins=20, kde=True, 
                     color=colors[state], label=state, alpha=0.4)

    plt.title(f"Distribution of State Residency across {num_trainers} Trainers")
    plt.xlabel("Percentage of Total Simulation Time (%)")
    plt.ylabel("Number of Trainers")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'trainer_state_distribution_{int(train_p*100)}_{int(eval_p*100)}_{int(unavail_p*100)}_stickiness_{stickiness}.png')
    plt.show()

def plot_stacked_trainer_composition(history_matrix, train_p, eval_p, 
                          unavail_p, stickiness, max_trainers=100):
    """
    history_matrix: (total_rounds, num_trainers)
    """
    # 1. Calculate residency percentages
    # We transpose so we have (num_trainers, 3)
    unavail_pct = np.mean(history_matrix == 0, axis=0) * 100
    eval_pct = np.mean(history_matrix == 1, axis=0) * 100
    train_pct = np.mean(history_matrix == 2, axis=0) * 100
    
    num_trainers = history_matrix.shape[1]
    trainer_indices = np.arange(1, num_trainers + 1)

    plt.figure(figsize=(15, 6))

    # 2. Plotting the stacks
    # Bottom: TRAIN
    plt.bar(trainer_indices, train_pct, color='green', label='AVL_TRAIN', alpha=0.8)
    
    # Middle: EVAL (starts at the top of TRAIN)
    plt.bar(trainer_indices, eval_pct, bottom=train_pct, 
            color='blue', label='AVL_EVAL', alpha=0.8)
    
    # Top: UN_AVL (starts at the top of TRAIN + EVAL)
    plt.bar(trainer_indices, unavail_pct, bottom=train_pct + eval_pct, 
            color='red', label='UN_AVL', alpha=0.8)

    # 3. Formatting
    plt.axhline(y=90, color='white', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("Trainer ID")
    plt.ylabel("Percentage of Total Time (%)")
    plt.title(f"Composition of State Residency per Trainer (Total: {num_trainers})")
    plt.xlim(0, num_trainers + 1)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
    
    plt.tight_layout()
    plt.savefig(f'trainer_stacked_composition_{int(train_p*100)}_{int(eval_p*100)}_{int(unavail_p*100)}_stickiness_{stickiness}.png')
    plt.show()

# # Setup 1
# train_p=1.0
# eval_p=0.0
# unavail_p=0.0
# stickiness=0.40

# # Setup 2
# train_p=0.90
# eval_p=0.10
# unavail_p=0.0
# stickiness=0.40

# Setup 3
train_p=0.50
eval_p=0.30
unavail_p=0.20
stickiness=0.40



# # For validation only
# train_p=0.50
# eval_p=0.30
# unavail_p=0.20
# stickiness=0.90


# Run the corrected version
all_states_history = batch_inject_and_plot(
    folder_path='/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/',
    max_trainers=100,
    train_p=train_p,
    eval_p=eval_p,
    unavail_p=unavail_p,
    stickiness=stickiness 
)

plot_trainer_distribution(all_states_history,train_p=train_p,
    eval_p=eval_p,
    unavail_p=unavail_p,
    stickiness=stickiness )
plot_stacked_trainer_composition(all_states_history,train_p=train_p,
    eval_p=eval_p,
    unavail_p=unavail_p,
    stickiness=stickiness )