import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- User Input ---
# List your 4 CSV file names here
csv_files = [
    # 'output/sync_1hr_train_times.csv',
    # 'output/sync_stagglers_1hr_train_times.csv',
    'output/train_times_noDelay_2_5hrs.csv',
    # 'output/train_times_delayBy20_3hrs.csv',
    # 'output/train_times_noDelay_slow_4hr.csv',      # This actually had delay
    # 'output/train_times_noDelay_2_5hrs.csv',          # Assuming this will not show up in the proportion
    # 'output/train_times_delay_slow_6hr.csv',
    'output/train_times_delayBy3_2.5hr_sample.csv',
    'async_with_stragglers.csv',
    'async_no_stragglers.csv',
]

# X-axis labels for the bars
x_labels = [
    'Sync\n(no stragglers)',
    'Sync\n(stragglers)',
    'Async\n(no stragglers)',
    'Async\n(stragglers)',
]

# --- Data Processing ---
training_proportions = []
idle_proportions = []

for file in csv_files:
    # Read only the last line of the CSV file
    # We use tail(1) to get the last row
    try:
        df = pd.read_csv(file).tail(1)

        # Extract the values
        # Find the second highest model_version value
        # Find the tuple columns

        # model_versions = df['_model_version']
        # max_version = model_versions.max()
        # second_highest_version = max_version - 1

        # # Find the index of the last row where model_version == second_highest_version
        # mask = model_versions == second_highest_version
        # if mask.any():
        #     # idx = mask[mask].index[-1]
        #     idx = df[mask].index[-1]
        #     source = f"model_version == {second_highest_version}, idx: {idx}"
        # else:
        #     idx = df.index[-1]
        #     source = f"fallback to last row (no model_version == {second_highest_version}), idx: {idx}"

        # todo: verify if this fixed the issue with incorrect stall times being graphed & then remove the comment block of code above
        tuple_cols = ['round_id', 'data_id', 'iteration_id']
        if all(col in df.columns for col in tuple_cols):
            # Extract the tuples
            tuples = list(zip(df['round_id'], df['data_id'], df['iteration_id']))
            # Remove rows with NaN in any of the tuple columns
            tuples = [t for t in tuples if not any(pd.isna(x) for x in t)]
            # Sort the tuples according to the cyclic nested order (iteration_id changes fastest, then data_id, then round_id)
            tuples_sorted = sorted(tuples)
            unique_tuples = []
            [unique_tuples.append(x) for x in tuples_sorted if x not in unique_tuples]
            if len(unique_tuples) >= 2:
                target_tuple = unique_tuples[-2]
                # Find the index of the last row in the df that matches the target_tuple
                mask = (
                    (df['round_id'] == target_tuple[0]) &
                    (df['data_id'] == target_tuple[1]) &
                    (df['iteration_id'] == target_tuple[2])
                )
                if mask.any():
                    idx = df[mask].index[-1]
                    source = f"second highest tuple {target_tuple}, idx: {idx}"
                else:
                    idx = df.index[-1]
                    source = f"fallback to last row (no match for {target_tuple}), idx: {idx}"
            elif len(unique_tuples) == 1:
                target_tuple = unique_tuples[-1]
                mask = (
                    (df['round_id'] == target_tuple[0]) &
                    (df['data_id'] == target_tuple[1]) &
                    (df['iteration_id'] == target_tuple[2])
                )
                if mask.any():
                    idx = df[mask].index[-1]
                    source = f"only tuple available {target_tuple}, idx: {idx}"
                else:
                    idx = df.index[-1]
                    source = f"fallback to last row (no match for {target_tuple}), idx: {idx}"
            else:
                idx = df.index[-1]
                source = f"fallback to last row (no tuples found), idx: {idx}"
        else:
            idx = df.index[-1]
            source = f"fallback to last row (tuple columns missing), idx: {idx}"

        # Collect and print picked values
        avg_training_time = df.loc[idx, 'mean:cumulative_recv_weights_time']
        total_time = df.loc[idx, 'time_since_start']
        label_name = x_labels[len(training_proportions)]
        print(f"[DEBUG][{label_name}] {source}, avg_training_time: {avg_training_time}, total_time: {total_time}")

        # Calculate proportions
        
        train_proportion = avg_training_time / total_time
        idle_proportion = 1 - train_proportion

        training_proportions.append(train_proportion)
        idle_proportions.append(idle_proportion)

    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found. Please check the filename and path.")
        # Add placeholder data to allow the script to continue for demonstration
        training_proportions.append(0)
        idle_proportions.append(0)
    except KeyError as e:
        print(f"Error: Column {e} not found in '{file}'. Please check your CSV file's header.")
        training_proportions.append(0)
        idle_proportions.append(0)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# Convert proportions to numpy arrays for plotting
training_proportions = np.array(training_proportions)
idle_proportions = np.array(idle_proportions)

# Bar positions
ind = np.arange(len(x_labels))

# Create the stacked bar chart
ax.bar(ind, training_proportions, label='Stall Time')
ax.bar(ind, idle_proportions, bottom=training_proportions, label='Train Time')

# --- Chart Customization ---
ax.set_ylabel('Proportion of Time')
ax.set_title('Proportion of Time Spent in Training vs. Idle')
ax.set_xticks(ind)
ax.set_xticklabels(x_labels)
ax.legend()

# Add a note about the number of clients
plt.figtext(0.5, 0.01, 'Average across 10 clients', ha='center', fontsize=10, style='italic')


# Display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the figtext
plt.savefig("plots/stacked_bar_chart.png")
plt.show()