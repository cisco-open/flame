import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback

# --- !! FORMAT-SPECIFIC CONFIG !! ---
# Edit these to match the column names in your CSVs for each format

# Config for 'multi_column' (Last Row) format
# Reads 'train_col' and 'total_col' from the *last row*
MULTI_COLUMN_CONFIG = {
    'train_col': 'mean:cumulative_train_time_sec',
    'total_col': 'time_since_start',
    'train_label': 'Train Time',
    'idle_label': 'Stall Time'
}

# Config for 'multi_row' (Aggregation) format
# Reads *all rows* and uses one column for labels, one for values
MULTI_ROW_CONFIG = {
    'label_col_index': 0, # 0 = first column
    'value_col_index': 1  # 1 = second column
}

# --- !! MAIN PLOTTING CONFIGURATION !! ---
PLOTTING_CONFIG = {
    # Switch for all files: 'multi_row' or 'multi_column'
    'format_strategy': 'multi_row', 
    
    # Label for the Y-axis (since it's absolute)
    'y_axis_label': 'Total Count', 
    
    # List of bars to plot
    'bars': [
        {
            'filepath': 'output/communication_summary-oort_unaware.csv',
            'x_label': 'oort (unaware)',
        },
        {
            'filepath': 'output/communication_summary-oort_async_oracular.csv',
            'x_label': 'oort async (oracular)',
        },
        {
            'filepath': 'output/communication_summary-felix.csv',
            'x_label': 'felix',
        },
    ]
}

# --- !! PROCESSING STRATEGIES !! ---
# (These functions now return *absolute values*, not proportions)

def multi_column(filepath):
    """
    "Strategy 1": Reads the last line of a CSV.
    Returns a dict of {label: absolute_value}.
    """
    try:
        df = pd.read_csv(filepath).tail(1)
        
        cfg = MULTI_COLUMN_CONFIG
        train_time = df[cfg['train_col']].iloc[0]
        total_time = df[cfg['total_col']].iloc[0]
        idle_time = total_time - train_time
        
        sections = {
            cfg['train_label']: train_time,
            cfg['idle_label']: idle_time
        }
        return sections, True
        
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
    except KeyError as e:
        print(f"Error: Column {e} not found. Check MULTI_COLUMN_CONFIG.")
    except Exception as e:
        print(f"Error processing '{filepath}' with 'multi_column':\n{traceback.format_exc()}")
        
    return None, False

def multi_row(filepath):
    """
    "Strategy 2": Reads all rows from a 2-column CSV.
    Returns a dict of {label: absolute_value}.
    """
    try:
        df = pd.read_csv(filepath)
        
        cfg = MULTI_ROW_CONFIG
        label_col = df.columns[cfg['label_col_index']]
        value_col = df.columns[cfg['value_col_index']]

        df = df.set_index(label_col)
        raw_values = df[value_col]
        
        sections = {}
        for label, value in raw_values.items():
            if ':' in label:
                friendly_label = label.split(':', 1)[1] # Split at first colon
            else:
                friendly_label = label
            sections[friendly_label] = value
        
        return sections, True

    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
    except Exception as e:
        print(f"Error processing '{filepath}' with 'multi_row':\n{traceback.format_exc()}")
        
    return None, False

# --- Data Processing ---
bar_data_list = []
x_labels = []

# A dictionary to map strategy names to functions
strategies = {
    'multi_row': multi_row,
    'multi_column': multi_column
}

try:
    strategy_func = strategies[PLOTTING_CONFIG['format_strategy']]
except KeyError:
    print(f"Error: Invalid 'format_strategy' in PLOTTING_CONFIG. Choose from {list(strategies.keys())}")
    exit()

print(f"Starting data processing using '{PLOTTING_CONFIG['format_strategy']}' strategy...")

for job in PLOTTING_CONFIG['bars']:
    print(f"Processing '{job['filepath']}'...")
    
    sections, success = strategy_func(job['filepath'])
    
    if success:
        bar_data_list.append(sections)
        x_labels.append(job['x_label'])
    else:
        print(f"--- Skipping job for '{job['filepath']}' due to error. ---")

if not bar_data_list:
    print("No data processed successfully. Exiting.")
    exit()

# --- Dynamic Plotting ---

# df_plot now contains ABSOLUTE values
df_plot = pd.DataFrame(bar_data_list).fillna(0)
df_plot.index = x_labels

# Calculate totals for each bar (row)
totals = df_plot.sum(axis=1)

# Calculate proportions just for text labels
df_proportions = df_plot.div(totals, axis=0).fillna(0)

all_sections = df_plot.columns.tolist()
ind = np.arange(len(x_labels))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(12, 7))
bottom_tracker = np.zeros(len(x_labels))

# Plot each section dynamically
for section_label in all_sections:
    absolute_values = df_plot[section_label].values
    proportion_values = df_proportions[section_label].values
    
    ax.bar(
        ind, 
        absolute_values, 
        width=bar_width, 
        bottom=bottom_tracker, 
        label=section_label
    )
    
    # --- Add Percentage Labels Inside Bars ---
    # Calculate the vertical center for each text label
    y_coords = bottom_tracker + (absolute_values / 2)
    
    for i in range(len(ind)):
        # Only add text if the proportion is large enough to see
        if proportion_values[i] > 0.05: # 5% threshold
            ax.text(
                ind[i], 
                y_coords[i], 
                f'{proportion_values[i]:.0%}', 
                ha='center', 
                va='center', 
                color='white', 
                fontsize=9,
                fontweight='bold'
            )
    
    # Add the current values to the bottom tracker for the next stack
    bottom_tracker += absolute_values

print("Plot generated successfully.")

# --- !! START FIX !! ---
# --- Add Total Labels on Top of Bars ---
# Use ax.annotate for offsets and .iloc to fix warning
for i in range(len(ind)):
    total_val = totals.iloc[i] # Use .iloc to fix FutureWarning
    
    ax.annotate(
        text=f'{total_val:.0f}',     # Text to display
        xy=(ind[i], total_val),       # (x,y) point to annotate
        xytext=(0, 3),             # Offset (0 points x, 3 points y)
        textcoords='offset points',  # Use 'offset points' for xytext
        ha='center', 
        va='bottom',
        fontsize=9
    )
# --- !! END FIX !! ---

# --- Chart Customization ---
ax.set_ylabel(PLOTTING_CONFIG['y_axis_label'])
ax.set_title('Comparison of communication volume by message count')
ax.set_xticks(ind)
ax.set_xticklabels(x_labels)

# Set the y-axis limit to 115% of the tallest bar to make space
# Check if totals is not empty to avoid .max() error
if not totals.empty:
    ax.set_ylim(top=totals.max() * 1.15) 

# Place legend outside the plot
ax.legend(title='Sections', bbox_to_anchor=(1.04, 1), loc='upper left')

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) 

plt.savefig("absolute_stacked_chart.png")
print("Saved chart to 'absolute_stacked_chart.png'")
# plt.show()