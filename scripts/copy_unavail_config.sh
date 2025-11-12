#!/bin/bash

###
# USER WARNING: trainer_0 may not exist in src directory!
###

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SRC_DIR="/home/dgarg39/aish_test/flame/lib/python/examples/async_cifar10/trainer/config_dir0.1_num300_traceFail_6d_3state_oort"
DST_DIR="/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts"
# We can re-use the same temporary file name
# TMP_FILE="$DST_DIR/tmp_config.json"

# The Python helper script we just created
PYTHON_SCRIPT="/home/dgarg39/aish_test/flame/scripts/copy_unavail_config_.py" 
# ---------------------

# --- Check for Python ---
# Use python3, as 'python' can sometimes be the old Python 2
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed or not in PATH."
    echo "This script requires python3 as an alternative to jq."
    exit 1
fi

# --- Check for helper script ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: The helper script '$PYTHON_SCRIPT' was not found."
    echo "Please place it in the same directory as this bash script."
    exit 1
fi

# --- Check if directories exist ---
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory not found at $SRC_DIR"
    exit 1
fi

if [ ! -d "$DST_DIR" ]; then
    echo "Error: Destination directory not found at $DST_DIR"
    exit 1
fi

echo "Starting batch key copy using Python..."

# --- Loop through all matching files ---
for src_file in "$SRC_DIR"/trainer_*.json; do

    # Get just the filename (e.g., "trainer_1.json")
    filename=$(basename "$src_file")

    # This handles the case where no files match the glob
    if [ ! -f "$src_file" ]; then
        echo "No files matching 'trainer_*.json' found in $SRC_DIR."
        break # Exit the loop
    fi

    # Define the path to the matching destination file
    dst_file="$DST_DIR/$filename"

    # Check if the matching destination file actually exists
    if [ ! -f "$dst_file" ]; then
        echo "Warning: Skipping. No matching file for $filename found in $DST_DIR."
        continue # Skip to the next file in the loop
    fi

    echo "Processing: $filename"

    # --- Call the Python script ---
    # Give it the source file and destination file as arguments
    python3 "$PYTHON_SCRIPT" "$src_file" "$dst_file"

done

echo "Batch processing complete."
