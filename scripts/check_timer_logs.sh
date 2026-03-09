#!/bin/bash

# Usage: ./check_timer_logs.sh <log_file>

LOG_FILE=$1

if [ -z "$LOG_FILE" ]; then
    echo "Usage: ./check_timer_logs.sh <log_file>"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    exit 1
fi

# List of all methods decorated with @timer_decorator in:
# - lib/python/examples/fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py
# - lib/python/examples/fwdllm/trainer/forward_training/FedSgdTrainer.py
METHODS=(
    # From tc_transformer_trainer_distribute.py
    "_make_model_functional"
    "_force_cuda_memory_cleanup"
    "_setup_training_state"
    "_select_optimal_perturbations"
    "_train_one_batch"
    "_compute_batch_stat_utility"
    "_prepare_perturbation_tensors"
    "_compute_forward_jvp"
    "_accumulate_and_extract_grads"
    "_training_loop"
    "_finalize_training"
    "train_model"
    "eval_model"
    
    # From FedSgdTrainer.py
    "_check_availability"
    "_perform_training"
    "_emulate_training_delay"
    "train_with_data_id"
)

echo "Checking timer logs in: $LOG_FILE"
echo "------------------------------------------------------------"

MISSING=0
for method in "${METHODS[@]}"; do
    # Search for the specific decorator log pattern
    COUNT=$(grep -c "Runtime of $method:" "$LOG_FILE")
    
    if [ "$COUNT" -gt 0 ]; then
        printf "[OK]      %-30s (Found %d times)\n" "$method" "$COUNT"
    else
        printf "[MISSING] %-30s (NOT FOUND)\n" "$method"
        MISSING=$((MISSING + 1))
    fi
done

echo "------------------------------------------------------------"
if [ "$MISSING" -eq 0 ]; then
    echo "✅ All decorated methods are present in the logs."
else
    echo "❌ $MISSING methods are missing from the logs."
fi