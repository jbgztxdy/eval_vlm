#!/bin/bash

source env.sh
# Define models and tasks
models=("phi3-vision" "phi3.5-vision")
tasks=("ifeval" "pii-leakage" "alpaca_eval")

# Loop over each model and task
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $model on $task..."
        
        # Replace the following line with your actual run command
        CUDA_VISIBLE_DEVICES=7 python main.py --model "$model" --task "$task" --type complete --batch-size 8 --log-dir ./log_pretest/ --result-dir ./results_pretest/
        
        echo "Finished $model on $task."
        echo "--------------------------"
    done
done