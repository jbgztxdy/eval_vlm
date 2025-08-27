#!/bin/bash

source env.sh
# Define models and tasks
models=("llm-llama3.1" "llm-qwen2-7b" "llm-qwen2.5-7b" "llm-vicuna13b")
tasks=("gsm8k")

# Loop over each model and task
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $model on $task..."
        
        # Replace the following line with your actual run command
        CUDA_VISIBLE_DEVICES=6 python main.py --model "$model" --task "$task" --type text --batch-size 4 --log-dir ./log_pretest/ --result-dir ./results_pretest/ --not-use-cache
        
        echo "Finished $model on $task."
        echo "--------------------------"
    done
done


# Define models and tasks
models=("internvl2-8b" "llama32-11b" "llava-1.5-13b" "llava-1.6-13b" "llava-ov-7b" "phi3-vision" "phi3.5-vision" "qwen2-vl-7b" "qwen25-vl-7b")
tasks=("gsm8k")

# Loop over each model and task
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $model on $task..."
        
        # Replace the following line with your actual run command
        CUDA_VISIBLE_DEVICES=6 python main.py --model "$model" --task "$task" --type complete --batch-size 4 --log-dir ./log_pretest/ --result-dir ./results_pretest/ --not-use-cache
        
        echo "Finished $model on $task."
        echo "--------------------------"
    done
done