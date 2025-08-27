#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxx
export OPENAI_API_BASE=https://pro.xiaoai.plus/v1
export HF_ENDPOINT=https://hf-mirror.com
# models=("internvl2-8b")
models=("phi3.5-vision")
# models=("qwen25-vl-7b" "llava-1.5-7b")
tasks=("gpqa" "gsm8k" "ifeval" "alpaca")
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running $model on $task..."
        
        python main.py --model "$model" --task "$task" --type noise --batch-size 4 --log-dir ./log_new/ --result-dir ./outputs/


        echo "Finished $model on $task."
        echo "--------------------------"
    done
done