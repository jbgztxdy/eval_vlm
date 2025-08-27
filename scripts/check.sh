#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxx
export OPENAI_API_BASE=https://pro.xiaoai.plus/v1
models=("qwen25-vl-7b" "llava-1.5-13b")
tasks=("spa-vl-harm")
strategies=("nearest" "random")
# python -u main.py --model llava-1.5-7b --task gpqa --test-strategy random --type noise --batch-size 4 --log-dir ./log_new/ --result-dir ./outputs/ --not-use-cache
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # echo "Running $model on $task without images..."
        # python main.py --model "$model" --task "$task" --type text --batch-size 4 --log-dir ./log_check/ --result-dir ./outputs_check/ --not-use-cache
        # echo "Finished $model on $task without images."
        # echo "--------------------------"
        echo "Running $model on $task without strategy..."
        python main.py --model "$model" --task "$task" --type default --batch-size 4 --log-dir ./log_check/ --result-dir ./outputs_check/ --not-use-cache
        echo "Finished $model on $task without strategy."
        echo "--------------------------"
        for strategy in "${strategies[@]}"; do
            echo "Running $model on $task with strategy $strategy..."
            
            python main.py --model "$model" --task "$task" --test-strategy "$strategy" --type default --batch-size 4 --log-dir ./log_check/ --result-dir ./outputs_check/ --not-use-cache
            
            echo "Finished $model on $task with strategy $strategy."
            echo "--------------------------"
        done
    done
done