#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=4
export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export OPENAI_API_BASE=https://pro.xiaoai.plus/v1
export HF_ENDPOINT=https://hf-mirror.com
export HUGGING_FACE_HUB_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxx
# echo "Running llava-1.5-7b on mmsafetybench_sd using default images..."``
# python -u main.py --model llava-1.5-7b --task mmsafetybench_sd_image_late_half --type default --batch-size 4 --log-dir ./log_infer/ --result-dir ./results_infer/ --not-use-cache
# echo "Finished llava-1.5-7b on mmsafetybench_sd using default images."
# echo "--------------------------"

echo "Running llm-vicuna7b on alpaca using default images..."
python -u main.py --model llm-gemma --task mmsafetybench_local_text --type text --vis save_hs --batch-size 4 --log-dir ./log_infer/ --result-dir ./results_infer/ --not-use-cache
# python -u main.py --model llava-1.5-7b --task ifeval --type nature --vis self_attn_ratio --batch-size 4 --log-dir ./log_infer/ --result-dir ./results_infer/ --not-use-cache
echo "Finished llm-vicuna7b on alpaca using default images."
echo "--------------------------"

# models=("llava-1.5-7b")
# tasks=("spa-vl-harm")
# strategies=("nearest" "random")
# for model in "${models[@]}"; do
#     for task in "${tasks[@]}"; do
#         echo "Running $model on $task without strategy..."
#         python main.py --model "$model" --task "$task" --type default --vis image --batch-size 4 --log-dir ./log_check/ --result-dir ./outputs_check/ --not-use-cache
#         echo "Finished $model on $task without strategy."
#         echo "--------------------------"
#         for strategy in "${strategies[@]}"; do
#             echo "Running $model on $task with strategy $strategy..."
            
#             python main.py --model "$model" --task "$task" --test-strategy "$strategy" --type default --vis image --batch-size 4 --log-dir ./log_check/ --result-dir ./outputs_check/ --not-use-cache
            
#             echo "Finished $model on $task with strategy $strategy."
#             echo "--------------------------"
#         done
#     done
# done