import numpy as np
from model import get_model
from task import get_task
import argparse
import os
from tqdm import tqdm
from utils import generate_log, write_json, load_log
import global_var

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for model task")

    parser.add_argument('--model', type=str, required=True, help='Name or path of the model')
    parser.add_argument('--task', type=str, required=True, help='Task to perform')
    
    allowed_types = ['text', 'nature', 'noise', 'color', 'complete', 'default', 'harm']
    parser.add_argument('--type', type=str, choices=allowed_types, required=True, help=f"Type of data combination. Must be one of: {', '.join(allowed_types)}")
    parser.add_argument('--run', type=int, default=1, help='Run ID (an integer, e.g., 1, 2, 3)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch Size for Generation')
    
    parser.add_argument('--not-use-cache', default=False, action="store_true")
    parser.add_argument('--vis', default=None, help='Visualize the attention map')
    
    parser.add_argument('--log-dir', type=str, default='./log/', help='Task to perform')
    parser.add_argument('--result-dir', type=str, default='./log/', help='Task to perform')
    parser.add_argument('--test-strategy', type=str, default=None, help='Strategy to handle image features')

    args = parser.parse_args()
    return args


def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def pipeline(args):
    global_var.set_value('vis', args.vis)
    model = get_model(args.model)
    global_var.set_value('model_name', args.model)
    task = get_task(args.task)
    global_var.set_value('task_name', args.task)
    log_dir = os.path.join(args.log_dir, model.model_name, task.task_name)
    os.makedirs(log_dir, exist_ok=True)
    task.load()
    prompts = task.get_text_dataset()
    
    results = {}
    
    if args.type in ['text', 'complete']:
        global_var.set_value('image_type', args.type)
        batched_prompts = batchify(prompts, args.batch_size)
        outputs = []
        log_path = os.path.join(log_dir, "text.json")
        if not args.not_use_cache and  os.path.exists(log_path):
            cached_prompts, cached_outputs = load_log(log_path)
            if len(cached_prompts) == len(prompts):
                print("Use Cached Results")
                results['text'] = task.evaluate(outputs=cached_outputs, generator=f"{args.model}_text")
        
        if 'text' not in results:
            for batch in tqdm(batched_prompts):
                outputs.extend(model.generate(batch, do_sample=False, max_new_tokens=task.max_new_tokens))
                # print(f"Generated {len(outputs)} outputs, forwarding……")
                # model.forward(prompt=batch, image=None)
            write_json(generate_log(prompts, outputs), log_path)

            if args.vis == 'save_hs':
                hs_save_path = os.path.join('hidden_states', f'{model.model_name}-{task.task_name}-{global_var.get_value("image_type")}.npy')
                hs = np.stack(model.saved_hs, axis=0)
                print(f"save_hs shape: {hs.shape}")
                np.save(hs_save_path, hs)

            results['text'] = task.evaluate(outputs=outputs, generator=f"{args.model}_text")
            
            
        
    if args.type != 'text':
        if args.type == 'complete':
            categories = ['nature', 'color', 'noise']
        else:
            categories = [args.type]
            
        for category in categories:
            if args.test_strategy is None:
                global_var.set_value('image_type', category)
            else:
                global_var.set_value('image_type', f"{category}_{args.test_strategy}")
            for i in range(args.run):
                batched_prompts = batchify(prompts, args.batch_size)
                images = task.get_unrelated_dataset(category)
                batched_images = batchify(images, args.batch_size)
                outputs = []
                
                if args.test_strategy is None:
                    log_path = os.path.join(log_dir, f"{category}_run{i}.json")
                else:
                    log_path = os.path.join(log_dir, f"{category}_{args.test_strategy}_run{i}.json")
                if not args.not_use_cache and os.path.exists(log_path):
                    cached_prompts, cached_outputs = load_log(log_path)
                    if len(cached_prompts) == len(prompts):
                        print("Use Cached Results")
                        results['{}_{}'.format(category, i)] = task.evaluate(outputs=cached_outputs, generator=f"{args.model}_{category}_{i}")
                if '{}_{}'.format(category, i) not in results:
                    for batch_text, batch_image in tqdm(list(zip(batched_prompts, batched_images))):
                        if args.test_strategy is None:
                            outputs.extend(model.generate(batch_text, batch_image, do_sample=False, max_new_tokens=task.max_new_tokens))
                        else:
                            outputs.extend(model.generate(batch_text, batch_image, do_sample=False, max_new_tokens=task.max_new_tokens, test_strategy=args.test_strategy))
                    
                    write_json(generate_log(prompts, outputs), log_path)
                    
                    if args.vis == 'save_hs':
                        hs_save_path = os.path.join('hidden_states', f'{model.model_name}-{task.task_name}-{global_var.get_value("image_type")}.npy')
                        hs = np.stack(model.saved_hs, axis=0)
                        print(f"save_hs shape: {hs.shape}")
                        np.save(hs_save_path, hs)

                    if args.test_strategy is None:
                        results['{}_{}'.format(category, i)] = task.evaluate(outputs=outputs, generator=f"{args.model}_{category}_{i}")
                    else:
                        results['{}_{}_{}'.format(category, args.test_strategy, i)] = task.evaluate(outputs=outputs, generator=f"{args.model}_{category}_{args.test_strategy}_{i}")


    return results


if __name__=='__main__':
    global_var._init()  # Initialize global variable dictionary
    args = get_args()
    results = pipeline(args)
    print("===================================================================")
    print(f"Evaluation Result with {args.model} on {args.task}")
    print(results)
    result_dir = os.path.join(args.result_dir, args.model, args.task)
    os.makedirs(result_dir, exist_ok=True)
    if args.test_strategy is None:
        write_json(results, os.path.join(result_dir, f"{args.model}-{args.task}-{args.type}.json"))
    else:
        write_json(results, os.path.join(result_dir, f"{args.model}-{args.task}-{args.type}-{args.test_strategy}.json"))