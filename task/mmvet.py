import json
import re
from .base import BaseTask
from datasets import DatasetDict, load_dataset, Dataset
from utils import load_json, generate_to_dataset
from typing import Dict
import copy
import os
import random
import global_var

prompt = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


def apply_template(item: Dict):
    question = (
            prompt
            + "\n"
            + " | ".join(
                [
                    item["prompt"],
                    item["answer"]
                    .replace("<AND>", " <AND> ")
                    .replace("<OR>", " <OR> "),
                    item["prediction"],
                    "",
                ]
            )
        )
    item["eval_prompt"] = question
    return item

def extract_evaluation(item: Dict):
    response = item['response']
    try:
        content = response.split(" ")[0].strip()
        score = float(content)
        item['score'] = score
    except:
        print(f"Error parsing response: {response}")
        item['score'] = 0.0
    return item

class mmvet(BaseTask):
    
    max_new_tokens=1024
    
    def load_text_dataset(self):
        data_dir = '/home/yangjingwen/workspace/data/mm-vet'
        text_data_path = os.path.join(data_dir, 'mm-vet.json')
        text_data = load_json(text_data_path)
        self.text_dataset = [item['question'] for item in text_data.values()]
        self.answers = [item['answer'] for item in text_data.values()]
        self.images = [os.path.join(data_dir, 'images', item['imagename']) for item in text_data.values()]
        print(f"共提取了 {len(self.text_dataset)} 个问题和 {len(self.images)} 个图像")
        assert len(self.text_dataset) == len(self.images), "Text dataset and images must have the same length"

    def load_unrelated_images(self):
        categories = {'nature': [], 'color': [], 'noise': [], 'default':[]}
        self.unrelated_dataset = copy.deepcopy(categories)
        supported_extensions = {'.png', '.jpeg', '.jpg'}
        
        for filename in os.listdir(self.unrelated_image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in supported_extensions:
                continue
            for category in categories:
                if filename.startswith(f"{category}_"):
                    categories[category].append(os.path.join(self.unrelated_image_dir,filename))
                    break
        categories['default'] = self.images
        self.unrelated_images = categories
    
    def get_unrelated_dataset(self, category="default", use_cache=False):
        if category == 'default':
            return self.images
        if not use_cache or len(self.unrelated_dataset[category]) != len(self.text_dataset):
            sampled_images = random.choices(self.unrelated_images[category], k=len(self.text_dataset))
            
            self.unrelated_dataset[category] = sampled_images
        
        return self.unrelated_dataset[category]
            
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        eval_dataset = []
        for i, (prompt, output) in enumerate(zip(self.text_dataset, outputs)):
            eval_dataset.append({
                "prompt": prompt, 
                "prediction": output,
                "answer": self.answers[i]
            })
        print(f"Evaluating {len(eval_dataset)} outputs")
        eval_dataset = Dataset.from_list(eval_dataset)
        prompted_dataset = eval_dataset.map(apply_template, num_proc=20, load_from_cache_file=False)
        evaluated_dataset = generate_to_dataset(prompted_dataset, 'gpt-4o-mini', 'eval_prompt', load_from_cache_file=False)
        processed_dataset = evaluated_dataset.map(extract_evaluation)
        
        count = 0
        for k in range(len(processed_dataset)):
            print(f"The {k} output information: {processed_dataset[k]}")
            count += processed_dataset[k]['score']
                
        return {"overall score": count/len(processed_dataset)}