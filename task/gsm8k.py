from .base import BaseTask
from utils import load_json, generate_to_dataset
from datasets import Dataset
from typing import Dict
import re
import json


EVAL_TEMPLATE = """
Here is a quesiton-answer pair in math along with the ground-truth label. You need to judge whether the answer is correct according to the provided label.
Since the answer can be a short sentence or paragraph, you should first extract the answer provided to the question and then compare it with the label. If the forms of the extracted answer and the label are not exactly the same, but can be derived into the same expression with trivial simplifications, it can still be taken as a correct answer.
Some answer-label pairs that can be taken as the same are presented below.
$2x+3$ = $3+2x$, 3/2 = 1.5, $x^2+2x+1$ = $y^2+2y+1$, $x^2+2x+1$ = $(x+1)^2$, 2/(-3) = (-2)/3, $\\frac{{\\sqrt{{2}}}}{{2}}$ = $\\frac{{1}}{{\\sqrt{{2}}}}$, 72 degrees = 72, 64 = 64 square feet.

Question: {prompt}

Label: {label}

Answer: {predict}
 
It is also possible that the answer does not contain an explicity answer. In this case, it should be taken as wrong. Do not try to solve the math question yourself, just compare the label and the answer. Give your conclusion at the end of your response with a new line. The format should be "Conclusion: True/False".
""".strip()


def apply_template(item: Dict):
    item["eval_prompt"] = EVAL_TEMPLATE.format(**item)
    return item

def extract_evaluation(item: Dict):
    ANSWER_PATTERN = r"(?i)Conclusion\s*:\s*([^\n]+)"
    response = item['response']
    # 检查response是否为None
    if response is None:
        print(f"Warning: response is None for item")
        item['score'] = 'false'
        return item
    match = re.search(ANSWER_PATTERN, response)
    extracted_answer = match.group(1) if match else None
    if extracted_answer is not None and "true" in extracted_answer.lower():
        item['score'] = 'true'
    else:
        item['score'] = 'false'
    return item

class GSM8k(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self):
        dataset = load_json("/data/zhangyichi/Modality-Bridging/data/gsm8k_test.json")
        self.text_dataset = [item['instruction'] for item in dataset]
        self.gt = [item['output'] for item in dataset]
        
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        eval_dataset = [{"prompt": prompt, "predict": output, "label": label} for prompt, output, label in zip(self.text_dataset, outputs, self.gt)]
        eval_dataset = Dataset.from_list(eval_dataset)
        prompted_dataset = eval_dataset.map(apply_template, num_proc=20, load_from_cache_file=False)
        evaluated_dataset = generate_to_dataset(prompted_dataset, 'gpt-4o-mini', 'eval_prompt', load_from_cache_file=False)
        processed_dataset = evaluated_dataset.map(extract_evaluation)
        
        count = 0
        for k in range(len(processed_dataset)):
            if processed_dataset[k]['score'] == 'false':
                print(processed_dataset[k])
            if processed_dataset[k]['score'] == 'true':
                count+=1
                
        with open('trained.json', 'w') as f:
            json.dump(processed_dataset.to_dict(), f)
                
        return {"accuracy": count/len(processed_dataset)}