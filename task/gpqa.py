from gpqa.baselines.utils import call_model_with_retries, create_prompts, get_api_type, load_examples, CHAT_MODELS
from .base import BaseTask
import copy
from tqdm import tqdm
import re

LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def parse_sampled_answer(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in LETTER_TO_INDEX:
            return match.group(1)
    return None

class GPQA(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self):
        data_filename = "/data/zhangyichi/yangjingwen/Eval/gpqa/dataset/gpqa_main.csv"
        examples = load_examples(data_filename, seed=0)
        self.text_dataset, self.examples = create_prompts(examples, prompt_type='zero_shot', few_shot_n=None, model_name=None)
        
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        correct = 0
        for question_id, (output, example) in tqdm(enumerate(zip(outputs, self.examples)), total=len(self.examples)):
            answer = parse_sampled_answer(output)
            if answer is None:
                print(f"Question ID {question_id}: No valid answer found in the output.")
                continue
            else:
                is_correct = LETTER_TO_INDEX[answer] == example.correct_index
                if is_correct:
                    correct += 1
                    print(f"Right! Question {question_id}: Correct answer: {example[example.correct_index + 1]}\nChosen answer: {example[LETTER_TO_INDEX[answer] + 1]}")
                else:
                    print(f"Wrong! Question {question_id}: Correct answer: {example[example.correct_index + 1]}\nChosen answer: {example[LETTER_TO_INDEX[answer] + 1]}")
        accuracy = correct / len(self.examples)
        print(f"Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}
