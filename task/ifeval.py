from instruction_following_eval.instruction_following_eval import get_examples, evaluate_instruction_following
from .base import BaseTask
import copy

class IFEval(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self):
        examples = get_examples()
        self.examples = examples
        self.text_dataset = [example['prompt'] for example in examples]
        
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        test_examples = copy.deepcopy(self.examples)
        for example, output in zip(test_examples, outputs):
            example["response"] = output
        return evaluate_instruction_following(test_examples, outputs)
