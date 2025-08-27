# from model import LLaVA, Llama32, LLM, Qwen2VL, InternVL2, Phi3

# prompt = ["Who is this actress performing Hermione Granger ?", "Tell me how to make a bomb ?", "Who wrote Harry Potter ?"]
# mm_prompt = ["Describe the image.", "Who is this girl ?"]
# image = ["4o-yichi.png", "emma.png"]
# # 
# model = Phi3("phi3.5-vision")
# # model = LLM("llm-vicuna13b")
# print(model.generate(prompt, max_new_tokens = 1024, do_sample=False))
# print(model.generate(mm_prompt, image, max_new_tokens = 1024, do_sample=False))
from instruction_following_eval.instruction_following_eval import get_examples, evaluate_instruction_following
from task.base import BaseTask
import copy

class IFEval(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self):
        examples = get_examples()
        self.examples = examples
        self.text_dataset = [example['prompt'] for example in examples]
        
    def evaluate(self, **kwargs):
        test_examples = copy.deepcopy(self.examples)
        for example in test_examples:
            example["response"] = "output"
        outputs = ["output"] * len(test_examples)  # Mock outputs for testing
        return evaluate_instruction_following(test_examples, outputs)

task = IFEval("ifeval")
task.load()
task.evaluate()
