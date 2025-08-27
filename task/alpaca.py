from .base import BaseTask
import datasets
from alpaca_eval.main import evaluate


class AlpacaEval(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self, *args, **kwargs):
        dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
        self.text_dataset = [example["instruction"] for example in dataset]
        
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        generator = kwargs.get("generator")
        
        model_outputs = []
        for inst, output in zip(self.text_dataset, outputs):
            model_outputs.append({
                "instruction": inst,
                "output": output,
                "generator": generator
            })
        
        leaderboard, annotations = evaluate(model_outputs, is_return_instead_of_print=True, precomputed_leaderboard=None, is_overwrite_leaderboard=True)
        # Convert the single row to JSON (as a dict)
        json_data = leaderboard.iloc[0].to_dict()
        return json_data
        