import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from .base import BaseModel

class Llama32(BaseModel):
    
    available_name = {
        "llama32-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct"
    }
    
    def init_model(self):
        model_path = self.available_name[self.model_name]
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        
    def prepare_input(self, prompts, images=None):
        if images:
            messages = [[
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                    }
                ] for prompt in prompts ]
        else:
               messages = [[
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        ],
                    }
                ] for prompt in prompts ]
        
        input_texts = []
        
        for message in messages:
        
            input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
            
            if "<|start_header_id|>system<|end_header_id|>" in input_text:
                input_text = "<|begin_of_text|><|"+input_text[input_text.find("<|eot_id|>"):].lstrip("<|eot_id|>")
            
            input_texts.append(input_text)
        
        if images:
            images = [[Image.open(image)] for image in images]
        
        inputs = self.processor(
            images,
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        return inputs
    
    @torch.no_grad()
    def generate(self, prompts, images=None, **generation_kwargs):
        inputs = self.prepare_input(prompts,images)
        input_len = inputs.input_ids.shape[1]
        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[input_len :] for out_ids in outputs
        ]
        generation = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return generation