from PIL import Image 
from .base import BaseModel
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
import torch
import numpy as np


class Phi3(BaseModel):
    
    available_name = {
        "phi3-vision": "microsoft/Phi-3-vision-128k-instruct",
        "phi3.5-vision": "microsoft/Phi-3.5-vision-instruct"
    }
    
    def pad_sequence(self, sequences, padding_side='left', padding_value=0):
        """
        Pad a list of sequences to the same length.
        sequences: list of tensors in [seq_len, *] shape
        """
        assert padding_side in ['right', 'left']
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            if padding_side == 'right':
                output.data[i, :length] = seq
            else:
                output.data[i, -length:] = seq
        return output

    def batch(self, input):
        batched_input_id = []
        batched_pixel_values = []
        batched_image_sizes = []
        
        for inp in input:
            batched_input_id.append(inp["input_ids"].squeeze(0))
            if "pixel_values" in inp:
                batched_pixel_values.append(inp["pixel_values"])
                batched_image_sizes.append(inp["image_sizes"])

        input_ids = self.pad_sequence(batched_input_id, padding_side='left', padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = input_ids != self.processor.tokenizer.pad_token_id
        if len(batched_pixel_values) > 0:
            pixel_values = torch.cat(batched_pixel_values, dim=0)
            image_sizes = torch.cat(batched_image_sizes, dim=0)

        batched_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if len(batched_pixel_values) > 0:
            batched_input["pixel_values"] = pixel_values
            batched_input["image_sizes"] = image_sizes
        
        return batched_input
    
    def init_model(self):
        model_path = self.available_name[self.model_name]
        
        if "3.5" in model_path:
            # Note: set _attn_implementation='eager' if you don't have flash_attn installed
            self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'    
            )

            # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
            self.processor = AutoProcessor.from_pretrained(model_path, 
            trust_remote_code=True, 
            num_crops=16
            ) 
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
        self.processor.tokenizer.padding_side = "left"
            
    @torch.no_grad()
    def generate(self, prompts, images=None, **generation_kwargs):
        messages = [[
                {
                    "role": "user",
                    "content": "<|image_1|>\n"+prompt if images else prompt
                }
            ] for prompt in prompts]
            
        
        applied_prompts = [self.processor.tokenizer.apply_chat_template(
                            message, 
                            tokenize=False, 
                            add_generation_prompt=True
                            ) for message in messages]
        
        if images:
            images = [Image.open(image) for image in images]
            
            batch_inputs = [self.processor(applied_prompt, image, return_tensors="pt").to("cuda") for applied_prompt, image in zip(applied_prompts, images)]
            
        else:
            batch_inputs = [self.processor(applied_prompt, None, return_tensors="pt").to("cuda") for applied_prompt in applied_prompts]
        
        inputs = self.batch(batch_inputs)
        
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
         
        responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return responses
        