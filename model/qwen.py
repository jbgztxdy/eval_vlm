from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
import transformers
from transformers import logging
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from .base import BaseModel

class Visualization_model(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_modules():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def __getattr__(self, name: str):
        # 先检查属性是否属于当前类（避免递归）
        if name in ["model", "forward", "__getattr__", "__call__"]:
            return super().__getattr__(name)
        # 再检查是否属于原始模型
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        """转发调用行为到原始模型（确保模型调用与原始一致）"""
        return self.model(*args, **kwargs)
    

class Qwen2VL(BaseModel):
    
    available_name = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen25-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct"
    }
    
    def init_model(self):
        model_path = self.available_name[self.model_name]
        if "5" not in model_path:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_path, torch_dtype="auto"
                        ).to("cuda")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype="auto"
                    ).to("cuda")
            # self.model = Visualization_model(self.model).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        print(self.processor.tokenizer.padding_side)
        self.processor.tokenizer.padding_side = "left"
            
    def prepare_input(self, prompts, images=None):
        if images is not None:
            conversations = [[
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                    ],
                }
            ] for prompt in prompts ]
            text_prompts = [self.processor.apply_chat_template(conversation, add_generation_prompt=True) for conversation in conversations]
            
            if isinstance(images[0], str):
                raw_images = [Image.open(image) for image in images]
            elif isinstance(images[0], Image.Image):
                raw_images = images
            inputs = self.processor(images=raw_images, text=text_prompts, padding=True, return_tensors='pt').to(self.model.device)
        else:
            conversations = [[
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    ],
                }
            ] for prompt in prompts ]
            print(conversations)
            text_prompts = [self.processor.apply_chat_template(conversation, add_generation_prompt=True) for conversation in conversations]
            
            inputs = self.processor(text=text_prompts, padding=True, return_tensors='pt').to(self.model.device)
        return inputs
            
    @torch.no_grad()
    def generate(self, prompts, images=None, **generation_kwargs):
        inputs = self.prepare_input(prompts, images)
        # if images is not None:
        #     img_positions = (inputs.input_ids[0]==151655).nonzero(as_tuple=True)[0]
        #     start_pos = img_positions[0]
        #     end_pos = img_positions[-1]
        #     img_len = img_positions.shape[0]
        # else:
        #     start_pos, end_pos, img_len = None, None, None
        input_len = inputs.input_ids.shape[1]
        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[input_len :] for out_ids in outputs
        ]
        generation = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return generation
    
    @torch.no_grad()
    def forward(self, prompt, image=None):
        inputs = self.prepare_input(prompt, image)
        print(inputs.keys())
        if image is not None:
            img_positions = (inputs.input_ids[0]==151655).nonzero(as_tuple=True)[0]
            start_pos = img_positions[0]
            end_pos = img_positions[-1]
            img_len = img_positions.shape[0]
        else:
            start_pos, end_pos, img_len = None, None, None
        input_len = inputs.input_ids.shape[1]
        print(input_len)
        output = self.model(**inputs, output_attentions=True)
        print(output.keys())
        print(f"attention: \n{output.attentions}")