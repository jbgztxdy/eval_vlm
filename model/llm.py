import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
import lmdeploy
from .base import BaseModel
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import global_var

VICUNA_TEMPLATE = "USER: {prompt} ASSISTANT: "

class LLM(BaseModel):
    
    available_name = {
        "llm-llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llm-llama3":"meta-llama/Meta-Llama-3-8B-Instruct",
        "llm-llama2":"meta-llama/Llama-2-7b-chat-hf",
        "llm-vicuna13b": "lmsys/vicuna-13b-v1.5",
        "llm-vicuna7b": "lmsys/vicuna-7b-v1.5",
        "llm-qwen2-7b": "Qwen/Qwen2-7B-Instruct",
        "llm-qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "llm-internlm2_5-7b": "internlm/internlm2_5-7b-chat",
        "llm-phi3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",
        "llm-phi3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
        "llm-gemma":"google/gemma-2b-it"
    }
    
    def init_model(self):
        model_path = self.available_name[self.model_name]
        if "internlm" in model_path.lower():
            self.pipeline = lmdeploy.pipeline(model_path)
        else:
            # self.pipeline = transformers.pipeline(
            #         "text-generation",
            #         model=model_path,
            #         model_kwargs={"torch_dtype": torch.bfloat16},
            #         trust_remote_code=True,
            #         device_map="cuda",
            #         batch_size=32
            #     )
            #######################################
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token_id = tokenizer.eos_token_id  # 补齐token设置
            tokenizer.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="eager"  # 关键：切换到eager模式
                )
            self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                    device_map="cuda",
                    batch_size=32
                )
            #######################################
        if self.pipeline.tokenizer:
            self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            self.pipeline.tokenizer.padding_side = 'left'
        # if "vicuna" in model_path.lower():
        #     self.pipeline.tokenizer.chat_template = VICUNA_TEMPLATE
        self.question_idx = 0  # 用于可视化时的计数
        self.saved_hs = []

    @torch.no_grad()    
    def generate(self, batch_prompts, images=None, **generation_kwargs):
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ] for prompt in batch_prompts]
        if 'vicuna' in self.model_name:
            messages = [VICUNA_TEMPLATE.format(prompt=message[0]['content']) for message in messages]

        if global_var.get_value('vis') is not None:
            attn_type = global_var.get_value('vis')
            outputs = self.pipeline(messages, **generation_kwargs, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)
            attentions = outputs["attentions"]
            hidden_states = outputs["hidden_states"]
            # for i, attn in enumerate(attentions):
            #     for j, attn_layer in enumerate(attn):
            #         print(f"forward {i}, layor {j} attention shape: {attn_layer.shape}")
            attentions = attentions[0]  #第一次forward中每层注意力为(batch,num_head,seq,seq)之后均为(batch,num_head,1,seq)（seq逐次加1）
            hidden_states = hidden_states[0]  #第一次forward中每层隐藏状态为(batch,seq,hidden_size)之后均为(batch,1,hidden_size)
            del outputs["attentions"]
            del outputs["hidden_states"]
            torch.cuda.empty_cache() # 防止内存爆炸
            output_dir = os.path.join("attn_map", self.model_name, global_var.get_value('task_name'), global_var.get_value('image_type'), attn_type)
            os.makedirs(output_dir, exist_ok=True)
            for batch_idx in range(attentions[0].shape[0]):
                self.question_idx += 1
                output_path = os.path.join(output_dir, f"{attn_type}_{self.question_idx}.png")
                if attn_type == 'all':
                    cols = 4
                    num_layers = len(attentions)
                    rows = (num_layers + cols - 1) // cols
                    fig, axs = plt.subplots(rows, cols, figsize=(2*4, 2*rows))
                    axs = axs.ravel()
                    for i, attn in enumerate(attentions):
                        attn = attn[batch_idx] # 取batch_idx的注意力
                        attn_map = attn.mean(dim=0) # (seq_len, seq_len)
                        attn_map = attn_map.cpu().to(torch.float32).numpy()
                        # print(f"attn_map shape: {attn_map.shape}")
                        # print(f"attn_map min: {attn_map.min()}, max: {attn_map.max()}")
                        # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8) # 归一化

                        # 只可视化注意力热图，不叠加原始图像
                        axs[i].imshow(attn_map, cmap='jet')  # 去掉 alpha，避免透明叠加
                        axs[i].set_title(f'Layer {i+1}')
                        axs[i].set_xlabel('Attended Position')  # 列：被关注的位置
                        axs[i].set_ylabel('Query Position')     # 行：当前关注的位置
                        axs[i].axis('on')  # 显示坐标轴，方便观察位置对应关系

                    for j in range(i+1, len(axs)):
                        axs[j].axis('off')

                    # plt.colorbar(im, ax=axs)    
                    plt.tight_layout()
                    plt.show()
                
                    # 保存图像
                    plt.savefig(output_path)
                    plt.close(fig)

                elif attn_type == 'mh_all': # 打印每个头的注意力图
                    for head_index in range(attentions[0].shape[1]):  # 遍历每个头
                        output_path = os.path.join(output_dir, f"{attn_type}_{self.question_idx}_head_{head_index}.png")
                        cols = 4
                        num_layers = len(attentions)
                        rows = (num_layers + cols - 1) // cols
                        fig, axs = plt.subplots(rows, cols, figsize=(2*4, 2*rows))
                        axs = axs.ravel()
                        for i, attn in enumerate(attentions):
                            attn = attn[batch_idx] # 取batch_idx的注意力
                            attn_map = attn[head_index] # (seq_len, seq_len)
                            attn_map = attn_map.cpu().to(torch.float32).numpy()
                            # print(f"attn_map shape: {attn_map.shape}")
                            # print(f"attn_map min: {attn_map.min()}, max: {attn_map.max()}")
                            # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8) # 归一化

                            # 只可视化注意力热图，不叠加原始图像
                            axs[i].imshow(attn_map, cmap='jet')  # 去掉 alpha，避免透明叠加
                            axs[i].set_title(f'Layer {i+1}')
                            axs[i].set_xlabel('Attended Position')  # 列：被关注的位置
                            axs[i].set_ylabel('Query Position')     # 行：当前关注的位置
                            axs[i].axis('on')  # 显示坐标轴，方便观察位置对应关系

                        for j in range(i+1, len(axs)):
                            axs[j].axis('off')

                        # plt.colorbar(im, ax=axs)    
                        plt.tight_layout()
                        plt.show()

                        # 保存图像
                        plt.savefig(output_path)
                        plt.close(fig)

                elif attn_type == 'save_hs':
                    local_hs = [hs[batch_idx][-1].cpu().float().numpy() for hs in hidden_states]
                    combined_np = np.stack(local_hs, axis=0)
                    self.saved_hs.append(combined_np)


            outputs = outputs["real_outputs"]
        else:
            outputs = self.pipeline(messages, **generation_kwargs)

        if 'vicuna' in self.model_name:
            generated_text = [item[0]["generated_text"].split('ASSISTANT:')[-1].strip() for item in outputs]
        elif 'internlm' in self.model_name:
            generated_text = [item.text for item in outputs]
        else:
            generated_text = [item[0]["generated_text"][-1]['content'] for item in outputs]
        return generated_text