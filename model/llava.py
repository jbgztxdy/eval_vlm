from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration
from .base import BaseModel
import global_var

class LLaVA(BaseModel):
    
    available_name = {
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
        "llava-1.6-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-1.6-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    }
    
    def init_model(self):
        model_path = self.available_name[self.model_name]
        if '1.6' in self.model_name:
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
        elif '1.5' in self.model_name:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
                    attn_implementation="eager", # 为提取注意力而添加
                    low_cpu_mem_usage=True, 
                ).to("cuda")
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to("cuda")

            self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(self.processor.tokenizer.padding_side)
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.question_idx = 0
        self.saved_hs = []
            
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
            inputs = self.processor(images=raw_images, text=text_prompts, return_tensors='pt', padding=True).to("cuda")
        else:
            conversations = [[
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    ],
                }
            ] for prompt in prompts ]
            text_prompts = [self.processor.apply_chat_template(conversation, add_generation_prompt=True) for conversation in conversations]
            
            inputs = self.processor(text=text_prompts, return_tensors='pt', padding=True).to("cuda")
        return inputs
            
    @torch.no_grad()
    def generate(self, prompts, images=None, **generation_kwargs):
        inputs = self.prepare_input(prompts, images)
        # if images is not None:
        #     img_positions = (inputs.input_ids[0]==32000).nonzero(as_tuple=True)[0]
        #     start_pos = img_positions[0]
        #     end_pos = img_positions[-1]
        #     img_len = img_positions.shape[0]
        # else:
        #     start_pos, end_pos, img_len = None, None, None
        input_len = inputs.input_ids.shape[1]
        if global_var.get_value('vis') is not None:
            attn_type = global_var.get_value('vis')
            outputs = self.model.generate(**inputs, **generation_kwargs, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            # for i, attn in enumerate(attentions):
            #     for j, attn_layer in enumerate(attn):
            #         print(f"forward {i}, layor {j} attention shape: {attn_layer.shape}")
            attentions = attentions[0]  #第一次forward中每层注意力为(batch,num_head,seq,seq)之后均为(batch,num_head,1,seq)（seq逐次加1）
            hidden_states = hidden_states[0]  #第一次forward中每层隐藏状态为(batch,seq,hidden_size)之后均为(batch,1,hidden_size)
            del outputs.attentions
            del outputs.hidden_states
            torch.cuda.empty_cache() # 防止内存爆炸
            output_dir = os.path.join("attn_map", self.model_name, global_var.get_value('task_name'), global_var.get_value('image_type'), attn_type)
            os.makedirs(output_dir, exist_ok=True)
            for batch_idx in range(attentions[0].shape[0]):
                self.question_idx += 1
                output_path = os.path.join(output_dir, f"{attn_type}_{self.question_idx}.png")
                if attn_type == 'image':
                    img = inputs.pixel_values[batch_idx].squeeze().permute(1, 2, 0).cpu().numpy() # (336, 336, 3)
                    img = (img - img.min()) / (img.max() - img.min())
                    num_layers = len(attentions)
                    cols = 4
                    rows = (num_layers + cols - 1) // cols
                    fig, axs = plt.subplots(rows, cols, figsize=(2*4, 2*rows))
                    axs = axs.ravel()
                    for i, attn in enumerate(attentions):
                        # attn: [batch_size, num_heads, seq_len, seq_len]
                        # [cls] atten
                        # attn_map = attn.mean(dim=1)[0, 0, 1:].reshape(-1, 1)  # (576, 1)
                        # img atten
                        attn = attn[batch_idx].unsqueeze(0) # 取batch_idx的注意力
                        attn_map = attn.mean(dim=1)[0, -1].reshape(-1, 1) # (seq_len, 1) 取最后一个token关于其他的attention
                        # 转换为2D注意力图
                        config = self.model.config.vision_config
                        patch_size = config.patch_size # 14
                        image_size = config.image_size # 336
                        num_patches = (image_size // patch_size) ** 2 # 576
                        image_mask = inputs.input_ids[batch_idx] == self.model.config.image_token_id
                        attn_map = attn_map[image_mask].reshape(1, 1, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
                        # 上采样到原图尺寸
                        attn_map = F.interpolate(
                            attn_map.to("cpu"),  
                            scale_factor=patch_size,
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()
                        
                        attn_map = attn_map.cpu().numpy()
                        # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8) # 归一化

                        axs[i].imshow(img)
                        axs[i].imshow(attn_map, cmap='jet', alpha=0.5)
                        axs[i].set_title(f'Layer {i+1}')
                        axs[i].axis('off')

                    for j in range(i+1, len(axs)):
                        axs[j].axis('off')

                    # plt.colorbar(im, ax=axs)    
                    plt.tight_layout()
                    plt.show()

                    # 保存图像
                    plt.savefig(output_path)
                    plt.close(fig)

                elif attn_type == 'all':
                    cols = 4
                    num_layers = len(attentions)
                    rows = (num_layers + cols - 1) // cols
                    fig, axs = plt.subplots(rows, cols, figsize=(2*4, 2*rows))
                    axs = axs.ravel()
                    for i, attn in enumerate(attentions):
                        attn = attn[batch_idx] # 取batch_idx的注意力
                        attn_map = attn.mean(dim=0) # (seq_len, seq_len)
                        attn_map = attn_map.cpu().numpy()
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

                elif attn_type == 'text_using_image':
                    cols = 4
                    num_layers = len(attentions)
                    rows = (num_layers + cols - 1) // cols
                    fig, axs = plt.subplots(rows, cols, figsize=(2*4, 2*rows))
                    axs = axs.ravel()
                    for i, attn in enumerate(attentions):
                        attn = attn[batch_idx] # 取batch_idx的注意力
                        attn_map = attn.mean(dim=0) # (seq_len, seq_len)
                        config = self.model.config.vision_config
                        patch_size = config.patch_size # 14
                        image_size = config.image_size # 336
                        num_patches = (image_size // patch_size) ** 2 # 576
                        image_mask = inputs.input_ids[batch_idx] == self.model.config.image_token_id
                        text_mask = ~image_mask  # 使用 ~ 运算符翻转布尔值
                        text_indices = torch.nonzero(text_mask, as_tuple=True)[0]  # 获取一维索引
                        attn_map = attn_map[text_indices, :][:, text_indices]  # 取出文本token对应的子矩阵
                        attn_map = attn_map.cpu().numpy()
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
                            attn_map = attn_map.cpu().numpy()
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

                elif attn_type == 'vt_attn_ratio':
                    config = self.model.config.vision_config
                    patch_size = config.patch_size  # 14
                    image_size = config.image_size  # 336
                    num_patches = (image_size // patch_size) ** 2  # 576
                    image_mask = inputs.input_ids[batch_idx] == self.model.config.image_token_id
                    image_token_count = image_mask.sum().item()  # 图像token数量

                    layer_ratios = []  # 存储各层的图像注意力占比
                    for i, attn in enumerate(attentions):
                        attn = attn[batch_idx]  # 取当前batch的注意力
                        attn_map = attn.mean(dim=0)  # 平均所有头的注意力，形状为(seq_len, seq_len)
                        attn_map = attn_map[-1] # 取最后一个token的注意力，形状为(seq_len,)

                        # 提取图像部分的注意力（行或列涉及图像token的部分）
                        image_attn = attn_map[image_mask]  # 行是图像token时的注意力
                        total_attn = attn_map.sum()  # 该层总注意力

                        if total_attn.item() != 0:
                            ratio = image_attn.sum().item() / total_attn.item()  # 图像注意力占比
                        else:
                            ratio = 0.0  # 总注意力为0时，占比设为0
                        layer_ratios.append(ratio)

                    # 绘制柱状图
                    layers = list(range(0, len(attentions)))
                    plt.figure(figsize=(10, 6))
                    plt.bar(layers, layer_ratios, color='skyblue')
                    plt.xlabel('Layer')
                    plt.ylabel('Image Attention Ratio')
                    plt.title('Image Attention Ratio Across Layers')
                    plt.xticks(layers)
                    plt.tight_layout()

                    # 保存图像
                    output_path = os.path.join(output_dir, f"{attn_type}_{self.question_idx}.png")
                    plt.savefig(output_path)
                    plt.close()
                
                elif attn_type == 'self_attn_ratio':
                    layer_ratios = []  # 存储各层的图像注意力占比
                    for i, attn in enumerate(attentions):
                        attn = attn[batch_idx]  # 取当前batch的注意力
                        attn_map = attn.mean(dim=0)  # 平均所有头的注意力，形状为(seq_len, seq_len)
                        attn_map = attn_map[-1] # 取最后一个token的注意力，形状为(seq_len,)

                        self_attn = attn_map[-1]
                        total_attn = attn_map.sum()  # 该层总注意力

                        if total_attn.item() != 0:
                            ratio = self_attn.sum().item() / total_attn.item()  # 注意力占比
                        else:
                            ratio = 0.0  # 总注意力为0时，占比设为0
                        layer_ratios.append(ratio)

                    # 绘制柱状图
                    layers = list(range(0, len(attentions)))
                    plt.figure(figsize=(10, 6))
                    plt.bar(layers, layer_ratios, color='skyblue')
                    plt.xlabel('Layer')
                    plt.ylabel('Self Attention Ratio')
                    plt.title('Self Attention Ratio Across Layers')
                    plt.xticks(layers)
                    plt.tight_layout()

                    # 保存图像
                    output_path = os.path.join(output_dir, f"{attn_type}_{self.question_idx}.png")
                    plt.savefig(output_path)
                    plt.close()

                elif attn_type == 'save_hs':
                    local_hs = [hs[batch_idx][-1].cpu().numpy() for hs in hidden_states]
                    combined_np = np.stack(local_hs, axis=0)
                    self.saved_hs.append(combined_np)


            outputs = outputs.sequences
        else:
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
            img_positions = (inputs.input_ids[0]==32000).nonzero(as_tuple=True)[0]
            start_pos = img_positions[0]
            end_pos = img_positions[-1]
            img_len = img_positions.shape[0]
        else:
            start_pos, end_pos, img_len = None, None, None
        input_len = inputs.input_ids.shape[1]
        # 保存原始配置
        original_config = self.model.config.use_flash_attention_2
        # 需要输出注意力时，临时禁用 SDPA
        self.model.config.use_flash_attention_2 = False
        output = self.model(**inputs, output_attentions=True)
        print(output.image_hidden_states.shape)
        print(img_len)
        print(output.attentions[0].shape)
        print(f"attention: \n{output.attentions}")
        # 恢复原始配置，继续使用 SDPA 加速
        self.model.config.use_flash_attention_2 = original_config