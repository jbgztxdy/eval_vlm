from abc import ABC, abstractmethod
import os
import copy
import random

class BaseTask(ABC):
    
    max_new_tokens = 200
    
    def __init__(self, name):
        self.unrelated_image_dir = "/home/yangjingwen/workspace/Eval/data/unrelated_images"
        self.task_name = name
        
    def load(self):
        self.load_text_dataset()
        self.load_unrelated_images()
        print(f"Task {self.task_name} initialized.")



    @abstractmethod
    def load_text_dataset(self, *args, **kwargs):
        self.text_dataset = None
    
    def load_unrelated_images(self):
        categories = {'nature': [], 'color': [], 'noise': []}
        self.unrelated_dataset = copy.deepcopy(categories)
        supported_extensions = {'.png', '.jpeg', '.jpg'}
        
        for filename in os.listdir(self.unrelated_image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in supported_extensions:
                continue
            for category in categories:
                if filename.startswith(f"{category}_"):
                    categories[category].append(os.path.join(self.unrelated_image_dir,filename))
                    break
            
        self.unrelated_images = categories
        
        
    def get_text_dataset(self, *args, **kwargs):
        return self.text_dataset
    
    def get_unrelated_dataset(self, category="nature", use_cache=False):
        if not use_cache or len(self.unrelated_dataset[category]) != len(self.text_dataset):
            sampled_images = random.choices(self.unrelated_images[category], k=len(self.text_dataset))
            
            self.unrelated_dataset[category] = sampled_images
        
        return self.unrelated_dataset[category]
        
    @abstractmethod
    def evaluate(self, **kwargs):
        pass