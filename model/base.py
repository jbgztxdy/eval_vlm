from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    
    available_name = {}
    
    def __init__(self, name):
        self.model_name = name
        assert self.model_name in self.available_name
        self.init_model()
        print(f"Model {self.model_name} initialized.")
    
    @abstractmethod
    def init_model(self):
        pass
    
    @torch.no_grad()
    def generate(self, prompts, images=None, **generation_kwargs):
        pass
    
    @torch.no_grad()
    def forward(self, prompts, images=None):
        pass