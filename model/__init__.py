# from .llava import LLaVA
# from .qwen import Qwen2VL
# from .internvl import InternVL2
# from .llama3_2 import Llama32
# from .llm import LLM
# from .phi3 import Phi3
from .base import BaseModel

def get_model(name) -> BaseModel:
    if "llava" in name.lower():
        from .llava import LLaVA
        return LLaVA(name)
    elif "llm" in name.lower():
        from .llm import LLM
        return LLM(name)
    elif "qwen2" in name.lower():
        from .qwen import Qwen2VL
        return Qwen2VL(name)
    elif "internvl2" in name.lower():
        from .internvl import InternVL2
        return InternVL2(name)
    elif "llama" in name.lower():
        from .llama3_2 import Llama32
        return Llama32(name)
    elif "phi3" in name.lower():
        from .phi3 import Phi3
        return Phi3(name)
    else:
        raise KeyError