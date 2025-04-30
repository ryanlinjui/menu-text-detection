import json
from abc import ABC, abstractmethod

import numpy as np

SCHEMA_FILE_PATH = "tools/schema.json"
PROMPT_FILE_PATH = "tools/prompt.txt"

def load_schema() -> dict:
    with open(SCHEMA_FILE_PATH, "r") as f:
        return json.load(f)

def load_prompt() -> str:
    with open(PROMPT_FILE_PATH, "r") as f:
        return f.read().strip()

class LLMBase(ABC):
    schema = load_schema()
    prompt = load_prompt()
        
    @classmethod
    @abstractmethod
    def call(cls, image: np.ndarray, model: str, token: str) -> dict:
        raise NotImplementedError