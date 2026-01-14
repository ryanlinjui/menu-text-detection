from typing import List
from abc import ABC, abstractmethod

from PIL import Image

PROMPT = "The provided images display a menu. IMPORTANT: There may be MULTIPLE images representing different pages. You MUST examine EVERY image provided and combine all extracted information into the final result. Do not miss any dishes from any page."

class LLMBase(ABC):
    @classmethod
    @abstractmethod
    def call(cls, images: List[Image.Image], model: str, token: str) -> dict:
        raise NotImplementedError