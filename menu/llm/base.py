from abc import ABC, abstractmethod

from PIL import Image
import numpy as np

class LLMBase(ABC):
    @classmethod
    @abstractmethod
    def call(cls, image: Image.Image, model: str, token: str) -> dict:
        raise NotImplementedError