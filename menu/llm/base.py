from abc import ABC, abstractmethod

import numpy as np

class LLMBase(ABC):
    @classmethod
    @abstractmethod
    def call(image: np.ndarray, model: str, token: str) -> dict:
        raise NotImplementedError