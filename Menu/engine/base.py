from abc import ABC, abstractmethod
from PIL import Image

from utils.analysis.tools import TOOL_1
from Menu.config import (
    DEFAULT_GPT_MODEL
)

class MenuEngine(ABC):

    def __init__(self, img:Image.Image):
        self.menu_img = img
        self.text_vertex = None,
        self.text = None
        self.gpt = DEFAULT_GPT_MODEL
        self.tool = TOOL_1
        self.menu_info = None

    @property
    @abstractmethod
    def dataset(self) -> dict:
        return NotImplemented

    @abstractmethod
    def detection(self):
        return NotImplemented

    @abstractmethod
    def analysis(self):
        return NotImplemented

    def run(self) -> dict:
        self.detection()
        self.analysis()
        return self.menu_info