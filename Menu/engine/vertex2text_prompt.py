from PIL import Image

from .base import MenuEngine
from utils.detection import (
    gcloud,
    vertex2text
)
from utils.analysis import gpt

class Vertex2TextPrompt(MenuEngine):

    @property
    def dataset(self) -> dict:
        return {
            "menu.jpg": self.menu_img,
            "text-vertex.json": self.text_vertex,
            "text.txt": self.text,
            "tool.json": self.tool,
            "menu.json": self.menu_info
        }

    def detection(self):
        self.text_vertex = gcloud.run(self.menu_img)
        self.text = vertex2text(self.text_vertex)

    def analysis(self):
        self.menu_info = gpt.run(
            text=self.text,
            tools=self.tool,
            model=self.gpt
        )