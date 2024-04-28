from PIL import Image

from .base import MenuEngine
from utils.detection import (
    gcloud,
    vertex2text
)
from utils.analysis import gpt

class Vertex2TextPromptBiDir(MenuEngine):

    @property
    def dataset(self) -> dict:
        return {
            "menu-0.jpg": self.menu_img[0],
            "menu-90.jpg": self.menu_img[1],
            "text-vertex-0.json": self.text_vertex[0],
            "text-vertex-90.json": self.text_vertex[1],
            "text.txt": self.text,
            "tool.json": self.tool,
            "menu.json": self.menu_info
        }

    def detection(self):
        self.menu_img = [
            self.menu_img,
            self.menu_img.rotate(90, expand=True)
        ]

        self.text_vertex = [
            gcloud.run(self.menu_img[0]),
            gcloud.run(self.menu_img[1])
        ]
        
        self.text = \
            f"Choose the most correctly menu information:\n"\
            "------------------- \n\n" \
            f"{vertex2text(self.text_vertex[0])}" \
            "\n\n-----------------\n\n" \
            f"{vertex2text(self.text_vertex[1])}"

    def analysis(self):
        self.menu_info = gpt.run(
            text=self.text,
            tools=self.tool,
            model=self.gpt
        )