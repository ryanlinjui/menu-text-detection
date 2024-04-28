import os
import shutil
from PIL import Image
from io import BytesIO
import json
from pillow_heif import register_heif_opener

from utils import auto_orient
from Menu.engine import *
from Menu.config import *

# HEIF Image Support
register_heif_opener()

class MenuInstance:
    def __init__(self, img: Image.Image|str|bytes):
        self.dataset = {
            RAW_DATA: None,
            AUTO_ORIENT: None,
            Original.__name__: {},
            Vertex2TextPrompt.__name__: {},
            Vertex2TextPromptBiDir.__name__: {}
        }

        if isinstance(img, Image.Image):
            self.rawdata_img = img
        elif isinstance(img, str):
            self.rawdata_img = Image.open(img)
        elif isinstance(img, bytes):
            self.rawdata_img = Image.open(BytesIO(img))
        else:
            raise ValueError("Unsupported file type, only PIL.Image.Image or str or bytes type are allowed")

        self.auto_orient_img = auto_orient(self.rawdata_img.copy())

        self.dataset[RAW_DATA] = self.rawdata_img
        self.dataset[AUTO_ORIENT] = self.auto_orient_img

    def run(self, engine: MenuEngine) -> dict:
        
        if issubclass(engine, MenuEngine):
            engine_instance = engine(self.auto_orient_img)
            menu_info = engine_instance.run()
            self.dataset[engine_instance.__class__.__name__] = engine_instance.dataset
            return menu_info
        else:
            raise ValueError("You have to use MenuEngine class type.")

    def save(self, dataset_name:str):
        root_directory = os.path.join(DATASET_DIRECTORY, dataset_name)
        shutil.rmtree(root_directory, ignore_errors=True)
        os.makedirs(root_directory, exist_ok=True)
        
        stack = [(root_directory, self.dataset.items())]
        while stack:
            current_directory, current_content = stack.pop()

            for content_name, content_value in current_content:
                content_path = os.path.join(current_directory, content_name)

                if isinstance(content_value, Image.Image) and content_path.endswith(".jpg"):  # If content is a PIL Image, save it as a jpg file
                    content_value.save(content_path, "JPEG")

                elif isinstance(content_value, str) and content_path.endswith(".txt"):  # If content is a string, save it as a txt file
                    with open(content_path, "w") as txt_file:
                        txt_file.write(content_value)

                elif isinstance(content_value, dict) and content_path.endswith(".json"):  # If content is a dictionary, save it as a json file
                    with open(content_path, "w") as json_file:
                        json.dump(content_value, json_file, indent=4, ensure_ascii=False)

                elif isinstance(content_value, dict):  # If content is a dictionary, create a subdirectory
                    os.makedirs(content_path, exist_ok=True)
                    stack.append((content_path, content_value.items()))