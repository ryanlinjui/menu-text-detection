import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI

from .base import LLMBase

FUNCTION_CALL = json.load(open("tools/schema_openai.json", "r"))

class OpenAIAPI(LLMBase):
    @classmethod
    def call(cls, image: np.ndarray, model: str, token: str) -> dict:
        client = OpenAI(api_key=token)  # Initialize the client with the API key
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        encode_img = base64.b64encode(buffer.getvalue()).decode("utf-8") # Convert the image for the API

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encode_img}",
                        },
                    ],
                }
            ],
            tools=[FUNCTION_CALL],
        )
        if response and response.output:
            if hasattr(response.output[0], "arguments"):
                return json.loads(response.output[0].arguments)
        return {}