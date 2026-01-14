import json
import base64
from io import BytesIO
from typing import List

from PIL import Image
from openai import OpenAI

from .base import LLMBase, PROMPT

FUNCTION_CALL = json.load(open("tools/schema_openai.json", "r"))

class OpenAIAPI(LLMBase):
    @classmethod
    def call(cls, images: List[Image.Image], model: str, token: str) -> dict:
        client = OpenAI(api_key=token)  # Initialize the client with the API key
        
        content = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            encode_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_img}"},
            })

        content.append({"type": "text", "text": PROMPT})
        
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            tools=[FUNCTION_CALL],
        )
        if response and response.output:
            if hasattr(response.output[0], "arguments"):
                return json.loads(response.output[0].arguments)
        return {}