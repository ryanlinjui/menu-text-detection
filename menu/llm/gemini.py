import json
from typing import List

from PIL import Image
from google import genai
from google.genai import types

from .base import LLMBase, PROMPT

FUNCTION_CALL = json.load(open("tools/schema_gemini.json", "r"))

class GeminiAPI(LLMBase):
    @classmethod
    def call(cls, images: List[Image.Image], model: str, token: str) -> dict:
        client = genai.Client(api_key=token) # Initialize the client with the API key

        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[FUNCTION_CALL])],
            tool_config={
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [FUNCTION_CALL["name"]]
                }
            }
        )        
        
        response = client.models.generate_content(
            model=model,
            contents=images + [PROMPT],
            config=config
        )
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            return function_call.args

        return {}