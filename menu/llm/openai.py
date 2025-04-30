import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI

from menu.llm.base import LLMBase

def build_properties(schema: dict) -> dict:
    TYPE_MAPPING = {
        "string": "string",
        "float": "number",
        "integer": "integer"
    }
    
    properties = {}
    for key, value in schema.items():
        if isinstance(value, list):
            if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], str):
                # Handle simple properties (string, float, integer)
                properties[key] = {
                    "type": TYPE_MAPPING.get(value[1], "string"),
                    "description": value[0]
                }
            elif len(value) == 1 and isinstance(value[0], dict):
                # Handle array of objects (items)
                item_properties = {}
                for item_key, item_value in value[0].items():
                    item_properties[item_key] = {
                        "type": TYPE_MAPPING.get(item_value[1], "string"),
                        "description": item_value[0]
                    }
                
                properties[key] = {
                    "type": "array",
                    "description": f"List of {key}",
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": list(value[0].keys())
                    }
                }
    
    return properties

class OpenAIAPI(LLMBase):
    @classmethod
    def call(cls: LLMBase, image: np.ndarray, model: str, token: str) -> dict:
        client = OpenAI(api_key=token)  # Initialize the client with the API key
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        encode_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        properties = build_properties(cls.schema)
        
        # Define the function parameters
        function = {
            "name": "extract_menu_data",
            "description": "Extract menu data from the image",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(cls.schema.keys())
            }
        }
        
        # Function calling API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": cls.prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract menu information from this image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_img}"}}
                    ]}
                ],
                tools=[{"type": "function", "function": function}],
                tool_choice={"type": "function", "function": {"name": "extract_menu_data"}}
            )
            
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return json.loads(tool_call.function.arguments)
            return {"error": "No function call in response", "details": str(response)}
                
        except Exception as e:
            return {"error": "OpenAI API error", "details": str(e)}
