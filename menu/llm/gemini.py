import numpy as np
from PIL import Image
from google import genai
from google.genai import types

from menu.llm.base import LLMBase

def build_properties(schema: dict) -> dict:
    TYPE_MAPPING = {
        "string": "STRING",
        "float": "NUMBER",
        "integer": "NUMBER"
    }
    
    properties = {}
    for key, value in schema.items():
        if isinstance(value, list):
            if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], str):
                # Handle simple properties (string, float, integer)
                properties[key] = {
                    "type": TYPE_MAPPING.get(value[1], "STRING"),
                    "description": value[0]
                }
            elif len(value) == 1 and isinstance(value[0], dict):
                # Handle array of objects (items)
                item_properties = {}
                for item_key, item_value in value[0].items():
                    item_properties[item_key] = {
                        "type": TYPE_MAPPING.get(item_value[1], "STRING"),
                        "description": item_value[0]
                    }
                
                properties[key] = {
                    "type": "ARRAY",
                    "description": f"List of {key}",
                    "items": {
                        "type": "OBJECT",
                        "properties": item_properties,
                        "required": list(value[0].keys())
                    }
                }
    
    return properties

class GeminiAPI(LLMBase):
    @classmethod
    def call(cls: LLMBase, image: np.ndarray, model: str, token: str) -> dict:
        client = genai.Client(api_key=token) # Initialize the client with the API key
        encode_img = Image.fromarray(image) # Convert the image for the API
        properties = build_properties(cls.schema) # Build JSON properties from schema for function call

        # Define the function parameters
        function = types.FunctionDeclaration(
            name="extract_menu_data",
            parameters=types.Schema(
                type="OBJECT",
                properties=properties,
                required=list(cls.schema.keys())
            )
        )
        
        # Function calling API
        try:
            response = client.models.generate_content(
                model=model,
                contents=[cls.prompt, encode_img],
                config=types.GenerateContentConfig(tools=[types.Tool(function_declarations=[function])])
            )
            if response.function_calls:
                # Access function arguments using the correct attribute structure
                # The arguments are accessed using function_call.args instead of function_call.arguments
                return response.function_calls[0].args
            
            # Try to parse JSON from the text response if no function call is detected
            try:
                # Extract JSON from the response text if it's wrapped in markdown code blocks
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1]
                    if "```" in text:
                        text = text.split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1]
                    if "```" in text:
                        text = text.split("```")[0]
                
                # Clean up the text and try to parse it as JSON
                text = text.strip()
                import json
                result = json.loads(text)
                return result
            except Exception as json_error:
                return {"error": "No function call in response and failed to parse JSON", 
                        "details": response.text, 
                        "json_error": str(json_error)}
                
        except Exception as e:
            return {"error": "Gemini API error", "details": str(e)}