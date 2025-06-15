import os
import json

import numpy as np
import gradio as gr
from dotenv import load_dotenv

from menu.llm import (
    GeminiAPI,
    OpenAIAPI
)
from menu.donut import DonutFinetuned

load_dotenv(override=True)
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN", "")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN", "")

SOURCE_CODE_GH_URL = "https://github.com/ryanlinjui/menu-text-detection"
BADGE_URL = "https://img.shields.io/badge/GitHub_Code-Click_Here!!-default?logo=github"

GITHUB_RAW_URL = "https://raw.githubusercontent.com/ryanlinjui/menu-text-detection/main"
EXAMPLE_IMAGE_LIST = [
    f"{GITHUB_RAW_URL}/examples/menu-hd.jpg",
    f"{GITHUB_RAW_URL}/examples/menu-vs.jpg",
    f"{GITHUB_RAW_URL}/examples/menu-si.jpg"
]
MODEL_LIST = [
    "Donut Model",
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gpt-4.1",
    "gpt-4o",
    "o4-mini"
]

def handle(image: np.ndarray, model: str, api_token: str) -> str:
    if image is None:
        raise gr.Error("Please upload an image first.")
    
    if model == MODEL_LIST[0]:
        result = DonutFinetuned.predict(image)
    
    elif model in MODEL_LIST[1:]:
        if len(api_token) < 10:
            raise gr.Error(f"Please provide a valid token for {model}.")
        try:
            if model in MODEL_LIST[1:4]:
                result = GeminiAPI.call(image, model, api_token)
            else:
                result = OpenAIAPI.call(image, model, api_token)
        except Exception as e:
            raise gr.Error(f"Failed to process with API model {model}: {str(e)}")
    else:
        raise gr.Error("Invalid model selection. Please choose a valid model.")
    
    return json.dumps(result, indent=4, ensure_ascii=False, sort_keys=True)

def UserInterface() -> gr.Interface:
    with gr.Blocks(
        delete_cache=(86400, 86400),
        css="""
        .image-panel {
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .image-panel img {
            object-fit: contain;
            max-height: 600px;
            max-width: 600px;
            width: 100%;
        }
        .large-text textarea {
            font-size: 20px !important;
            height: 600px !important;
            width: 100% !important;
        }
        """
    ) as gradio_interface:
        gr.HTML(f'<a href="{SOURCE_CODE_GH_URL}"><img src="{BADGE_URL}" alt="GitHub Code"/></a>')
        gr.Markdown("# Menu Text Detection")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=500):
                gr.Markdown("## 📷 Menu Image")
                menu_image = gr.Image(
                    type="numpy", 
                    label="Input menu image",
                    elem_classes="image-panel"
                )
                
                gr.Markdown("## 🤖 Model Selection")
                model_choice_dropdown = gr.Dropdown(
                    choices=MODEL_LIST,
                    value=MODEL_LIST[0],
                    label="Select Text Detection Model"
                )
                
                api_token_textbox = gr.Textbox(
                    label="API Token",
                    placeholder="Enter your API token here...",
                    type="password",
                    visible=False
                )
                
                generate_button = gr.Button("Generate Menu Information", variant="primary")

                gr.Examples(
                    examples=EXAMPLE_IMAGE_LIST,
                    inputs=menu_image,
                    label="Example Menu Images"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🍽️ Menu Info")
                menu_json_textbox = gr.Textbox(
                    label="Ouput JSON",
                    interactive=False,
                    text_align="left",
                    elem_classes="large-text"
                )
        
        def update_token_visibility(choice):
            if choice in MODEL_LIST[1:]:
                current_token = ""
                if choice in MODEL_LIST[1:4]:
                    current_token = GEMINI_API_TOKEN
                elif choice in MODEL_LIST[4:]:
                    current_token = OPENAI_API_TOKEN
                return gr.update(visible=True, value=current_token)
            else:
                return gr.update(visible=False)
                
        model_choice_dropdown.change(
            fn=update_token_visibility,
            inputs=model_choice_dropdown,
            outputs=api_token_textbox
        )
        
        generate_button.click(
            fn=handle,
            inputs=[menu_image, model_choice_dropdown, api_token_textbox],
            outputs=menu_json_textbox
        )

    return gradio_interface

if __name__ == "__main__":
    demo = UserInterface()
    demo.launch()