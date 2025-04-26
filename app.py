import json

import numpy as np
import gradio as gr

SOURCE_CODE_GH_URL = "https://github.com/ryanlinjui/menu-text-detection"
BADGE_URL = "https://img.shields.io/badge/GitHub_Code-Click_Here!!-default?logo=github"

EXAMPLE_IMAGE_LIST = [
    "examples/menu-hd.jpg",
    "examples/menu-vs.jpg",
    "examples/menu-si.jpg"
]
MODEL_LIST = [
    "Donut Model",
    "Gemini API: Gemini-1.5-turbo",
    "Gemini API: Gemini-2.0-turbo",
    "Gemini API: Gemini-2.0-turbo-16k",
    "OpenAI API: gpt-4.0-turbo",
    "OpenAI API: gpt-3.5-turbo",
    "OpenAI API: gpt-4.0-turbo-16k"
]

def handle(image: np.ndarray, model: str, api_token: str) -> str:
    if image is None:
        raise gr.Error("Please upload an image first.")

    result = {
        "restaurant_name": "阿平麵食館",
        "business_hours": "",
        "contact": {
            "address": "新北市安民街309號",
            "phone": "(02)22112580"
        },
        "dish": [{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"},{"name": "三寶牛肉麵", "price": "180"}]
    }
    if model == MODEL_LIST[0]:
        result["model_info"] = "Using local Donut model for text detection"

    elif model == MODEL_LIST[1:4]:
        if len(api_token) < 10:
            raise gr.Error("Please provide a valid Gemini API token.")
        
        result["token_status"] = f"Valid Gemini API token provided: {api_token}"

    elif model == MODEL_LIST[4:]:
        if len(api_token) < 10:
            raise gr.Error("Please provide a valid OpenAI API token.")
        
        result["token_status"] = f"Valid OpenAI API token provided: {api_token}"
    
    else:
        raise gr.Error("Invalid model selection. Please choose a valid model.")
    
    return json.dumps(result, indent=4)

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
        gr.Markdown(
            f"""
            [![GitHub Code]({BADGE_URL})]({SOURCE_CODE_GH_URL})
            # Menu Text Detection
            """
        )
        
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
                return gr.Textbox(visible=True)
            else:
                return gr.Textbox(visible=False)
                
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
    demo.launch(share=True)