import os
import json
import requests
from io import BytesIO
from typing import List

import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from pillow_heif import register_heif_opener

from menu.llm import (
    GeminiAPI,
    OpenAIAPI
)
from menu.donut import DonutFinetuned

donut_finetuned = DonutFinetuned("ryanlinjui/donut-base-finetuned-menu")

register_heif_opener()
load_dotenv(override=True)
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN", "")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN", "")

SOURCE_CODE_GH_URL = "https://github.com/ryanlinjui/menu-text-detection"
BADGE_URL = "https://img.shields.io/badge/GitHub_Code-Click_Here!!-default?logo=github"

GITHUB_RAW_URL = "https://raw.githubusercontent.com/ryanlinjui/menu-text-detection/main"
EXAMPLE_IMAGE_LIST = [
    [f"{GITHUB_RAW_URL}/examples/menu-hd.jpg"],
    [f"{GITHUB_RAW_URL}/examples/menu-vs.jpg"],
    [f"{GITHUB_RAW_URL}/examples/menu-si.jpg"]
]
FINETUNED_MODEL_LIST = [
    "Donut (Document Parsing Task) Fine-tuned Model"
]
LLM_MODEL_LIST = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gpt-4.1",
    "gpt-4o",
    "o4-mini"
]
CSS_STYLE = """
    .image-panel img {
        max-height: 500px;
        margin-top: -100px;
    }
    .large-text textarea {
        font-size: 20px !important;
        height: 600px !important;
        width: 100% !important;
    }
    .control-row {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .page-info {
        text-align: center !important;
        font-size: 20px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        font-weight: 900 !important;
        color: #374151; /* Darker gray for clarity */
    }
    .page-info p {
        margin: 0 !important;
        width: 100% !important;
        text-align: center !important;
    }
    .upload-btn {
        margin-top: 2px !important;
        background-color: #e0f2fe !important; /* Light blue background */
        color: #0369a1 !important; /* Dark blue text */
        border: 1px solid #0ea5e9 !important;
    }
    .upload-btn:hover {
        background-color: #bae6fd !important;
    }
    .clear-btn {
        margin-top: 2px !important;
    }
    .image-container {
        height: 650px !important;
        display: flex;
        flex-direction: column;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 4px;
    }
"""

def handle(images: List[str], model: str, api_token: str) -> str:
    if not images:
        raise gr.Error("Please upload an image first.")

    # Convert file paths or URLs to PIL Images
    pil_images = []
    for img in images:
        if img.startswith("http://") or img.startswith("https://"):
            try:
                response = requests.get(img)
                response.raise_for_status()
                pil_images.append(Image.open(BytesIO(response.content)))
            except Exception as e:
                raise gr.Error(f"Failed to load image from URL: {str(e)}")
        else:
            pil_images.append(Image.open(img))
    
    if model == FINETUNED_MODEL_LIST[0]:
        result = donut_finetuned.predict(pil_images[0])

    elif model in LLM_MODEL_LIST:
        if len(api_token) < 10:
            raise gr.Error(f"Please provide a valid token for {model}.")
        try:
            if model in LLM_MODEL_LIST[:3]:
                result = GeminiAPI.call(pil_images, model, api_token)
            else:
                result = OpenAIAPI.call(pil_images, model, api_token)
        except Exception as e:
            raise gr.Error(f"Failed to process with API model {model}: {str(e)}")
    else:
        raise gr.Error("Invalid model selection. Please choose a valid model.")
    
    return json.dumps(result, indent=4, ensure_ascii=False, sort_keys=True)

def UserInterface() -> gr.Interface:
    with gr.Blocks(delete_cache=(86400, 86400)) as gradio_interface:
        gr.HTML(f'<a href="{SOURCE_CODE_GH_URL}"><img src="{BADGE_URL}" alt="GitHub Code"/></a>')
        gr.Markdown("# Menu Text Detection")

        images_state = gr.State([])
        current_index_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1, min_width=500):
                gr.Markdown("## 📷 Menu Image")
                
                with gr.Column(elem_classes="image-container"):
                    menu_image_display = gr.Image(
                        label="Input menu image", 
                        type="filepath", 
                        elem_classes="image-panel",
                        interactive=False,
                        show_label=True,
                        height="100%",
                        width="100%"
                    )
                    with gr.Row(elem_classes="control-row"):
                        prev_btn = gr.Button("◀️ Previous", variant="secondary", scale=1)
                        with gr.Column(scale=2, min_width=50):
                            page_info = gr.Markdown("Page 1 / 1", elem_classes="page-info")
                        next_btn = gr.Button("Next ▶️", variant="secondary", scale=1)
                    
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            "📷 Upload Menu Images", 
                            file_types=["image"], 
                            file_count="multiple",
                            scale=3,
                            elem_classes="upload-btn",
                            variant="primary"
                        )
                        clear_btn = gr.Button("🗑️ Remove", variant="stop", scale=1, elem_classes="clear-btn")

                gr.Markdown("## 🤖 Model Selection")
                model_choice_dropdown = gr.Dropdown(
                    choices=FINETUNED_MODEL_LIST + LLM_MODEL_LIST,
                    value=FINETUNED_MODEL_LIST[0],
                    label="Select Text Detection Model"
                )
                
                api_token_textbox = gr.Textbox(
                    label="API Token",
                    placeholder="Enter your API token here...",
                    type="password",
                    visible=False
                )
                
                generate_button = gr.Button("Generate Menu Information", variant="primary")
                example_receiver = gr.Image(visible=False, label="Example Preview", type="filepath")
                
                examples_component = gr.Examples(
                    examples=[[img_list[0]] for img_list in EXAMPLE_IMAGE_LIST],
                    inputs=example_receiver,
                    label="Example Menu Images"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🍽️ Menu Info")
                menu_json_textbox = gr.Textbox(
                    label="Output JSON",
                    interactive=True,
                    text_align="left",
                    elem_classes="large-text"
                )

        def update_display(images, index):
            if not images:
                return None, "Page 1 / 1"
            idx = max(0, min(index, len(images) - 1))
            return images[idx], f"Page {idx + 1} / {len(images)}"

        def on_upload(new_files, current_images):
            if current_images is None:
                current_images = []
            if new_files:
                new_paths = [f.name for f in new_files]
                current_images.extend(new_paths)
            new_index = len(current_images) - 1 
            img, info = update_display(current_images, new_index)
            return current_images, new_index, img, info

        upload_btn.upload(
            fn=on_upload,
            inputs=[upload_btn, images_state],
            outputs=[images_state, current_index_state, menu_image_display, page_info]
        )

        def on_clear(images, index):
            if not images:
                return [], 0, None, "Page 1 / 1"
            
            new_images = list(images)
            if 0 <= index < len(new_images):
                new_images.pop(index)
            
            if not new_images:
                 return [], 0, None, "Page 1 / 1"

            new_index = index
            if new_index >= len(new_images):
                new_index = len(new_images) - 1
            
            img, info = update_display(new_images, new_index)
            return new_images, new_index, img, info

        clear_btn.click(
            fn=on_clear,
            inputs=[images_state, current_index_state],
            outputs=[images_state, current_index_state, menu_image_display, page_info]
        )

        def on_prev(images, index):
            if not images:
                return 0, None, "Page 1 / 1"
            new_index = max(0, index - 1)
            img, info = update_display(images, new_index)
            return new_index, img, info

        def on_next(images, index):
            if not images:
                return 0, None, "Page 1 / 1"
            new_index = min(len(images) - 1, index + 1)
            img, info = update_display(images, new_index)
            return new_index, img, info

        prev_btn.click(on_prev, [images_state, current_index_state], [current_index_state, menu_image_display, page_info])
        next_btn.click(on_next, [images_state, current_index_state], [current_index_state, menu_image_display, page_info])

        def on_example_click(evt: gr.SelectData):
            if evt.index is None:
                return [], 0, None, "Page 1 / 1"
            
            # Retrieve the full batch based on the clicked index
            if 0 <= evt.index < len(EXAMPLE_IMAGE_LIST):
                current_images = EXAMPLE_IMAGE_LIST[evt.index]
            else:
                current_images = []
            
            new_index = 0
            img, info = update_display(current_images, new_index)
            return current_images, new_index, img, info

        examples_component.dataset.select(
            fn=on_example_click,
            inputs=None,
            outputs=[images_state, current_index_state, menu_image_display, page_info]
        )
                        
        def update_token_visibility(choice):
            if choice in LLM_MODEL_LIST:
                current_token = ""
                if choice in LLM_MODEL_LIST[:3]:
                    current_token = GEMINI_API_TOKEN
                else:
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
            inputs=[images_state, model_choice_dropdown, api_token_textbox],
            outputs=menu_json_textbox
        )

    return gradio_interface

if __name__ == "__main__":
    demo = UserInterface()
    demo.launch(css=CSS_STYLE)