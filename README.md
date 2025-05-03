# Menu Text Detection System

Extract structured menu information from images into JSON using a fine-tuned Donut E2E model.  
> Based on [Donut by Clova AI (ECCV ’22)](https://github.com/clovaai/donut)

<div align="center">

<img src="./assets/demo.gif" alt="demo" width="500"/><br>

[![Gradio Space Demo](https://img.shields.io/badge/GradioSpace-Demo-important?logo=huggingface)](https://huggingface.co/spaces/ryanlinjui/menu-text-detection)<br>
[![Hugging Face Models & Datasets](https://img.shields.io/badge/HuggingFace-Models_&_Datasets-important?logo=huggingface)](https://huggingface.co/collections/ryanlinjui/menu-text-detection-670ccf527626bb004bbfb39b)

</div>

## 🚀 Features

### Overview
Currently supports the following information from menu images:

- **Restaurant Name**  
- **Business Hours**  
- **Address**  
- **Phone Number**
- **Dish Information**
  - Name  
  - Price  

> For the JSON schema, see [tools directory](./tools).

### Supported Methods to Extract Menu Information
- Fine-tuned Donut model
- OpenAI GPT API  
- Google Gemini API

## 💻 Training / Fine-Tuning

### Setup
Use [pixi](https://pixi.sh/latest) to set up the development environment:

```bash
pixi shell
pixi install
```

### Run Training Script
```
python train.py
```

### Run Demo Locally
```
python app.py
```

### Collect Menu Image Datasets
1. Collect menu images from the web or your own dataset.
  - [Google AI Studio](https://aistudio.google.com) or [OpenAI ChatGPT](https://chatgpt.com) by this prompt
  - Gemini API by doing the function calling. Start the gradio app and input image and access token to test it.
2. Use `metadata.jsonl` to store the metadata of the images.
3. Push the dataset to Hugging Face Datasets.