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
Use [uv](https://github.com/astral-sh/uv) to set up the development environment:

```bash
uv sync
```

### Training Script (Datasets collecting, Fine-Tuning)
Please refer [`train.ipynb`](./train.ipynb). Use Jupyter Notebook for training:

```bash
uv run jupyter-notebook
```

> For VSCode users, please install Jupyter extension, then select `.venv/bin/python` as your kernel.

### Run Demo Locally
```bash
uv run python app.py
```
