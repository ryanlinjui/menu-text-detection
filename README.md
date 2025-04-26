# Menu Text Detection System

Extract structured menu information from images into JSON using a fine-tuned Donut E2E model.  
> Based on [Donut by Clova AI (ECCV ’22)](https://github.com/clovaai/donut)

<div align="center">

<img src="./assets/demo.gif" alt="demo" width="500"/><br>

Demo (coming soon): [![Gradio Space Demo](https://img.shields.io/badge/GradioSpace-Demo-important?logo=huggingface)]()<br>
[![Hugging Face Models & Datasets](https://img.shields.io/badge/HuggingFace-Models_&_Datasets-important?logo=huggingface)](https://huggingface.co/collections/ryanlinjui/menu-text-detection-670ccf527626bb004bbfb39b)

</div>

## 🚀 Features

Automatically extract the following menu information as JSON:

- **Restaurant Name**  
- **Business Hours**  
- **Contact Information**  
  - Address  
  - Phone  
- **Dish Information**  
  - Name  
  - Price  

> For the JSON schema, see [schema.json]()

## 🔧 Supported Methods

- OpenAI GPT API  
- Google Gemini API  
- Fine-tuned Donut model  

## 💻 Training / Fine-Tuning

### Setup

Use [pixi](https://pixi.sh/latest) to set up the development environment:

```bash
pixi shell
pixi install
```

### Run Training Script
TBD
