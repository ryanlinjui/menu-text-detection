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

# Collect Menu Image Datasets
saf
- [Google AI Studio](https://aistudio.google.com) or [OpenAI ChatGPT](https://chatgpt.com) by this prompt
- Gemini API by doing the function calling. Start the gradio app and input image and access token to test it.

{"file_name":"94.jpg","ground_truth":{"gt_parse":{"restaurant":"安田麵屋","address":"","phone":"","business_hours":"","items":[{"name":"豚骨拉麵","price":"120"},{"name":"明太子烏龍麵","price":"100"}]}}}

### Setup

Use [pixi](https://pixi.sh/latest) to set up the development environment:

```bash
pixi shell
pixi install
```

### Run Training Script
TBD

```
resume_from_checkpoint_path: null # only used for resume_from_checkpoint option in PL
result_path: "./result"
pretrained_model_name_or_path: "naver-clova-ix/donut-base" # loading a pre-trained model (from moldehub or path)
dataset_name_or_paths: ["ryanlinjui/donut-menu-zh-TW"] # loading datasets (from moldehub or path)
sort_json_key: False # cord dataset is preprocessed, and publicly available at https://huggingface.co/datasets/naver-clova-ix/cord-v2
train_batch_sizes: [4]
val_batch_sizes: [1]
input_size: [1280, 960] # when the input resolution differs from the pre-training setting, some weights will be newly initialized (but the model training would be okay)
max_length: 768
align_long_axis: False
num_nodes: 1
seed: 2022
lr: 3e-5
warmup_steps: 30 # 800/8*30/10, 10%
num_training_samples_per_epoch: 80
max_epochs: 30
max_steps: -1
num_workers: 8
val_check_interval: 1.0
check_val_every_n_epoch: 1
gradient_clip_val: 1.0
verbose: True
```