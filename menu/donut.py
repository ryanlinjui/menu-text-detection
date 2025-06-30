"""
This file is modified from the HuggingFace transformers tutorial script for fine-tuning Donut on a custom dataset.
It's defined from `.ipynb` to the module implementation for better reusability and maintainability.
Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb
"""

import re
import random
from typing import Any, List, Tuple

import torch
import numpy as np
from PIL import Image
from nltk import edit_distance
import pytorch_lightning as pl
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import upload_folder
from pillow_heif import register_heif_opener
from pytorch_lightning.callbacks import Callback
from transformers import pipeline, DonutProcessor
from transformers import VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig
from pytorch_lightning.loggers import TensorBoardLogger

TASK_PROMPT_NAME = "<s_menu-text-detection>"
register_heif_opener()

class DonutFinetuned:
    def __init__(self, pretrained_model_repo_id: str = "ryanlinjui/donut-test"):
        self.processor = DonutProcessor.from_pretrained(pretrained_model_repo_id)
        self.pipe = pipeline(
            task="image-text-to-text",
            model=pretrained_model_repo_id,
            processor=self.processor
        )

    def predict(self, image: np.ndarray) -> dict:
        image = Image.fromarray(image)
        outputs = self.pipe(text=TASK_PROMPT_NAME, images=image)[0]["generated_text"]
        return self.processor.token2json(outputs)
    
class DonutTrainer:
    processor = None
    max_length = 768
    image_size = [1280, 960]
    added_tokens = []
    train_dataloader = None
    val_dataloader = None
    huggingface_model_id = None

    class DonutDataset(Dataset):
        """
        PyTorch Dataset for Donut. This class takes a HuggingFace Dataset as input.

        Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
        and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

        Args:
            dataset: HuggingFace DatasetDict containing the dataset to be used
            max_length: the max number of tokens for the target sequences
            split: whether to load "train", "validation" or "test" split
            ignore_id: ignore_index for torch.nn.CrossEntropyLoss
            task_start_token: the special token to be fed to the decoder to conduct the target task
            prompt_end_token: the special token at the end of the sequences
            sort_json_key: whether or not to sort the JSON keys
        """

        def __init__(
            self,
            dataset: DatasetDict,
            ground_truth_key: str,
            max_length: int,
            split: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
        ):
            super().__init__()

            self.dataset = dataset[split]
            self.ground_truth_key = ground_truth_key
            self.max_length = max_length
            self.split = split
            self.ignore_id = ignore_id
            self.task_start_token = task_start_token
            self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
            self.sort_json_key = sort_json_key

            self.dataset_length = len(self.dataset)

            self.gt_token_sequences = []
            for sample in self.dataset:
                ground_truth = sample[self.ground_truth_key]
                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            update_special_tokens_for_json_key=self.split == "train",
                            sort_json_key=self.sort_json_key,
                        )
                        + DonutTrainer.processor.tokenizer.eos_token
                        for gt_json in [ground_truth]  # load json from list of json
                    ]
                )

            self.add_tokens([self.task_start_token, self.prompt_end_token])
            self.prompt_end_token_id = DonutTrainer.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

        def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
            """
            Convert an ordered JSON object into a token sequence
            """
            if type(obj) == dict:
                if len(obj) == 1 and "text_sequence" in obj:
                    return obj["text_sequence"]
                else:
                    output = ""
                    if sort_json_key:
                        keys = sorted(obj.keys(), reverse=True)
                    else:
                        keys = obj.keys()
                    for k in keys:
                        if update_special_tokens_for_json_key:
                            self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                        output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                        )
                    return output
            elif type(obj) == list:
                return r"<sep/>".join(
                    [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
                )
            else:
                obj = str(obj)
                if f"<{obj}/>" in DonutTrainer.added_tokens:
                    obj = f"<{obj}/>"  # for categorical special tokens
                return obj

        def add_tokens(self, list_of_tokens: List[str]):
            """
            Add special tokens to tokenizer and resize the token embeddings of the decoder
            """
            newly_added_num = DonutTrainer.processor.tokenizer.add_tokens(list_of_tokens)
            if newly_added_num > 0:
                DonutTrainer.model.decoder.resize_token_embeddings(len(DonutTrainer.processor.tokenizer))
                DonutTrainer.added_tokens.extend(list_of_tokens)

        def __len__(self) -> int:
            return self.dataset_length

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Load image from image_path of given dataset_path and convert into input_tensor and labels
            Convert gt data into input_ids (tokenized string)
            Returns:
                input_tensor : preprocessed image
                input_ids : tokenized gt_data
                labels : masked labels (model doesn't need to predict prompt and pad token)
            """
            sample = self.dataset[idx]

            # inputs
            pixel_values = DonutTrainer.processor(sample["image"], random_padding=self.split == "train", return_tensors="pt").pixel_values
            pixel_values = pixel_values.squeeze()

            # targets
            target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
            input_ids = DonutTrainer.processor.tokenizer(
                target_sequence,
                add_special_tokens=False,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            labels = input_ids.clone()
            labels[labels == DonutTrainer.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
            # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return pixel_values, labels, target_sequence

    class DonutModelPLModule(pl.LightningModule):
        def __init__(self, config, processor, model):
            super().__init__()
            self.config = config
            self.processor = processor
            self.model = model

        def training_step(self, batch, batch_idx):
            pixel_values, labels, _ = batch

            outputs = self.model(pixel_values, labels=labels)
            loss = outputs.loss
            
            # Log training metrics
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
            
            return loss

        def validation_step(self, batch, batch_idx, dataset_idx=0):
            pixel_values, labels, answers = batch
            batch_size = pixel_values.shape[0]
            # we feed the prompt to the model
            decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

            outputs = self.model.generate(pixel_values,
                                    decoder_input_ids=decoder_input_ids,
                                    max_length=DonutTrainer.max_length,
                                    early_stopping=True,
                                    pad_token_id=self.processor.tokenizer.pad_token_id,
                                    eos_token_id=self.processor.tokenizer.eos_token_id,
                                    use_cache=True,
                                    num_beams=1,
                                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                    return_dict_in_generate=True,)

            predictions = []
            for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
                seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
                seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
                predictions.append(seq)

            scores = []
            for pred, answer in zip(predictions, answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                # NOT NEEDED ANYMORE
                # answer = re.sub(r"<.*?>", "", answer, count=1)
                answer = answer.replace(self.processor.tokenizer.eos_token, "")
                scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

                if self.config.get("verbose", False) and len(scores) == 1:
                    print(f"Prediction: {pred}")
                    print(f"    Answer: {answer}")
                    print(f" Normed ED: {scores[0]}")

            avg_edit_distance = np.mean(scores)
            print(f"Average Edit Distance: {avg_edit_distance:.4f}")
            self.log("val_edit_distance", avg_edit_distance, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_accuracy", 1 - avg_edit_distance, on_step=False, on_epoch=True, prog_bar=True)

            return scores

        def configure_optimizers(self):
            # you could also add a learning rate scheduler if you want
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
            return optimizer

        def train_dataloader(self):
            return DonutTrainer.train_dataloader

        def val_dataloader(self):
            return DonutTrainer.val_dataloader

    class PushToHubCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
            pl_module.model.push_to_hub(DonutTrainer.huggingface_model_id, commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            self._upload_logs(trainer.logger.log_dir, trainer.current_epoch)

        def on_train_end(self, trainer, pl_module):
            print(f"Pushing model to the hub after training")
            pl_module.processor.push_to_hub(DonutTrainer.huggingface_model_id,commit_message=f"Training done")
            pl_module.model.push_to_hub(DonutTrainer.huggingface_model_id, commit_message=f"Training done")
            self._upload_logs(trainer.logger.log_dir, "final")

        def _upload_logs(self, log_dir: str, epoch_info):
            try:
                print(f"Attempting to upload logs from: {log_dir}")
                upload_folder(log_dir, repo_id=DonutTrainer.huggingface_model_id, 
                            folder_path="tensorboard_logs",
                            commit_message=f"Upload logs - epoch {epoch_info}", ignore_patterns=["*.tmp", "*.lock"])
                print(f"Successfully uploaded logs for epoch {epoch_info}")
            except Exception as e:
                print(f"Failed to upload logs: {e}")
                pass

    @classmethod
    def train(
        cls,
        dataset: DatasetDict,
        pretrained_model_repo_id: str,
        huggingface_model_id: str,
        epochs: int,
        train_batch_size: int,
        val_batch_size: int,
        learning_rate: float,
        val_check_interval: float,
        check_val_every_n_epoch: int,
        gradient_clip_val: float,
        num_training_samples_per_epoch: int,
        num_nodes: int,
        warmup_steps: int,
        ground_truth_key: str = "ground_truth",
    ):
        cls.huggingface_model_id = huggingface_model_id
        config = VisionEncoderDecoderConfig.from_pretrained(pretrained_model_repo_id)
        config.encoder.image_size = cls.image_size
        config.decoder.max_length = cls.max_length
        
        cls.processor = DonutProcessor.from_pretrained(pretrained_model_repo_id)
        cls.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_repo_id, config=config)
        cls.processor.image_processor.size = cls.image_size[::-1]
        cls.processor.image_processor.do_align_long_axis = False

        train_dataset = cls.DonutDataset(
            dataset=dataset,
            ground_truth_key=ground_truth_key,
            max_length=cls.max_length,
            split="train",
            task_start_token=TASK_PROMPT_NAME,
            prompt_end_token=TASK_PROMPT_NAME,
            sort_json_key=True
        )
        val_dataset = cls.DonutDataset(
            dataset=dataset,
            ground_truth_key=ground_truth_key,
            max_length=cls.max_length,
            split="validation",
            task_start_token=TASK_PROMPT_NAME,
            prompt_end_token=TASK_PROMPT_NAME,
            sort_json_key=True
        )

        cls.model.config.pad_token_id = cls.processor.tokenizer.pad_token_id
        cls.model.config.decoder_start_token_id = cls.processor.tokenizer.convert_tokens_to_ids([TASK_PROMPT_NAME])[0]

        cls.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        cls.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        config = {
            "max_epochs": epochs,
            "val_check_interval": val_check_interval, # how many times we want to validate during an epoch
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "gradient_clip_val": gradient_clip_val,
            "num_training_samples_per_epoch": num_training_samples_per_epoch,
            "lr": learning_rate,
            "train_batch_sizes": [train_batch_size],
            "val_batch_sizes": [val_batch_size],
            # "seed":2022,
            "num_nodes": num_nodes,
            "warmup_steps": warmup_steps, # 10%
            "result_path": "./.checkpoints",
            "verbose": True,
        }
        model_module = cls.DonutModelPLModule(config, cls.processor, cls.model)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {device} device")
        trainer = pl.Trainer(
                accelerator="gpu" if device == "cuda" else "mps" if device == "mps" else "cpu",
                devices=1 if device == "cuda" else 0,
                max_epochs=config.get("max_epochs"),
                val_check_interval=config.get("val_check_interval"),
                check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
                gradient_clip_val=config.get("gradient_clip_val"),
                precision=16 if device == "cuda" else 32, # we'll use mixed precision if device == "cuda"
                num_sanity_val_steps=0,
                logger=TensorBoardLogger(save_dir="./.checkpoints", name="donut_training", version=None),
                callbacks=[cls.PushToHubCallback()]
        )
        trainer.fit(model_module)