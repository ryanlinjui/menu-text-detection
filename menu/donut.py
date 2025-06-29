import json
from typing import Any, Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from datasets import DatasetDict
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import pipeline
from nltk.metrics import edit_distance

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Generate predictions for metrics computation
        decoder_input_ids = torch.full(
            (pixel_values.shape[0], 1),
            self.model.config.decoder_start_token_id,
            device=pixel_values.device
        )
        
        with torch.no_grad():
            predictions = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.config.get("max_length", 768),
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True
            )
        
        # Get target sequences from batch
        target_sequences = batch.get("target_sequence", [])
        
        self.validation_step_outputs.append({
            "val_loss": loss,
            "predictions": predictions.sequences,
            "target_sequences": target_sequences
        })
        
        self.log_dict({"val_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute edit distance metrics using the compute_metrics function
        if not self.validation_step_outputs:
            print("No validation outputs to process")
            return
            
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        
        for output in self.validation_step_outputs:
            predictions = output["predictions"]
            target_sequences = output["target_sequences"]
            
            decoded_preds = self.processor.tokenizer.batch_decode(predictions, skip_special_tokens=False)
            all_predictions.extend(decoded_preds)
            
            # Handle different types of target sequences
            if isinstance(target_sequences, str):
                all_targets.append(target_sequences)
            elif isinstance(target_sequences, torch.Tensor):
                all_targets.extend(target_sequences.tolist() if target_sequences.dim() > 0 else [target_sequences.item()])
            elif isinstance(target_sequences, (list, tuple)):
                all_targets.extend(target_sequences)
        
        # Create a mock eval_pred object for compute_metrics
        class MockEvalPred:
            def __init__(self, predictions, targets):
                self.predictions = predictions
                self.targets = targets
        
        # Convert predictions back to token IDs for compute_metrics
        pred_token_ids = []
        for pred_text in all_predictions:
            tokens = self.processor.tokenizer(pred_text, return_tensors="pt", add_special_tokens=False)
            pred_token_ids.append(tokens.input_ids.squeeze().tolist())
        
        eval_pred = MockEvalPred(pred_token_ids, all_targets)
        
        # Use the compute_metrics function
        metrics = self.compute_metrics(eval_pred)
        avg_ned = metrics["normed_edit_distance"]
        
        # Log the metric
        self.log("val_ned", avg_ned, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_step_outputs.clear()
    
    def compute_metrics(self, eval_pred) -> dict:
        from functools import reduce
        
        # Get filtered tokens - add task prompt to ensure clean comparison
        filtered_tokens = [
            self.processor.tokenizer.bos_token,
            self.processor.tokenizer.eos_token,
            self.processor.tokenizer.pad_token,
            self.processor.tokenizer.unk_token,
            "<s_menu-text-detection>"  # Add task prompt to filtered tokens
        ]
        
        decoded_preds = self.processor.tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=False)
        
        normed_eds = []
        for idx, pred in enumerate(decoded_preds):
            if idx < len(eval_pred.targets):
                prediction_sequence = reduce(lambda s, t: s.replace(t, ""), filtered_tokens, pred)
                target_sequence = reduce(lambda s, t: s.replace(t, ""), filtered_tokens, str(eval_pred.targets[idx]))
                ed = edit_distance(prediction_sequence, target_sequence) / max(len(prediction_sequence), len(target_sequence), 1)
                normed_eds.append(ed)

                print(f"[Sample {idx+1}]")
                print(f"  Prediction: {prediction_sequence}")
                print(f"  Target: {target_sequence}")
                print(f"  Normalized Edit Distance: {ed:.4f}")
                print("-" * 40)

        avg_ned = float(np.mean(normed_eds)) if normed_eds else 1.0
        print(f"Average Normalized Edit Distance: {avg_ned:.4f}")
        
        return {"normed_edit_distance": avg_ned}

    def configure_optimizers(self):
        # Use same config as original notebook
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 3e-5), weight_decay=0.01)
        
        if "warmup_steps" in self.config:
            from transformers import get_linear_schedule_with_warmup
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config["warmup_steps"],
                num_training_steps=total_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        return optimizer

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log_dict({"test_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class DonutFinetuned:
    MODEL_REPO_ID = "ryanlinjui/donut-test"
    TASK_PROMPT_NAME = "<s_menu-text-detection>"

    processor = DonutProcessor.from_pretrained(MODEL_REPO_ID)
    pipe = pipeline(
        task="image-text-to-text",
        model=MODEL_REPO_ID,
        processor=processor
    )

    @classmethod
    def predict(cls, image: np.ndarray) -> dict:
        image = Image.fromarray(image)
        outputs = cls.pipe(text=cls.TASK_PROMPT_NAME, images=image)[0]["generated_text"]
        return cls.processor.token2json(outputs)

class DonutDatasets:
    """
    Modified from: 
        https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb
    
    Donut PyTorch Dataset Wrapper (supports train/validation/test splits)
        - Dynamic field names and JSON-to-token conversion
        - Returns PyTorch Datasets with __getitem__ producing tensors
        - Splits controlled by train_split/validation_split/test_split
        - Only single JSON annotation supported
        - Supports subscripting: datasets["train"], datasets["validation"], datasets["test"]
    Args:
        - datasets: DatasetDict containing train/validation/test splits
        - processor: DonutProcessor for image processing
        - image_column: Column name for images in the dataset
        - annotation_column: Column name for annotations in the dataset
        - task_start_token: Token to start the task
        - prompt_end_token: Token to end the prompt
        - max_length: Maximum length of tokenized sequences
        - train_split: Fraction of data to use for training (0.0-1.0)
        - validation_split: Fraction of data to use for validation (0.0-1.0)
        - test_split: Fraction of data to use for testing (0.0-1.0)
        - ignore_index: Index to ignore in labels (default: -100)
        - sort_json_key: Whether to sort JSON keys (default: True)
        - seed: Random seed for reproducibility. If None, use OS random seed (default: None)
        - shuffle: Whether to shuffle the dataset (default: True)
    Returns:
        - DonutDatasets object with train/validation/test splits
    Example:
        datasets = DonutDatasets(
            datasets=dataset_dict,
            processor=processor,
            image_column="image",
            annotation_column="annotation",
            task_start_token="<s_task>",
            prompt_end_token="<s_prompt>",
            max_length=512,
            train_split=0.8,
            validation_split=0.1,
            test_split=0.1
        )
        train_dataset = datasets["train"]
        validation_dataset = datasets["validation"]
        test_dataset = datasets["test"]
    Note:
        - The dataset must be a DatasetDict with train/validation/test splits
        - The processor must be a DonutProcessor instance
        - The image_column and annotation_column must exist in the dataset
        - The task_start_token and prompt_end_token must be unique tokens
        - The max_length should be set according to the model's maximum input length
        - The ignore_index is used for padding in labels (default: -100)
        - The sort_json_key option determines whether JSON keys are sorted or not
    """
    def __init__(
        self,
        datasets: DatasetDict,
        processor: DonutProcessor,
        image_column: str,
        annotation_column: str,
        task_start_token: str,
        prompt_end_token: str,
        image_size: tuple = (1280, 960),
        max_length: int = 512,
        train_split: float = 1.0,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        ignore_index: int = -100,
        sort_json_key: bool = True,
        seed: Optional[int] = None,
        shuffle: bool = True
    ):
        assert abs(train_split + validation_split + test_split - 1.0) < 1e-6, (
            "train/validation/test splits must sum to 1"
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_column = image_column
        self.annotation_column = annotation_column
        self.image_size = image_size
        self.max_length = max_length
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token or task_start_token
        self.ignore_index = ignore_index
        self.sort_json_key = sort_json_key

        # Resize image size by using the PIL
        resized_datasets = {}
        for split, dataset in datasets.items():
            resized_datasets[split] = dataset.map(
                lambda x: {"image": x[self.image_column].resize(self.image_size, Image.LANCZOS).convert("RGB")},
                remove_columns=[self.image_column],
                num_proc=4,
            )
        datasets = DatasetDict(resized_datasets)

        # Perform split on provided datasets
        raw = datasets
        parts: Dict[str, Any] = {}
        if train_split < 1.0:
            split1 = raw["train"].train_test_split(test_size=1 - train_split, seed=seed, shuffle=shuffle)
            parts["train"] = split1["train"]
            rest = split1["test"]
            if validation_split > 0:
                val_frac = validation_split / (validation_split + test_split)
                split2 = rest.train_test_split(test_size=1 - val_frac, seed=seed, shuffle=shuffle)
                parts["validation"] = split2["train"]
                parts["test"] = split2["test"]
            else:
                parts["test"] = rest
        else:
            parts = dict(raw)

        # Create individual split datasets
        self._splits: Dict[str, Dataset] = {}
        for name, ds in parts.items():
            self._splits[name] = _SplitDataset(
                hf_dataset=ds,
                processor=self.processor,
                image_column=self.image_column,
                annotation_column=self.annotation_column,
                max_length=self.max_length,
                ignore_index=self.ignore_index,
                sort_json_key=self.sort_json_key,
                task_start_token=self.task_start_token,
                prompt_end_token=self.prompt_end_token,
            )

    def __getitem__(self, split: str) -> Dataset:
        """
        Return the dataset split by name, e.g., datasets["train"]
        """
        if split in self._splits:
            return self._splits[split]
        raise KeyError(f"Unknown split '{split}'. Available splits: {list(self._splits.keys())}")

    def __repr__(self):
        return f"DonutDatasets(splits={list(self._splits.keys())})"


class _SplitDataset(Dataset):
    """
    PyTorch Dataset for a single split, returns (pixel_values, labels, target_sequence)
    """
    def __init__(
        self,
        hf_dataset,
        processor: DonutProcessor,
        image_column: str,
        annotation_column: str,
        max_length: int,
        ignore_index: int,
        sort_json_key: bool,
        task_start_token: str,
        prompt_end_token: str,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.annotation_column = annotation_column
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.sort_json_key = sort_json_key
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token

        # Prepare tokenized ground-truth sequences (single annotation)
        self.gt_token_sequences = []
        for sample in self.hf_dataset:
            gt = sample[self.annotation_column]
            if isinstance(gt, str):
                gt = json.loads(gt)
            # Don't include task start token in target sequence
            seq = self._json_to_token(gt) + self.tokenizer.eos_token
            self.gt_token_sequences.append(seq)

        # Add special tokens to tokenizer
        self.tokenizer.add_tokens([self.task_start_token, self.prompt_end_token])

    def _json_to_token(self, obj: Any) -> str:
        if isinstance(obj, dict):
            keys = sorted(obj.keys()) if self.sort_json_key else obj.keys()
            seq = ""
            for k in keys:
                open_tag = f"<s_{k}>"
                close_tag = f"</s_{k}>"
                self.tokenizer.add_special_tokens({"additional_special_tokens": [open_tag, close_tag]})
                seq += open_tag + self._json_to_token(obj[k]) + close_tag
            return seq
        if isinstance(obj, list):
            return r"<sep/>".join(self._json_to_token(x) for x in obj)
        return str(obj)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        sample = self.hf_dataset[idx]
        pixel_values = self.processor(sample[self.image_column], return_tensors="pt").pixel_values.squeeze()
        target_seq = self.gt_token_sequences[idx]
        tokens = self.tokenizer(
            target_seq,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": tokens.attention_mask.squeeze(0),
            "labels": labels,
            "target_sequence": target_seq
        }

def load_trained_model_from_checkpoint(checkpoint_path: str, config: dict, processor: DonutProcessor, model: VisionEncoderDecoderModel):
    """
    Load a trained PyTorch Lightning model from checkpoint
    
    Args:
        checkpoint_path: Path to the .ckpt file
        config: Configuration dictionary used during training
        processor: DonutProcessor instance
        model: Base VisionEncoderDecoderModel instance
    
    Returns:
        DonutModelPLModule: Loaded PyTorch Lightning module
    """
    pl_module = DonutModelPLModule.load_from_checkpoint(
        checkpoint_path, 
        config=config, 
        processor=processor, 
        model=model
    )
    return pl_module