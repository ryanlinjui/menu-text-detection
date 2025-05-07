import json
from typing import Any, Dict

import numpy as np
from PIL import Image
from datasets import DatasetDict
from torch.utils.data import Dataset
from transformers import pipeline, DonutProcessor

class DonutFinetuned:
    DEFAULT_PIPELINE = pipeline(
        task="image-to-text",
        model="naver-clova-ix/donut-base"
    )
    @classmethod
    def predict(cls, image: np.ndarray) -> dict:
        image = Image.fromarray(image)
        result = cls.DEFAULT_PIPELINE(image)
        return result

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
    """
    def __init__(
        self,
        datasets: DatasetDict,
        processor: DonutProcessor,
        image_column: str,
        annotation_column: str,
        task_start_token: str,
        prompt_end_token: str,
        max_length: int = 512,
        train_split: float = 1.0,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        ignore_index: int = -100,
        sort_json_key: bool = True
    ):
        assert abs(train_split + validation_split + test_split - 1.0) < 1e-6, (
            "train/validation/test splits must sum to 1"
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_column = image_column
        self.annotation_column = annotation_column
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.sort_json_key = sort_json_key
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token or task_start_token

        # Perform split on provided datasets
        raw = datasets
        parts: Dict[str, Any] = {}
        if train_split < 1.0:
            split1 = raw["train"].train_test_split(test_size=1 - train_split, seed=42)
            parts["train"] = split1["train"]
            rest = split1["test"]
            if validation_split > 0:
                val_frac = validation_split / (validation_split + test_split)
                split2 = rest.train_test_split(test_size=1 - val_frac, seed=42)
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
            "labels": labels
        }