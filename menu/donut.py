import json
from typing import Any, Dict, Optional

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