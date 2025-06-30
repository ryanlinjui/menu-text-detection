from typing import Optional

from datasets import Dataset, DatasetDict

def split_dataset(
    dataset: Dataset,
    train: float,
    validation: float,
    test: float,
    seed: Optional[int] = None
) -> DatasetDict:
    """
    Split a single-split Hugging Face Dataset into train/validation/test subsets.

    Args:
        dataset (Dataset): The input dataset (e.g. load_dataset(...)['train']).
        train (float): Proportion of data for the train split (0 < train < 1).
        val (float): Proportion of data for the validation split (0 < val < 1).
        test (float): Proportion of data for the test split (0 < test < 1).
                            Must satisfy train + val + test == 1.0.
        seed (int): Random seed for reproducibility (default: None).

    Returns:
        DatasetDict: A dictionary with keys "train", "validation", and "test".
    """
    # Verify ratios sum to 1.0
    total = train + validation + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train + validation + test must equal 1.0 (got {total})")

    # First split: extract train vs. temp (validation + test)
    temp_size = validation + test
    split_1 = dataset.train_test_split(test_size=temp_size, seed=seed)
    train_ds = split_1["train"]
    temp_ds  = split_1["test"]

    # Second split: divide temp into validation vs. test
    relative_test_size = test / temp_size
    split_2 = temp_ds.train_test_split(test_size=relative_test_size, seed=seed)
    validation_ds  = split_2["train"]
    test_ds = split_2["test"]

    # Return a DatasetDict with all three splits
    return DatasetDict({
        "train":      train_ds,
        "validation": validation_ds,
        "test":       test_ds,
    })