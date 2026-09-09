"""
This module handles dataset-related operations for the tool-calling

- Loading the xLAM function-calling dataset from Hugging Face
- Managing dataset access and dataset-related configuration
- Loading saved dataset splits when required

This module focuses only on obtaining and managing the dataset.
Dataset preprocessing, transformation, and splitting logic are handled
separately in preprocessing.py.
"""

from datasets import load_dataset


def load_xlam_dataset(dataset_name, token=None):
    """
    Load a function-calling dataset from Hugging Face.

    Args:
        dataset_name (str):
            Name of the dataset on Hugging Face.

        token (str, optional):
            Hugging Face authentication token required for
            gated or private datasets.

    Returns:
        DatasetDict:
            The loaded Hugging Face dataset.
    """

    dataset = load_dataset(
        dataset_name,
        token=token
    )

    return dataset