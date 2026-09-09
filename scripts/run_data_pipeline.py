"""
data.py

This script runs the data preparation pipeline for the tool-calling
fine-tuning project.

The pipeline performs the following steps:

1. Load the xLAM dataset from Hugging Face
2. Create complexity groups based on the number of expected tool calls
3. Perform a grouped stratified train/validation/test split
4. Verify and display the resulting dataset sizes

The reusable dataset loading and preprocessing logic is implemented
in the src directory.
"""

import os
from dotenv import load_dotenv
from src.data import load_xlam_dataset
from config.environment import DATASET_NAME
from src.preprocessing import (
    create_complexity_group,
    create_train_val_test_split
)


# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()
from pathlib import Path


HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# MAIN DATA PIPELINE
# ============================================================

def main():

    #Load the dataset
    dataset = load_xlam_dataset(
        dataset_name=DATASET_NAME,
        token=HF_TOKEN
    )

    #Create complexity groups
    dataset_grouped = dataset["train"].map(
        create_complexity_group
    )

    #Create Dataset splits
    train_dataset, validation_dataset, test_dataset = (
        create_train_val_test_split(dataset_grouped)
    )

    #Save dataset splits

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data"
    train_path = output_dir / "train"
    validation_path = output_dir / "validation"
    test_path = output_dir / "test"

    train_dataset.save_to_disk(str(train_path))
    validation_dataset.save_to_disk(str(validation_path))
    test_dataset.save_to_disk(str(test_path))




if __name__ == "__main__":
    main()