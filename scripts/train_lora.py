"""
train_lora.py

LoRA fine-tuning pipeline for function calling.

Pipeline:
    1. Load configuration
    2. Set random seed
    3. Load xLAM dataset
    4. Create complexity groups
    5. Create stratified train/validation/test splits
    6. Create representative train and validation datasets
    7. Format datasets for supervised fine-tuning
    8. Load Qwen base model
    9. Apply LoRA adapters
    10. Configure SFTTrainer
    11. Train and evaluate the model
    12. Track experiment using MLflow
    13. Save the final LoRA adapter

Usage:
    python scripts/train_lora.py
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import mlflow

from transformers import set_seed

from trl import (
    SFTConfig,
    SFTTrainer,
)


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data import load_xlam_dataset

from src.preprocessing import (
    prepare_dataset_splits,
    create_representative_sample,
    format_dataset_for_sft,
)

from src.model import (
    load_base_model,
    apply_lora,
    get_parameter_counts,
)

from src.tracking import (
    setup_mlflow,
    log_parameters,
    log_metrics,
)

from config.environment import (
    PROJECT_ROOT,
    BASE_MODEL_NAME,
)


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    """
    Configure application logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path):
    """
    Load LoRA experiment configuration from YAML.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Experiment configuration.
    """

    config_path = Path(config_path)

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    logger.info(
        "Configuration loaded from: %s",
        config_path,
    )

    return config


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility(seed):
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    logger.info(
        "Random seed set to %d",
        seed,
    )


# ============================================================
# GPU INFORMATION
# ============================================================

def log_gpu_information():
    """
    Log GPU availability and hardware information.

    Returns
    -------
    dict
        GPU information.
    """

    logger.info("=" * 70)
    logger.info("GPU INFORMATION")
    logger.info("=" * 70)

    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
    }

    logger.info(
        "CUDA available: %s",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        gpu_memory_gb = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3)
        )

        gpu_info["gpu_name"] = gpu_name
        gpu_info["gpu_memory_gb"] = gpu_memory_gb

        logger.info(
            "GPU: %s",
            gpu_name,
        )

        logger.info(
            "GPU memory: %.2f GB",
            gpu_memory_gb,
        )

    return gpu_info


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_training_datasets(
    config,
):
    """
    Load and prepare representative training and validation datasets.

    Pipeline:
        Raw dataset
            ↓
        Complexity grouping
            ↓
        Stratified 80/10/10 split
            ↓
        Representative sampling
            ↓
        Prompt/completion formatting

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    tuple
        formatted_train_dataset,
        formatted_validation_dataset
    """

    logger.info("=" * 70)
    logger.info("LOADING DATASET")
    logger.info("=" * 70)

    dataset = load_xlam_dataset()

    # xLAM contains the train split
    raw_dataset = dataset["train"]

    logger.info(
        "Total dataset examples: %d",
        len(raw_dataset),
    )

    # --------------------------------------------------------
    # CREATE STRATIFIED SPLITS
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("CREATING STRATIFIED SPLITS")
    logger.info("=" * 70)

    seed = config["experiment"]["seed"]

    splits = prepare_dataset_splits(
        dataset=raw_dataset,
        random_state=seed,
    )

    train_dataset = splits["train"]

    validation_dataset = splits["validation"]

    test_dataset = splits["test"]

    logger.info(
        "Train examples: %d",
        len(train_dataset),
    )

    logger.info(
        "Validation examples: %d",
        len(validation_dataset),
    )

    logger.info(
        "Test examples: %d",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # CREATE REPRESENTATIVE SUBSETS
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("CREATING REPRESENTATIVE DATASETS")
    logger.info("=" * 70)

    train_samples = config["data"]["train_samples"]

    validation_samples = config["data"][
        "validation_samples"
    ]

    representative_train = (
        create_representative_sample(
            dataset=train_dataset,
            sample_size=train_samples,
            random_state=seed,
        )
    )

    representative_validation = (
        create_representative_sample(
            dataset=validation_dataset,
            sample_size=validation_samples,
            random_state=seed,
        )
    )

    logger.info(
        "Representative training examples: %d",
        len(representative_train),
    )

    logger.info(
        "Representative validation examples: %d",
        len(representative_validation),
    )

    # --------------------------------------------------------
    # FORMAT FOR SFT
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("FORMATTING DATASETS FOR SFT")
    logger.info("=" * 70)

    formatted_train = format_dataset_for_sft(
        representative_train
    )

    formatted_validation = format_dataset_for_sft(
        representative_validation
    )

    logger.info(
        "Formatted training columns: %s",
        formatted_train.column_names,
    )

    logger.info(
        "Formatted validation columns: %s",
        formatted_validation.column_names,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(formatted_train) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(formatted_validation) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    required_columns = {
        "prompt",
        "completion",
    }

    if not required_columns.issubset(
        formatted_train.column_names
    ):

        raise ValueError(
            "Training dataset does not contain "
            "prompt and completion columns."
        )

    logger.info(
        "Dataset preparation completed successfully."
    )

    return (
        formatted_train,
        formatted_validation,
    )


# ============================================================
# MODEL PREPARATION
# ============================================================

def prepare_model(
    config,
):
    """
    Load the base model and apply LoRA adapters.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    tuple
        model,
        tokenizer,
        parameter_counts
    """

    logger.info("=" * 70)
    logger.info("LOADING BASE MODEL")
    logger.info("=" * 70)

    model_name = config["model"]["name"]

    model, tokenizer = load_base_model(
        model_name=model_name
    )

    logger.info(
        "Base model loaded: %s",
        model_name,
    )

    # --------------------------------------------------------
    # APPLY LORA
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("APPLYING LORA")
    logger.info("=" * 70)

    model = apply_lora(
        model=model,
        lora_config=config["lora"],
    )

    # --------------------------------------------------------
    # PARAMETER COUNTS
    # --------------------------------------------------------

    parameter_counts = get_parameter_counts(
        model
    )

    logger.info(
        "Total parameters: %s",
        f"{parameter_counts['total_parameters']:,}",
    )

    logger.info(
        "Trainable parameters: %s",
        f"{parameter_counts['trainable_parameters']:,}",
    )

    logger.info(
        "Trainable percentage: %.4f%%",
        parameter_counts[
            "trainable_percentage"
        ],
    )

    return (
        model,
        tokenizer,
        parameter_counts,
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

def create_training_config(
    config,
):
    """
    Create the SFTConfig used by SFTTrainer.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    SFTConfig
        Training configuration.
    """

    training = config["training"]

    output_directory = (
        PROJECT_ROOT
        / config["output"]["output_dir"]
    )

    return SFTConfig(

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_dir=str(
            output_directory
        ),

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        num_train_epochs=training[
            "num_train_epochs"
        ],

        per_device_train_batch_size=training[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=training[
            "per_device_eval_batch_size"
        ],

        gradient_accumulation_steps=training[
            "gradient_accumulation_steps"
        ],

        learning_rate=training[
            "learning_rate"
        ],

        warmup_ratio=training[
            "warmup_ratio"
        ],

        weight_decay=training[
            "weight_decay"
        ],

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        logging_steps=training[
            "logging_steps"
        ],

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        eval_strategy=training[
            "evaluation_strategy"
        ],

        eval_steps=training[
            "eval_steps"
        ],

        # ----------------------------------------------------
        # CHECKPOINTING
        # ----------------------------------------------------

        save_strategy=training[
            "save_strategy"
        ],

        save_steps=training[
            "save_steps"
        ],

        save_total_limit=training[
            "save_total_limit"
        ],

        load_best_model_at_end=training[
            "load_best_model_at_end"
        ],

        metric_for_best_model=training[
            "metric_for_best_model"
        ],

        greater_is_better=training[
            "greater_is_better"
        ],

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        fp16=training[
            "fp16"
        ],

        # ----------------------------------------------------
        # SEQUENCE LENGTH
        # ----------------------------------------------------

        max_length=config["data"][
            "max_length"
        ],

        # ----------------------------------------------------
        # COMPLETION-ONLY TRAINING
        # ----------------------------------------------------

        completion_only_loss=True,

        # ----------------------------------------------------
        # REPORTING
        # ----------------------------------------------------

        report_to="none",
    )


# ============================================================
# MLFLOW PARAMETERS
# ============================================================

def build_mlflow_parameters(
    config,
    parameter_counts,
    train_size,
    validation_size,
):
    """
    Build experiment parameters for MLflow logging.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    parameter_counts : dict
        Model parameter statistics.

    train_size : int
        Number of training examples.

    validation_size : int
        Number of validation examples.

    Returns
    -------
    dict
        Parameters to log in MLflow.
    """

    parameters = {

        # Experiment
        "experiment_name": config[
            "experiment"
        ]["name"],

        "seed": config[
            "experiment"
        ]["seed"],

        # Model
        "base_model": config[
            "model"
        ]["name"],

        # Dataset
        "train_samples": train_size,

        "validation_samples": validation_size,

        "max_length": config[
            "data"
        ]["max_length"],

        # LoRA
        "lora_r": config["lora"]["r"],

        "lora_alpha": config[
            "lora"
        ]["lora_alpha"],

        "lora_dropout": config[
            "lora"
        ]["lora_dropout"],

        "target_modules": ",".join(
            config["lora"]["target_modules"]
        ),

        # Training
        "epochs": config["training"][
            "num_train_epochs"
        ],

        "train_batch_size": config["training"][
            "per_device_train_batch_size"
        ],

        "eval_batch_size": config["training"][
            "per_device_eval_batch_size"
        ],

        "gradient_accumulation_steps": (
            config["training"][
                "gradient_accumulation_steps"
            ]
        ),

        "learning_rate": config[
            "training"
        ]["learning_rate"],

        # Model parameters
        "total_parameters": parameter_counts[
            "total_parameters"
        ],

        "trainable_parameters": parameter_counts[
            "trainable_parameters"
        ],

        "trainable_percentage": parameter_counts[
            "trainable_percentage"
        ],
    }

    return parameters


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_lora(
    config,
):
    """
    Execute the complete LoRA fine-tuning pipeline.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    """

    logger.info("=" * 70)
    logger.info("STARTING LORA FINE-TUNING")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed = config["experiment"]["seed"]

    set_reproducibility(seed)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_info = log_gpu_information()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
    ) = prepare_training_datasets(
        config
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    (
        model,
        tokenizer,
        parameter_counts,
    ) = prepare_model(
        config
    )

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    training_arguments = create_training_config(
        config
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("INITIALIZING SFT TRAINER")
    logger.info("=" * 70)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "SFTTrainer initialized successfully."
    )

    # --------------------------------------------------------
    # MLFLOW SETUP
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("SETTING UP MLFLOW")
    logger.info("=" * 70)

    setup_mlflow()

    mlflow_parameters = (
        build_mlflow_parameters(
            config=config,
            parameter_counts=parameter_counts,
            train_size=len(train_dataset),
            validation_size=len(validation_dataset),
        )
    )

    # --------------------------------------------------------
    # START EXPERIMENT
    # --------------------------------------------------------

    run_name = config["mlflow"]["run_name"]

    with mlflow.start_run(
        run_name=run_name
    ):

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        log_parameters(
            mlflow_parameters
        )

        # GPU information
        mlflow.log_params(
            {
                key: str(value)
                for key, value
                in gpu_info.items()
            }
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        logger.info("=" * 70)
        logger.info("TRAINING STARTED")
        logger.info("=" * 70)

        training_result = trainer.train()

        # ----------------------------------------------------
        # FINAL TRAINING METRICS
        # ----------------------------------------------------

        train_metrics = (
            training_result.metrics
        )

        logger.info(
            "Training completed."
        )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        logger.info("=" * 70)
        logger.info("FINAL VALIDATION")
        logger.info("=" * 70)

        evaluation_metrics = trainer.evaluate()

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        final_metrics = {}

        final_metrics.update(
            train_metrics
        )

        final_metrics.update(
            evaluation_metrics
        )

        log_metrics(
            final_metrics
        )

        # ----------------------------------------------------
        # SAVE FINAL LORA ADAPTER
        # ----------------------------------------------------

        logger.info("=" * 70)
        logger.info("SAVING FINAL LORA ADAPTER")
        logger.info("=" * 70)

        final_model_directory = (
            PROJECT_ROOT
            / config["output"][
                "final_model_dir"
            ]
        )

        final_model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        trainer.save_model(
            str(final_model_directory)
        )

        tokenizer.save_pretrained(
            str(final_model_directory)
        )

        # ----------------------------------------------------
        # LOG FINAL MODEL ARTIFACT
        # ----------------------------------------------------

        mlflow.log_artifacts(
            str(final_model_directory),
            artifact_path="lora_adapter",
        )

        logger.info(
            "Final LoRA adapter saved to: %s",
            final_model_directory,
        )

        # ----------------------------------------------------
        # LOG RUN INFORMATION
        # ----------------------------------------------------

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        logger.info(
            "MLflow Run ID: %s",
            mlflow.active_run().info.run_id,
        )

        logger.info(
            "Final evaluation metrics:"
        )

        for metric_name, metric_value in (
            evaluation_metrics.items()
        ):

            logger.info(
                "%s: %s",
                metric_name,
                metric_value,
            )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Application entry point.
    """

    config_path = (
        PROJECT_ROOT
        / "config"
        / "lora.yaml"
    )

    config = load_config(
        config_path
    )

    train_lora(
        config
    )


if __name__ == "__main__":

    main()