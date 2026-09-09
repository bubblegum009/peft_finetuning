"""
QLoRA fine-tuning pipeline for function calling.

Pipeline:
    1. Load configuration
    2. Set random seed
    3. Load representative training/validation datasets
    4. Load 4-bit Qwen base model
    5. Prepare model for k-bit training
    6. Apply LoRA adapters
    7. Configure SFTTrainer
    8. Train the model
    9. Evaluate validation set
    10. Track experiment using MLflow
    11. Save final QLoRA adapter

Usage:
    python scripts/train_qlora.py
"""

import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.preprocessing import format_dataset_for_sft

from src.qlora_model import (
    load_qlora_base_model,
    apply_qlora,
    get_parameter_counts,
    prepare_trainable_parameters,
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
# CONFIGURATION
# ============================================================

def load_config(config_path):
    """
    Load QLoRA configuration from YAML.
    """

    config_path = Path(config_path)

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_training_datasets():

    train_path = (
        PROJECT_ROOT
        / "data"
        / "representative_qlora"
        / "train"
    )

    validation_path = (
        PROJECT_ROOT
        / "data"
        / "representative_qlora"
        / "validation"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not validation_path.exists():

        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{validation_path}"
        )

    train_dataset = load_from_disk(
        str(train_path)
    )

    validation_dataset = load_from_disk(
        str(validation_path)
    )

    formatted_train = format_dataset_for_sft(
        train_dataset
    )

    formatted_validation = format_dataset_for_sft(
        validation_dataset
    )

    if len(formatted_train) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(formatted_validation) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    return (
        formatted_train,
        formatted_validation,
    )


# ============================================================
# MODEL PREPARATION
# ============================================================

def prepare_model(config):

    model_name = config["model"]["name"]

    model, tokenizer = load_qlora_base_model(
        model_name=model_name
    )

    model = apply_qlora(
        model=model,
        lora_config=config["lora"],
    )

    model = prepare_trainable_parameters(
        model
    )

    parameter_counts = get_parameter_counts(
        model
    )

    return (
        model,
        tokenizer,
        parameter_counts,
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

def create_training_config(config):

    training = config["training"]

    output_directory = (
        PROJECT_ROOT
        / config["output"]["output_dir"]
    )

    return SFTConfig(

        output_dir=str(
            output_directory
        ),

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

        logging_steps=training[
            "logging_steps"
        ],

        eval_strategy=training[
            "evaluation_strategy"
        ],

        eval_steps=training[
            "eval_steps"
        ],

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

        fp16=False,
        bf16=False,

        gradient_checkpointing=training[
            "gradient_checkpointing"
        ],

        optim="paged_adamw_8bit",

        max_length=config["data"][
            "max_length"
        ],

        completion_only_loss=training[
            "completion_only_loss"
        ],

        packing=training[
            "packing"
        ],

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

    return {

        "experiment_name":
            config["experiment"]["name"],

        "seed":
            config["experiment"]["seed"],

        "base_model":
            config["model"]["name"],

        "model_type":
            "qlora",

        "train_samples":
            train_size,

        "validation_samples":
            validation_size,

        "max_length":
            config["data"]["max_length"],

        # Quantization
        "load_in_4bit":
            config["quantization"]["load_in_4bit"],

        "quant_type":
            config["quantization"]["quant_type"],

        "double_quant":
            config["quantization"]["use_double_quant"],

        "compute_dtype":
            config["quantization"]["compute_dtype"],

        # LoRA
        "lora_r":
            config["lora"]["r"],

        "lora_alpha":
            config["lora"]["lora_alpha"],

        "lora_dropout":
            config["lora"]["lora_dropout"],

        "target_modules":
            ",".join(
                config["lora"]["target_modules"]
            ),

        # Training
        "epochs":
            config["training"]["num_train_epochs"],

        "train_batch_size":
            config["training"][
                "per_device_train_batch_size"
            ],

        "eval_batch_size":
            config["training"][
                "per_device_eval_batch_size"
            ],

        "gradient_accumulation_steps":
            config["training"][
                "gradient_accumulation_steps"
            ],

        "learning_rate":
            config["training"]["learning_rate"],

        "optimizer":
            "paged_adamw_8bit",

        # Model parameters
        "total_parameters":
            parameter_counts["total_parameters"],

        "trainable_parameters":
            parameter_counts["trainable_parameters"],

        "trainable_percentage":
            parameter_counts["trainable_percentage"],
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_qlora(config):

    seed = config["experiment"]["seed"]

    set_reproducibility(seed)

    (
        train_dataset,
        validation_dataset,
    ) = prepare_training_datasets()

    (
        model,
        tokenizer,
        parameter_counts,
    ) = prepare_model(config)

    training_arguments = (
        create_training_config(config)
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    # Re-apply FP32 to trainable LoRA parameters
    # after trainer initialization.
    for name, parameter in (
        trainer.model.named_parameters()
    ):

        if parameter.requires_grad:

            parameter.data = parameter.data.float()

    setup_mlflow()

    mlflow_parameters = (
        build_mlflow_parameters(
            config=config,
            parameter_counts=parameter_counts,
            train_size=len(train_dataset),
            validation_size=len(validation_dataset),
        )
    )

    run_name = config["mlflow"]["run_name"]

    with mlflow.start_run(
        run_name=run_name
    ):

        log_parameters(
            mlflow_parameters
        )

        training_result = (
            trainer.train()
        )

        train_metrics = (
            training_result.metrics
        )

        evaluation_metrics = (
            trainer.evaluate()
        )

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

        mlflow.log_artifacts(
            str(final_model_directory),
            artifact_path="qlora_adapter",
        )


# ============================================================
# MAIN
# ============================================================

def main():

    config_path = (
        PROJECT_ROOT
        / "config"
        / "qlora.yaml"
    )

    config = load_config(
        config_path
    )

    train_qlora(
        config
    )


if __name__ == "__main__":

    main()