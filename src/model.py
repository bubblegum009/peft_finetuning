"""
model.py

Model loading and LoRA configuration utilities for the
function-calling fine-tuning project.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from config.environment import BASE_MODEL_NAME


# ============================================================
# BASE MODEL LOADING
# ============================================================

def load_base_model(
    model_name=BASE_MODEL_NAME,
):
    """
    Load the base language model and tokenizer.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.

    Returns
    -------
    model
        Loaded language model.

    tokenizer
        Corresponding tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # Required for batch padding during training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return model, tokenizer


# ============================================================
# LORA CONFIGURATION
# ============================================================

def apply_lora(
    model,
    lora_config,
):
    """
    Apply LoRA adapters to the base language model.

    Parameters
    ----------
    model
        Pretrained causal language model.

    lora_config : dict
        Dictionary containing LoRA configuration values.

    Returns
    -------
    model
        PEFT model with LoRA adapters applied.
    """

    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config["target_modules"],
    )

    model = get_peft_model(
        model,
        config,
    )

    return model


# ============================================================
# MODEL PARAMETER UTILITIES
# ============================================================

def get_parameter_counts(model):
    """
    Calculate total and trainable model parameters.

    Parameters
    ----------
    model
        PyTorch model.

    Returns
    -------
    dict
        Dictionary containing total parameters,
        trainable parameters, and trainable percentage.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_percentage = (
        100 * trainable_parameters / total_parameters
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": trainable_percentage,
    }