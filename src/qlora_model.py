"""
QLoRA model utilities for the function-calling
fine-tuning project.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from config.environment import BASE_MODEL_NAME


# ============================================================
# BASE MODEL LOADING
# ============================================================

def load_qlora_base_model(
    model_name=BASE_MODEL_NAME,
):
    """
    Load the base Qwen model using 4-bit NF4 quantization.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.

    Returns
    -------
    model
        4-bit quantized language model.

    tokenizer
        Corresponding tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(
        model
    )

    return model, tokenizer


# ============================================================
# LoRA CONFIGURATION
# ============================================================

def apply_qlora(
    model,
    lora_config,
):
    """
    Apply LoRA adapters to the quantized base model.

    Parameters
    ----------
    model
        4-bit quantized base model.

    lora_config : dict
        LoRA configuration.

    Returns
    -------
    model
        PEFT model with LoRA adapters.
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
        100
        * trainable_parameters
        / total_parameters
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": trainable_percentage,
    }


# ============================================================
# TRAINABLE PARAMETER DTYPE
# ============================================================

def prepare_trainable_parameters(model):
    """
    Keep trainable LoRA parameters in FP32.

    This is a stability configuration used for the
    T4 training environment.
    """

    for parameter in model.parameters():

        if parameter.requires_grad:

            parameter.data = parameter.data.float()

    return model