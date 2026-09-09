"""
preprocessing.py

Dataset preprocessing utilities for the xLAM function-calling dataset.

Responsibilities:
    - Parse expected tool calls
    - Create complexity groups based on the number of tool calls
    - Create stratified train, validation, and test splits
    - Create representative subsets for experimentation
    - Format examples into prompt/completion format for SFTTrainer

Tokenization and completion-only loss masking are intentionally NOT
performed here. Those responsibilities are delegated to SFTTrainer.
"""

import json

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# COMPLEXITY GROUPING
# ============================================================

def get_complexity_group(example):
    """
    Assign a complexity group based on the number of expected
    tool calls in the example.

    Groups:
        - 1_call
        - 2_calls
        - 3_calls
        - 4_plus_calls

    Parameters
    ----------
    example : dict
        Dataset example containing the 'answers' field.

    Returns
    -------
    dict
        Dictionary containing the complexity group.
    """

    answers = json.loads(example["answers"])
    num_calls = len(answers)

    if num_calls == 1:
        complexity_group = "1_call"

    elif num_calls == 2:
        complexity_group = "2_calls"

    elif num_calls == 3:
        complexity_group = "3_calls"

    else:
        complexity_group = "4_plus_calls"

    return {
        "complexity_group": complexity_group
    }


def add_complexity_groups(dataset):
    """
    Add a complexity_group column to the dataset.

    Parameters
    ----------
    dataset
        Hugging Face Dataset.

    Returns
    -------
    Dataset
        Dataset containing the additional complexity_group column.
    """

    return dataset.map(get_complexity_group)


# ============================================================
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_stratified_splits(
    dataset,
    random_state=42,
):
    """
    Create stratified train, validation, and test splits.

    Split proportions:
        - Train: 80%
        - Validation: 10%
        - Test: 10%

    Stratification is performed using the complexity_group column.

    Parameters
    ----------
    dataset
        Hugging Face Dataset containing the complexity_group column.

    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
            - train
            - validation
            - test
    """

    indices = np.arange(len(dataset))

    complexity_groups = np.array(
        dataset["complexity_group"]
    )

    # --------------------------------------------------------
    # FIRST SPLIT
    # --------------------------------------------------------
    # 80% Train
    # 20% Temporary (Validation + Test)
    # --------------------------------------------------------

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=complexity_groups,
    )

    # --------------------------------------------------------
    # SECOND SPLIT
    # --------------------------------------------------------
    # Temporary set is 20% of the full dataset.
    #
    # Split equally:
    # 10% Validation
    # 10% Test
    # --------------------------------------------------------

    temp_complexity_groups = complexity_groups[
        temp_indices
    ]

    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_complexity_groups,
    )

    return {
        "train": dataset.select(
            train_indices.tolist()
        ),

        "validation": dataset.select(
            validation_indices.tolist()
        ),

        "test": dataset.select(
            test_indices.tolist()
        ),
    }


# ============================================================
# REPRESENTATIVE SAMPLING
# ============================================================

def create_representative_sample(
    dataset,
    sample_size,
    random_state=42,
):
    """
    Create a representative sample stratified by complexity group.

    This preserves the approximate distribution of:
        - 1_call
        - 2_calls
        - 3_calls
        - 4_plus_calls

    Parameters
    ----------
    dataset
        Hugging Face Dataset containing the complexity_group column.

    sample_size : int
        Number of examples to include in the representative sample.

    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    Dataset
        Stratified representative sample.
    """

    if sample_size > len(dataset):

        raise ValueError(
            f"Sample size ({sample_size}) cannot be larger "
            f"than dataset size ({len(dataset)})."
        )

    indices = np.arange(
        len(dataset)
    )

    complexity_groups = np.array(
        dataset["complexity_group"]
    )

    # --------------------------------------------------------
    # Stratified sampling
    #
    # train_test_split is used to select exactly sample_size
    # representative examples.
    # --------------------------------------------------------

    _, selected_indices = train_test_split(
        indices,
        test_size=sample_size,
        random_state=random_state,
        stratify=complexity_groups,
    )

    return dataset.select(
        selected_indices.tolist()
    )


# ============================================================
# SFT PROMPT / COMPLETION FORMATTING
# ============================================================

SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
"""


def format_tool_call_example(example):
    """
    Convert an xLAM dataset example into prompt/completion format.

    The output format is compatible with prompt-completion based
    supervised fine-tuning.

    The resulting structure is:

        {
            "prompt": "...",
            "completion": "..."
        }

    Parameters
    ----------
    example : dict
        Raw xLAM dataset example containing:
            - query
            - tools
            - answers

    Returns
    -------
    dict
        Dictionary containing prompt and completion.
    """

    # --------------------------------------------------------
    # PARSE JSON FIELDS
    # --------------------------------------------------------

    tools = json.loads(
        example["tools"]
    )

    answers = json.loads(
        example["answers"]
    )

    # --------------------------------------------------------
    # FORMAT TOOL DEFINITIONS
    # --------------------------------------------------------

    tools_text = "\n".join(
        json.dumps(
            tool,
            ensure_ascii=False,
        )
        for tool in tools
    )

    # --------------------------------------------------------
    # FORMAT EXPECTED TOOL CALLS
    # --------------------------------------------------------

    completion_parts = []

    for answer in answers:

        tool_call = (
            "<tool_call>\n"
            + json.dumps(
                answer,
                ensure_ascii=False,
            )
            + "\n</tool_call>"
        )

        completion_parts.append(
            tool_call
        )

    completion = (
        "\n".join(completion_parts)
        + "\n<|im_end|>"
    )

    # --------------------------------------------------------
    # FORMAT PROMPT USING QWEN CHAT TEMPLATE
    # --------------------------------------------------------

    prompt = (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<tools>\n"
        + tools_text
        + "\n</tools>\n\n"
        + "For each function call, return a json object with "
        + "function name and arguments within "
        + "<tool_call></tool_call> XML tags:\n"
        + "<tool_call>\n"
        + '{"name": <function-name>, '
        + '"arguments": <args-json-object>}\n'
        + "</tool_call>"
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + example["query"]
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )

    return {
        "prompt": prompt,
        "completion": completion,
    }


# ============================================================
# FORMAT DATASET FOR SUPERVISED FINE-TUNING
# ============================================================

def format_dataset_for_sft(dataset):
    """
    Format an xLAM dataset into prompt/completion pairs for
    supervised fine-tuning.

    The returned dataset contains only:

        - prompt
        - completion

    Tokenization and label creation are handled later by
    SFTTrainer.

    Parameters
    ----------
    dataset
        Hugging Face Dataset containing raw xLAM examples.

    Returns
    -------
    Dataset
        Formatted dataset containing prompt and completion columns.
    """

    return dataset.map(
        format_tool_call_example,
        remove_columns=dataset.column_names,
    )


# ============================================================
# DATASET PREPARATION PIPELINE
# ============================================================

def prepare_dataset_splits(
    dataset,
    random_state=42,
):
    """
    Prepare the complete stratified dataset splits.

    Pipeline:

        Raw dataset
            ↓
        Add complexity groups
            ↓
        Stratified splitting
            ↓
        Train / Validation / Test

    Parameters
    ----------
    dataset
        Raw Hugging Face Dataset.

    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
            - train
            - validation
            - test
    """

    # --------------------------------------------------------
    # ADD COMPLEXITY GROUPS
    # --------------------------------------------------------

    dataset = add_complexity_groups(
        dataset
    )

    # --------------------------------------------------------
    # CREATE STRATIFIED SPLITS
    # --------------------------------------------------------

    splits = create_stratified_splits(
        dataset,
        random_state=random_state,
    )

    return splits