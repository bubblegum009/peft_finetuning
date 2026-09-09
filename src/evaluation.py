"""
This module contains evaluation utilities for the tool-calling
fine-tuning project.

The module supports:

- Per-example prediction evaluation
- Overall evaluation metrics
- Complexity-based evaluation metrics
- MLflow metric logging

These utilities can be used for evaluating the base model,
LoRA fine-tuned models, and QLoRA fine-tuned models.
"""

import mlflow
import pandas as pd


# ============================================================
# PER-EXAMPLE EVALUATION
# ============================================================

def evaluate_prediction(
    expected_calls,
    predicted_calls
):
    """
    Evaluate predicted tool calls against expected tool calls.

    Args:
        expected_calls (list):
            Ground truth tool calls.

        predicted_calls (list):
            Tool calls predicted by the model.

    Returns:
        dict:
            Evaluation metrics for a single example.
    """

    exact_match = (
        expected_calls == predicted_calls
    )

    expected_call_count = len(
        expected_calls
    )

    predicted_call_count = len(
        predicted_calls
    )

    call_count_match = (
        expected_call_count
        == predicted_call_count
    )


    # --------------------------------------------------------
    # Tool name evaluation
    # --------------------------------------------------------

    expected_tool_names = [
        call.get("name")
        for call in expected_calls
    ]

    predicted_tool_names = [
        call.get("name")
        for call in predicted_calls
    ]

    tool_name_match = (
        expected_tool_names
        == predicted_tool_names
    )


    # --------------------------------------------------------
    # Argument evaluation
    # --------------------------------------------------------

    expected_arguments = [
        call.get("arguments")
        for call in expected_calls
    ]

    predicted_arguments = [
        call.get("arguments")
        for call in predicted_calls
    ]

    arguments_match = (
        expected_arguments
        == predicted_arguments
    )


    return {

        "exact_match":
            int(exact_match),

        "tool_name_match":
            int(tool_name_match),

        "arguments_match":
            int(arguments_match),

        "call_count_match":
            int(call_count_match),

        "expected_call_count":
            expected_call_count,

        "predicted_call_count":
            predicted_call_count
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    evaluation_results
):
    """
    Calculate aggregate evaluation metrics.

    Args:
        evaluation_results (list):
            List containing evaluation results.

    Returns:
        dict:
            Overall evaluation metrics.
    """

    results_df = pd.DataFrame(
        evaluation_results
    )


    total_examples = len(
        results_df
    )


    return {

        "total_examples":
            total_examples,

        "exact_match_accuracy":
            results_df["exact_match"].mean()
            * 100,

        "tool_name_accuracy":
            results_df["tool_name_match"].mean()
            * 100,

        "arguments_accuracy":
            results_df["arguments_match"].mean()
            * 100,

        "call_count_accuracy":
            results_df["call_count_match"].mean()
            * 100
    }


# ============================================================
# COMPLEXITY-BASED METRICS
# ============================================================

def calculate_complexity_metrics(
    evaluation_results
):
    """
    Calculate evaluation metrics grouped by query complexity.

    Args:
        evaluation_results (list):
            List containing evaluation results.

    Returns:
        pandas.DataFrame:
            Metrics grouped by complexity level.
    """

    results_df = pd.DataFrame(
        evaluation_results
    )


    complexity_metrics = (

        results_df

        .groupby(
            "complexity_group"
        )

        .agg(

            samples=(
                "exact_match",
                "count"
            ),

            exact_match_accuracy=(
                "exact_match",
                lambda x: x.mean() * 100
            ),

            tool_name_accuracy=(
                "tool_name_match",
                lambda x: x.mean() * 100
            ),

            arguments_accuracy=(
                "arguments_match",
                lambda x: x.mean() * 100
            ),

            call_count_accuracy=(
                "call_count_match",
                lambda x: x.mean() * 100
            )
        )

    )


    return complexity_metrics


# ============================================================
# MLFLOW LOGGING
# ============================================================

def log_metrics_to_mlflow(
    overall_metrics,
    complexity_metrics,
    model_type
):
    """
    Log evaluation metrics to MLflow.

    Args:
        overall_metrics (dict):
            Dictionary containing overall evaluation metrics.

        complexity_metrics (pd.DataFrame):
            Complexity-wise evaluation metrics.

        model_type (str):
            Name identifying the evaluated model.

            Example:
                "baseline"
                "lora"
                "qlora"
    """

    # --------------------------------------------------------
    # Log overall metrics
    # --------------------------------------------------------

    mlflow.log_metrics({

        f"{model_type}_{metric}":
            value

        for metric, value
        in overall_metrics.items()

        if metric != "total_examples"
    })


    # --------------------------------------------------------
    # Log number of evaluated examples
    # --------------------------------------------------------

    mlflow.log_param(

        f"{model_type}_total_examples",

        overall_metrics[
            "total_examples"
        ]
    )


    # --------------------------------------------------------
    # Log complexity metrics
    # --------------------------------------------------------

    for complexity_group, row in complexity_metrics.iterrows():

        complexity_name = str(
            complexity_group
        ).replace(
            "+",
            "_plus"
        )


        mlflow.log_metrics({

            (
                f"{model_type}_"
                f"{complexity_name}_"
                f"{metric}"
            ):
                float(value)

            for metric, value
            in row.items()

            if metric != "samples"
        })


        mlflow.log_param(

            (
                f"{model_type}_"
                f"{complexity_name}_samples"
            ),

            int(
                row["samples"]
            )
        )