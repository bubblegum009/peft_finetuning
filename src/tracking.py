"""
This module contains MLflow experiment tracking utilities for the
tool-calling fine-tuning project.

The module supports:

- MLflow experiment configuration
- Parameter logging
- Metric logging
- Artifact logging
"""

import math
from pathlib import Path

import mlflow


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

def setup_mlflow(
    tracking_uri,
    experiment_name
):
    """
    Configure the MLflow tracking URI and experiment.

    Args:
        tracking_uri (str):
            Location where MLflow tracking data is stored.

        experiment_name (str):
            Name of the MLflow experiment.

    Returns:
        str:
            Experiment ID.
    """

    mlflow.set_tracking_uri(
        tracking_uri
    )

    experiment = mlflow.set_experiment(
        experiment_name
    )

    return experiment.experiment_id


# ============================================================
# PARAMETER LOGGING
# ============================================================

def log_parameters(
    parameters
):
    """
    Log experiment parameters to MLflow.

    Args:
        parameters (dict):
            Dictionary containing experiment parameters.
    """

    mlflow.log_params(
        parameters
    )


# ============================================================
# METRIC LOGGING
# ============================================================

def log_metrics(
    metrics
):
    """
    Log valid numeric metrics to MLflow.

    NaN and infinite values are excluded because MLflow metrics
    should contain valid finite numeric values.

    Args:
        metrics (dict):
            Dictionary containing metric values.
    """

    numeric_metrics = {

        key: float(value)

        for key, value
        in metrics.items()

        if isinstance(
            value,
            (int, float)
        )
        and math.isfinite(
            float(value)
        )
    }

    if numeric_metrics:

        mlflow.log_metrics(
            numeric_metrics
        )


# ============================================================
# ARTIFACT LOGGING
# ============================================================

def log_artifact(
    file_path
):
    """
    Log a file as an MLflow artifact.

    Args:
        file_path (str or Path):
            Path to the file.
    """

    file_path = Path(
        file_path
    )

    if file_path.exists():

        mlflow.log_artifact(
            str(file_path)
        )


def log_artifacts(
    file_paths
):
    """
    Log multiple files as MLflow artifacts.

    Args:
        file_paths (list):
            List containing file paths.
    """

    for file_path in file_paths:

        log_artifact(
            file_path
        )