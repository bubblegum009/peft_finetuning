"""
This script runs baseline evaluation for the tool-calling
fine-tuning project.

The pipeline performs the following steps:

1. Configure MLflow experiment tracking
2. Load the preprocessed test dataset
3. Monitor initial system resource usage
4. Load the base Qwen model
5. Measure model loading time
6. Run inference on the evaluation dataset
7. Compare predicted tool calls with expected tool calls
8. Calculate overall evaluation metrics
9. Calculate performance metrics by complexity group
10. Save evaluation results
11. Log parameters, metrics, system metrics, and artifacts to MLflow
"""

import json

import mlflow
import pandas as pd
from datasets import load_from_disk

from config.environment import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    BASELINE_NUM_SAMPLES,
    TEST_DATA_PATH,
    RESULTS_DIR,
    BASELINE_RESULTS_FILE,
    BASELINE_COMPLEXITY_FILE,
    BASELINE_METRICS_FILE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

from src.model import load_base_model

from src.inference import predict_tool_calls

from src.evaluation import (
    evaluate_prediction,
    calculate_overall_metrics,
    calculate_complexity_metrics,
)

from src.monitoring import (
    start_timer,
    get_elapsed_time,
    get_system_metrics,
)

from src.tracking import (
    setup_mlflow,
    log_parameters,
    log_metrics,
    log_artifacts,
)


# ============================================================
# MAIN BASELINE PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    setup_mlflow(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )


    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = load_from_disk(
        str(TEST_DATA_PATH)
    )


    # --------------------------------------------------------
    # Select evaluation samples
    # --------------------------------------------------------

    evaluation_dataset = test_dataset.select(
        range(
            min(
                BASELINE_NUM_SAMPLES,
                len(test_dataset),
            )
        )
    )


    # ========================================================
    # START MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        run_name="baseline_qwen"
    ):


        # ----------------------------------------------------
        # Log experiment parameters
        # ----------------------------------------------------

        log_parameters({

            "model_name":
                MODEL_NAME,

            "model_type":
                "baseline",

            "max_new_tokens":
                MAX_NEW_TOKENS,

            "evaluation_samples":
                len(evaluation_dataset),
        })


        # ----------------------------------------------------
        # Capture initial system metrics
        # ----------------------------------------------------

        initial_system_metrics = (
            get_system_metrics()
        )

        initial_metrics = {

            f"initial_{key}":
                value

            for key, value
            in initial_system_metrics.items()
        }

        log_metrics(
            initial_metrics
        )


        # ----------------------------------------------------
        # Load model and measure loading time
        # ----------------------------------------------------

        model_loading_start_time = (
            start_timer()
        )

        model, tokenizer = load_base_model(
            model_name=MODEL_NAME
        )

        model_loading_time_seconds = (
            get_elapsed_time(
                model_loading_start_time
            )
        )


        # ----------------------------------------------------
        # Capture system metrics after model loading
        # ----------------------------------------------------

        model_loaded_system_metrics = (
            get_system_metrics()
        )

        model_loaded_metrics = {

            f"model_loaded_{key}":
                value

            for key, value
            in model_loaded_system_metrics.items()
        }

        log_metrics(
            model_loaded_metrics
        )


        # ----------------------------------------------------
        # Start inference timer
        # ----------------------------------------------------

        evaluation_start_time = (
            start_timer()
        )


        # ----------------------------------------------------
        # Run baseline inference
        # ----------------------------------------------------

        evaluation_results = []


        for example in evaluation_dataset:


            # ------------------------------------------------
            # Generate prediction
            # ------------------------------------------------

            predicted_calls, raw_response = (
                predict_tool_calls(
                    example=example,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            )


            # ------------------------------------------------
            # Load expected tool calls
            # ------------------------------------------------

            expected_calls = json.loads(
                example["answers"]
            )


            # ------------------------------------------------
            # Evaluate prediction
            # ------------------------------------------------

            prediction_metrics = (
                evaluate_prediction(
                    expected_calls=expected_calls,
                    predicted_calls=predicted_calls,
                )
            )


            # ------------------------------------------------
            # Store evaluation result
            # ------------------------------------------------

            result = {

                "id":
                    example["id"],

                "query":
                    example["query"],

                "complexity_group":
                    example["complexity_group"],

                "expected_calls":
                    json.dumps(
                        expected_calls
                    ),

                "predicted_calls":
                    json.dumps(
                        predicted_calls
                    ),

                "raw_response":
                    raw_response,

                **prediction_metrics,
            }


            evaluation_results.append(
                result
            )


        # ----------------------------------------------------
        # Calculate inference time
        # ----------------------------------------------------

        evaluation_time_seconds = (
            get_elapsed_time(
                evaluation_start_time
            )
        )


        average_inference_time_seconds = (
            evaluation_time_seconds
            / len(evaluation_dataset)
        )


        # ----------------------------------------------------
        # Capture final system metrics
        # ----------------------------------------------------

        final_system_metrics = (
            get_system_metrics()
        )

        final_metrics = {

            f"final_{key}":
                value

            for key, value
            in final_system_metrics.items()
        }


        # ----------------------------------------------------
        # Save detailed evaluation results
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            evaluation_results
        )

        results_df.to_csv(
            BASELINE_RESULTS_FILE,
            index=False,
        )


        # ----------------------------------------------------
        # Calculate overall metrics
        # ----------------------------------------------------

        overall_metrics = (
            calculate_overall_metrics(
                evaluation_results
            )
        )


        # ----------------------------------------------------
        # Execution metrics
        # ----------------------------------------------------

        execution_metrics = {

            "model_loading_time_seconds":
                model_loading_time_seconds,

            "evaluation_time_seconds":
                evaluation_time_seconds,

            "average_inference_time_seconds":
                average_inference_time_seconds,
        }


        # ----------------------------------------------------
        # Save overall metrics
        # ----------------------------------------------------

        saved_metrics = {

            **overall_metrics,

            **execution_metrics,
        }


        with open(
            BASELINE_METRICS_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                saved_metrics,
                file,
                indent=4,
            )


        # ----------------------------------------------------
        # Calculate complexity metrics
        # ----------------------------------------------------

        complexity_results = (
            calculate_complexity_metrics(
                evaluation_results
            )
        )


        # ----------------------------------------------------
        # Save complexity metrics
        # ----------------------------------------------------

        complexity_results.to_csv(
            BASELINE_COMPLEXITY_FILE
        )


        # ====================================================
        # LOG METRICS TO MLFLOW
        # ====================================================


        # ----------------------------------------------------
        # Log overall evaluation metrics
        # ----------------------------------------------------

        metrics_for_mlflow = {

            key: value

            for key, value
            in overall_metrics.items()

            if key != "total_examples"
        }

        log_metrics(
            metrics_for_mlflow
        )


        # ----------------------------------------------------
        # Log execution metrics
        # ----------------------------------------------------

        log_metrics(
            execution_metrics
        )


        # ----------------------------------------------------
        # Log final system metrics
        # ----------------------------------------------------

        log_metrics(
            final_metrics
        )


        # ----------------------------------------------------
        # Log evaluated example count
        # ----------------------------------------------------

        log_parameters({

            "total_examples":
                overall_metrics[
                    "total_examples"
                ]
        })


        # ====================================================
        # LOG COMPLEXITY METRICS
        # ====================================================

        for complexity_group, row in (
            complexity_results.iterrows()
        ):


            complexity_name = str(
                complexity_group
            ).replace(
                "+",
                "_plus"
            )


            complexity_metric_values = {

                (
                    f"{complexity_name}_"
                    f"{metric}"
                ):
                    float(value)

                for metric, value
                in row.items()

                if metric != "samples"
            }


            log_metrics(
                complexity_metric_values
            )


            log_parameters({

                (
                    f"{complexity_name}_samples"
                ):
                    int(
                        row["samples"]
                    )
            })


        # ====================================================
        # LOG ARTIFACTS
        # ====================================================

        log_artifacts([

            BASELINE_RESULTS_FILE,

            BASELINE_METRICS_FILE,

            BASELINE_COMPLEXITY_FILE,
        ])


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()