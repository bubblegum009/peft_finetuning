from pathlib import Path

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# DATASET CONFIGURATION
DATASET_NAME = "Salesforce/xlam-function-calling-60k"
TRAIN_DATA_PATH = DATA_DIR / "train"
VALIDATION_DATA_PATH = DATA_DIR / "validation"
TEST_DATA_PATH = DATA_DIR / "test"

# MODEL CONFIGURATION
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256

# BASELINE EVALUATION CONFIGURATION
BASELINE_NUM_SAMPLES = 5

# RESULT FILE PATHS
BASELINE_RESULTS_FILE = (
    RESULTS_DIR
    / f"baseline_results_{BASELINE_NUM_SAMPLES}.csv"
)
BASELINE_COMPLEXITY_FILE = (
    RESULTS_DIR
    / f"baseline_complexity_{BASELINE_NUM_SAMPLES}.csv"
)
BASELINE_METRICS_FILE = (
    RESULTS_DIR
    / f"baseline_metrics_{BASELINE_NUM_SAMPLES}.json"
)


# MLFLOW CONFIGURATION
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = (
    "tool_calling_finetuning"
)

# FINETUNED MODEL BENCHMARK CONFIGURATION
FINETUNED_NUM_SAMPLES = 1000

LORA_ADAPTER_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "final_lora_adapter"
)

FINETUNED_RESULTS_FILE = (
    RESULTS_DIR
    / f"lora_finetuned_results_{FINETUNED_NUM_SAMPLES}.csv"
)

FINETUNED_COMPLEXITY_FILE = (
    RESULTS_DIR
    / f"lora_finetuned_complexity_{FINETUNED_NUM_SAMPLES}.csv"
)

FINETUNED_METRICS_FILE = (
    RESULTS_DIR
    / f"lora_finetuned_metrics_{FINETUNED_NUM_SAMPLES}.json"
)

QLORA_ADAPTER_PATH = (
    CHECKPOINTS_DIR
    / "qlora"
    / "final_adapter"
)

QLORA_NUM_SAMPLES = 1000

QLORA_RESULTS_FILE = (
    RESULTS_DIR
    / "qlora_results_1000.csv"
)

QLORA_COMPLEXITY_FILE = (
    RESULTS_DIR
    / "qlora_complexity_1000.csv"
)

QLORA_METRICS_FILE = (
    RESULTS_DIR
    / "qlora_metrics_1000.json"
)