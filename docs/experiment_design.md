# Experiment Design

## 1. Project Objective

The objective of this project is to evaluate and improve the tool-calling capabilities of a large language model using parameter-efficient fine-tuning techniques.

The project evaluates the ability of a model to:

- Select the correct tool
- Generate the correct number of tool calls
- Generate the correct tool arguments
- Produce tool calls that exactly match the expected output

The experimental workflow consists of:

1. Establishing baseline performance using the base model
2. Fine-tuning the model using LoRA
3. Evaluating the LoRA fine-tuned model
4. Fine-tuning the model using QLoRA
5. Evaluating the QLoRA fine-tuned model
6. Comparing the experiments using MLflow

---

## 2. Base Model

The base model used for the experiments is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The base model is first evaluated without task-specific fine-tuning to establish baseline tool-calling performance.

The same base model will subsequently be used for the LoRA and QLoRA experiments.

---

## 3. Dataset

The dataset used in this project is:

```text
Salesforce/xlam-function-calling-60k
```

The dataset contains examples designed for function and tool-calling tasks.

Each example includes information required for evaluating tool-calling behavior, including:

- User query
- Available tool definitions
- Expected tool calls

---

## 4. Dataset Complexity Groups

Examples are categorized according to the number of expected tool calls.

The complexity groups are:

| Complexity Group | Description |
|---|---|
| `1_call` | The example requires one tool call |
| `2_calls` | The example requires two tool calls |
| `3_calls` | The example requires three tool calls |
| `4_plus_calls` | The example requires four or more tool calls |

The complexity grouping enables model performance to be evaluated across different levels of task difficulty.

---

## 5. Original Dataset Split

The original dataset is divided into training, validation, and test datasets.

| Dataset Split | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

Given the approximately 60,000 examples in the dataset, the resulting splits are approximately:

| Dataset Split | Approximate Samples |
|---|---:|
| Training | 48,000 |
| Validation | 6,000 |
| Test | 6,000 |

The dataset splitting process considers the complexity groups to ensure that the different tool-calling complexity levels are represented across the dataset splits.

---

## 6. Representative Experimental Dataset

Fine-tuning and evaluating the complete dataset requires significant computational resources.

Therefore, a smaller experimental dataset is used for the experiments.

The representative dataset configuration is:

| Dataset Split | Experimental Samples |
|---|---:|
| Training | 4,000 |
| Validation | 1,000 |
| Test | 1,000 |

The representative datasets are selected from the previously created training, validation, and test splits.

This preserves the separation between:

- Training data
- Validation data
- Test data

The smaller dataset configuration enables the project to perform baseline evaluation, LoRA fine-tuning, and QLoRA fine-tuning within available computational resources.

---

## 7. Experimental Configurations

The project compares three model configurations.

### Experiment 1: Baseline Model

The base model is evaluated without task-specific fine-tuning.

Model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The baseline evaluation establishes the initial performance of the model on the tool-calling task.

The baseline results will serve as the reference point for evaluating the impact of fine-tuning.

---

### Experiment 2: LoRA Fine-Tuning

The base model will be fine-tuned using Low-Rank Adaptation (LoRA).

LoRA is a parameter-efficient fine-tuning technique that introduces trainable low-rank adapters into selected model layers while keeping the majority of the original model parameters frozen.

The objective of this experiment is to determine whether LoRA fine-tuning improves the model's tool-calling performance.

The LoRA fine-tuned model will be evaluated using the same evaluation methodology and test dataset as the baseline model.

---

### Experiment 3: QLoRA Fine-Tuning

The base model will be fine-tuned using Quantized Low-Rank Adaptation (QLoRA).

QLoRA combines:

- Quantization of the base model
- Low-rank trainable adapters

The objective of this experiment is to investigate whether memory-efficient fine-tuning can achieve competitive tool-calling performance while reducing memory requirements.

The QLoRA fine-tuned model will be evaluated using the same evaluation methodology and test dataset as the baseline and LoRA models.

---

## 8. Experimental Comparison

The experiments are designed to compare:

```text
Baseline Model
      ↓
LoRA Fine-Tuned Model
      ↓
QLoRA Fine-Tuned Model
```

All model configurations will be evaluated using the same test dataset.

This ensures that performance differences can be attributed to the model configuration rather than differences in evaluation data.

---

## 9. Evaluation Dataset

The representative test dataset contains:

```text
1,000 examples
```

The same representative test dataset will be used to evaluate:

- The baseline model
- The LoRA fine-tuned model
- The QLoRA fine-tuned model

Using the same test dataset enables direct comparison between the three experimental configurations.

---

## 10. Evaluation Metrics

The model predictions are compared against the expected tool calls using multiple metrics.

### 10.1 Exact Match Accuracy

Exact match accuracy measures whether the predicted tool calls exactly match the expected tool calls.

An exact match requires:

- Correct number of tool calls
- Correct tool names
- Correct tool arguments

### 10.2 Tool Name Accuracy

Tool name accuracy measures whether the predicted tool names match the expected tool names.

This metric evaluates whether the model correctly identifies the tools required to answer the user query.

### 10.3 Arguments Accuracy

Arguments accuracy measures whether the arguments generated by the model match the expected tool arguments.

This metric evaluates the model's ability to extract and generate the required information for each tool call.

### 10.4 Call Count Accuracy

Call count accuracy measures whether the model predicts the correct number of tool calls.

This is particularly important for examples that require multiple tool calls.

---

## 11. Complexity-Based Evaluation

Performance metrics are calculated both:

1. Across the complete evaluation dataset
2. Separately for each complexity group

The complexity groups include:

```text
1_call
2_calls
3_calls
4_plus_calls
```

For each complexity group, the following metrics are calculated:

- Exact match accuracy
- Tool name accuracy
- Arguments accuracy
- Call count accuracy

This allows the experiments to analyze how model performance changes as the number of required tool calls increases.

---

## 12. Generation Configuration

Model predictions are generated using a consistent generation configuration across experiments where applicable.

The baseline evaluation uses:

```text
max_new_tokens = 256
```

Generation is configured deterministically to ensure that predictions can be consistently evaluated and compared.

Maintaining a consistent inference configuration allows comparisons between experiments to focus on differences introduced by fine-tuning.

---

## 13. Performance Monitoring

In addition to model evaluation metrics, system resource metrics are collected during experiments.

The monitored resources include:

- CPU utilization
- System memory usage
- System memory used
- System memory available
- GPU memory usage when CUDA is available

These metrics provide additional information about the computational requirements of the experiments.

---

## 14. Timing Metrics

The experiments track execution-related metrics.

The recorded timing metrics include:

- Model loading time, when model loading is part of the experiment execution
- Total evaluation time
- Individual inference time
- Average inference time per example

These metrics enable comparison of the computational efficiency of the experimental configurations.

---

## 15. Experiment Tracking

MLflow is used for experiment tracking.

Each experimental execution is recorded as an MLflow run.

The tracked information includes parameters, metrics, and experiment artifacts.

### Parameters

The experiment parameters may include:

- Model name
- Model type
- Maximum input tokens
- Maximum new tokens
- Number of evaluation samples
- Dataset configuration
- Fine-tuning configuration

Fine-tuning parameters will be logged for the LoRA and QLoRA experiments.

### Model Performance Metrics

The following metrics are logged:

- Exact match accuracy
- Tool name accuracy
- Arguments accuracy
- Call count accuracy

### Complexity-Based Metrics

The following metrics are logged separately for each complexity group:

- Exact match accuracy
- Tool name accuracy
- Arguments accuracy
- Call count accuracy

### System Metrics

The experiments log system resource metrics, including:

- CPU usage percentage
- Memory usage percentage
- Memory used in GB
- Memory available in GB
- GPU memory metrics when available

System metrics may be captured at different stages of the experiment, including:

- Initial state
- After model loading
- Final state

### Timing Metrics

The following timing metrics are logged:

- Model loading time
- Evaluation time
- Average inference time

---

## 16. Checkpointing Strategy

Baseline evaluation can require significant execution time, particularly when running inference on CPU.

To reduce the risk of losing evaluation progress, checkpointing is used during long-running evaluation jobs.

The evaluation process periodically saves completed results.

When evaluation is restarted:

1. The pipeline checks for an existing checkpoint
2. Previously completed results are loaded
3. The evaluation resumes from the next unprocessed example

This approach improves fault tolerance for long-running experiments.

---

## 17. Experimental Artifacts

Detailed evaluation results are saved as experiment artifacts.

The stored results include information such as:

- Example identifier
- Input query
- Complexity group
- Expected tool calls
- Predicted tool calls
- Raw model response
- Evaluation results

These artifacts enable detailed analysis of model predictions after an experiment is completed.

---

## 18. Experimental Hypotheses

The experiments are designed to investigate the following hypotheses.

### Hypothesis 1

Task-specific fine-tuning will improve tool-calling performance compared with the base model.

Expected comparison:

```text
LoRA / QLoRA Performance
>
Baseline Performance
```

### Hypothesis 2

Fine-tuning will improve performance across multiple tool-calling complexity levels.

The improvement will be analyzed separately for:

```text
1_call
2_calls
3_calls
4_plus_calls
```

### Hypothesis 3

LoRA and QLoRA may demonstrate different trade-offs between:

- Model performance
- Training efficiency
- Memory usage
- Computational requirements

---

## 19. Final Model Comparison

The final experiment comparison will evaluate the following model configurations:

| Model | Fine-Tuning Method |
|---|---|
| Qwen2.5-1.5B-Instruct | None |
| Qwen2.5-1.5B-Instruct | LoRA |
| Qwen2.5-1.5B-Instruct | QLoRA |

The comparison will consider:

### Tool-Calling Performance

- Exact match accuracy
- Tool name accuracy
- Arguments accuracy
- Call count accuracy

### Complexity-Based Performance

- Performance on single tool-call tasks
- Performance on two tool-call tasks
- Performance on three tool-call tasks
- Performance on tasks requiring four or more tool calls

### Computational Performance

- Evaluation time
- Average inference time
- CPU utilization
- System memory usage
- GPU memory usage

---

## 20. Expected Outcome

The expected outcome of this project is a comparative evaluation of baseline, LoRA, and QLoRA approaches for improving tool-calling performance.

The project aims to determine:

1. Whether parameter-efficient fine-tuning improves tool-calling accuracy
2. Which evaluation metrics improve after fine-tuning
3. How performance changes across different tool-calling complexity levels
4. How LoRA and QLoRA compare in terms of performance and computational efficiency

The final results will be tracked and compared using MLflow.