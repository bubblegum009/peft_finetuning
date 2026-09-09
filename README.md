# Tool Calling with Qwen 2.5 — SFT, LoRA & QLoRA

A controlled fine-tuning and evaluation project for improving structured tool/function calling with `Qwen/Qwen2.5-1.5B-Instruct`.

This project compares a pretrained baseline against two parameter-efficient fine-tuning approaches:

- **Baseline** — pretrained Qwen2.5-1.5B-Instruct
- **LoRA** — Low-Rank Adaptation
- **QLoRA** — 4-bit quantized base model + LoRA

The objective is to improve structured tool-calling reliability while analyzing the trade-offs between model quality, inference latency, and GPU memory usage.

---

## Project Overview

Large Language Models can be used as tool-calling agents, where the model selects an appropriate tool and generates the arguments required to invoke it.

For example:

```
<tool_call>
{"name":"stock_get_statistics","arguments":{"is_id":"TSLA"}}
</tool_call>
```

Reliable tool calling requires the model to correctly determine:

- Which tool to select
- Which arguments to provide
- How many tools need to be called
- How to generate the expected structured output

This project investigates whether supervised fine-tuning can improve these capabilities using the `Salesforce/xlam-function-calling-60k` dataset.

---

## Key Results

All three configurations were evaluated on the same representative 1,000-example test set.

### Quality Metrics

| Metric                   | Baseline | LoRA | QLoRA |
|--------------------------|----------|------|-------|
| **Exact Match Accuracy** | 67.20% | 74.10% | 76.90% |
| **Tool Name Accuracy**   | 94.05% | 94.30% | 97.30% |
| **Argument Accuracy**    | 74.06% | 74.20% | 77.20% |
| **Call Count Accuracy**  | 91.80% | 95.20% | 97.90% |


### Improvement over Baseline

| Metric                    | LoRA Improvement | QLoRA Improvement |
|---------------------------|------------------|-------------------|
| **Exact Match Accuracy**  | +6.90 pp         | +9.70 pp |
| **Tool Name Accuracy**    | +0.25 pp         | +3.25 pp |
| **Argument Accuracy**     | +0.16 pp         | +3.14 pp |
| **Call Count Accuracy**   | +3.40 pp         | +6.10 pp |


**QLoRA achieved the highest score across all reported quality metrics in this experiment.**

---

## Performance and Resource Usage

The experiments were conducted on an NVIDIA T4 GPU with approximately 14.56 GB of VRAM.

| Metric                      | Baseline| LoRA-- | QLoRA   |
|-----------------------------|---------|--------|---------|
| **Average Inference Time**  | 1.76 s  | 2.76 s | 4.39 s  |
| **Total Inference Time**    | 1,756 s | 2,762 s| 4,387 s |
| **Average GPU Memory Used** | 8.80 GB | 7.52 GB| 5.67 GB |
| **Maximum GPU Memory Used** | 8.84 GB | 7.79 GB| 5.83 GB |

### Observed Trade-off

| Configuration | Observation
|---------------|----------------
| **Baseline**  | Fastest inference but Lowest tool-calling quality 
| **LoRA**      | Improved quality over baseline but Higher memory usage and latency 
| **QLoRA**     | Best quality and lowest memory among fine-tuned models but Highest inference latency 

**QLoRA reduced average GPU memory usage from approximately 7.52 GB with LoRA to 5.67 GB, while achieving higher evaluation scores.**

---

## Model

The base model used throughout the project is:

```
Qwen/Qwen2.5-1.5B-Instruct
```

The model contains approximately 1.5 billion parameters.

### Configurations

| Configuration | Base Model | Fine-Tuning |
|---------------|------------|-------------|
| Baseline      | FP16       | None        |
| LoRA          | FP16       | LoRA        |
| QLoRA         | 4-bit NF4  | LoRA        |

---

## Dataset

The project uses: `Salesforce/xlam-function-calling-60k`

The dataset contains examples consisting of:
- User queries
- Available tool definitions
- Expected tool calls
- Tool arguments

The original dataset contains approximately 60,000 examples.

### Dataset Analysis

Examples were categorized according to the number of expected tool calls.

| Complexity | Samples | Percentage |
|------------|---------|------------|
| 1 call     | 28,461  | 47.44%     |
| 2 calls    | 25,422  | 42.37%     |
| 3 calls    | 4,697   | 7.83%      |
| 4+ calls   | 1,420   | 2.37%      |
|Total       | 60,000  | 100%       |

This distribution was used to construct representative training, validation, and test subsets.

### Data Splitting

The dataset was split using an 80/10/10 stratified strategy based on tool-call complexity.

| Split            | Full Dataset |
|------------------|--------------|
| Train            | 48,000       |
| Validation       | 6,000        |
| Test             | 6,000        |

For experimentation and faster iteration, representative subsets were created:

| Split       | Representative Set |
|-------------|--------------------|
| Train       | 4,000              |
| Validation  | 1,000              |
| Test        | 1,000              |  

**The same representative test set was used to benchmark all three configurations, providing a controlled comparison.**

---

## SFT Data Preparation

The raw dataset was transformed into a supervised fine-tuning format.

Each training example contains:

```
System prompt
    ↓
Available tool definitions
    ↓
User query
    ↓
Expected tool-call completion
```

### Example Target Completion

```
<tool_call>
{"name":"stock_get_statistics","arguments":{"is_id":"TSLA"}}
</tool_call>
```

Multi-tool examples contain multiple tool-call blocks.

### Completion-Only Loss

The fine-tuning setup uses completion-only loss. The model receives the complete prompt, including the available tools and user query, but the loss is calculated only over the target tool-call completion.

```
System instructions
Tool definitions
User query
    ↓
   Ignored by loss

Tool-call completion
    ↓
   Optimized by loss
```

This focuses training on learning the desired structured tool-calling behavior.

**Maximum sequence length used during preprocessing:** 1024 tokens

---

## LoRA Configuration

LoRA (Low-Rank Adaptation) was used for parameter-efficient fine-tuning.

Instead of updating all model parameters, LoRA introduces trainable low-rank matrices into selected transformer modules.

### LoRA Settings

| Parameter | Value |
|-----------|-------|
| Rank (r)  | 16    |
| Alpha     | 32    |
| Dropout   | 0.05  |

Target Modules : q_proj, k_proj, v_proj, o_proj 

### Trainable Parameters

| Parameter            | Count         |
|----------------------|---------------|
| Trainable Parameters | 4,358,144     |
| Total Parameters     | 1,548,072,448 |
| Trainable Percentage | ~0.28%        |
 
**Only a small fraction of the model parameters are updated during LoRA fine-tuning.**

---

## QLoRA Configuration

QLoRA combines LoRA with quantization.

For QLoRA training, the frozen base model was loaded using 4-bit NF4 quantization.

### QLoRA Settings

| Configuration       | Value |
|---------------------|-------|
| Quantization        | 4-bit |
| Quantization Type   | NF4 --|
| Double Quantization | True  |
| Compute Dtype       | FP16  |

The LoRA adapter configuration remained consistent with the LoRA experiment.

### Conceptual Comparison

```
LoRA
├─ FP16 frozen base model
└─ LoRA adapters

QLoRA
├─ 4-bit frozen base model
└─ LoRA adapters
```

The base model remains frozen while the LoRA adapters are trained.

---

## Training Configuration

The main training configuration was kept consistent between LoRA and QLoRA.

| Parameter                   | Value |
|-----------------------------|-------|
| Epochs                      | 3     |
| Learning Rate               | 2e-4  |
| Per-Device Train Batch Size | 2     |
| Gradient Accumulation Steps | 8     |
| Effective Batch Size        | 16    |
| Maximum Sequence Length     | 1024  |
| Scheduler                   |Cosine |
| Completion-Only Loss        |Enabled|
| Seed                        | 42    |

The same representative training and validation data were used for both LoRA and QLoRA experiments.

---

## Evaluation Methodology

Each model was evaluated using the same representative test set: **1,000 examples**

Generation was performed deterministically using:
- `do_sample = False`
- `max_new_tokens = 512`

A robust parser was implemented to normalize model outputs before calculating metrics. The parser supports multiple output representations:
- `<tool_call>...</tool_call>`
- Raw JSON tool calls
- Python-style function calls

This allows the evaluation to distinguish between actual tool-calling errors and differences in textual representation.

### Evaluation Metrics

**Exact Match Accuracy**  
Measures whether the complete parsed prediction exactly matches the expected tool-call structure. This is the strictest metric used in the evaluation.

**Tool Name Accuracy**  
Measures whether the model selected the correct tool. This isolates tool selection from argument generation.

**Argument Accuracy**  
Measures whether the generated arguments match the expected arguments. This helps identify cases where the model selected the correct tool but supplied incorrect parameters.

**Call Count Accuracy**  
Measures whether the model generated the correct number of tool calls. This is particularly important for multi-tool queries.

---

## Complexity Analysis

The evaluation set contains examples requiring different numbers of tool calls.

### Exact Match Accuracy by Complexity

| Complexity | Samples | Baseline | LoRA   | QLoRA  |
|------------|---------|----------|--------|--------|
| 1 call     | 474     | 73.63%   | 79.11% | 81.01% |
| 2 calls    | 424     | 63.68%   | 69.58% | 73.35% |
| 3 calls    | 78      | 56.41%   | 75.64% | 79.49% |
| 4+ calls   | 24      | 37.50%   | 50.00% | 50.00% |

**The results show that multi-tool calling is generally more challenging than single-tool calling. The 4+ call category contains only 24 examples, so its results should be interpreted cautiously.**

---

## Key Findings

### 1. Fine-Tuning Improves Structured Tool Calling

The pretrained baseline achieved:
- **Baseline:** 67.20% Exact Match Accuracy

LoRA improved this to:
- **LoRA:** 74.10%

While QLoRA achieved:
- **QLoRA:** 76.90%

on the same evaluation set. This represents a **9.70 percentage-point improvement** over the baseline for QLoRA.

### 2. QLoRA Achieved the Best Overall Quality

QLoRA achieved the highest score across every reported quality metric in this experiment:

| Metric                   | QLoRA- |
|--------------------------|--------|
| **Exact Match Accuracy** | 76.90% |
| **Tool Name Accuracy**   | 97.30% |
| **Argument Accuracy**    | 77.20% |
| **Call Count Accuracy**  | 97.90% |


### 3. Tool Selection Is Easier Than Exact Argument Generation

QLoRA achieved:
- **Tool Name Accuracy:** 97.30%
- **Exact Match Accuracy:** 76.90%

This indicates that the model usually identifies the correct tool, while generating exactly correct arguments remains a more challenging part of the task.

### 4. Multi-Tool Calling Is More Challenging

Performance generally decreases as the number of required tool calls increases.

Multi-tool queries require the model to maintain:
- Correct tool selection
- Correct argument generation
- Correct number of calls
- Correct ordering
- Correct output structure

The complexity-based evaluation makes these failure patterns visible.

### 5. QLoRA Provides a Strong Memory-Quality Trade-off

QLoRA achieved higher quality than LoRA while using substantially less GPU memory.

**Average GPU memory usage:**
- **LoRA:** ~7.52 GB
- **QLoRA:** ~5.67 GB

This makes QLoRA useful for memory-constrained environments. The trade-off observed in this experiment is higher inference latency.

---

## Inference Latency

Average measured inference time:

| Configuration | Average Inference Time |
|---------------|------------------------|
| **Baseline**  | 1.76 s                 |
| **LoRA**      | 2.76 s                 |
| **QLoRA**     | 4.39 s                 |

The baseline was the fastest configuration in the measured environment. The fine-tuned models achieved higher tool-calling quality at the cost of increased inference latency.

**Latency was measured during the 1,000-example evaluation runs on an NVIDIA T4 GPU.**

---

## Resource Monitoring

The evaluation pipeline recorded:
- CPU utilization
- RAM utilization
- GPU utilization
- GPU memory used
- GPU memory reserved
- Per-example inference latency

This provides additional information beyond model accuracy and helps analyze practical deployment trade-offs.

---

## MLflow Experiment Tracking

MLflow was used for experiment tracking and evaluation artifact management.

The MLflow tracking database is local and is not hosted publicly. The experiments were tracked using a local SQLite database: `mlflow.db`

**The local MLflow database is intentionally not committed to the GitHub repository.**

MLflow was used to record:
- Experiment parameters
- Evaluation metrics
- Resource utilization
- Evaluation artifacts
- Experiment run information

### Evaluation Run IDs

| Experiment   | MLflow Run ID |
|--------------|---------------|
| **Baseline** | e9e6eae477ed404f8680ebdacde127a7 |
| **LoRA**     | c466e38d8e6840ee978efad |
| **QLoRA**    | 4de60bf1f1814d20924e7b68bb0db4fe |

These run IDs refer to the local MLflow tracking database used during experimentation and are provided for experiment traceability.

---

## Repository Structure

```
.
├── config/
│
├── docs/
│
├── notebooks/
│
├── scripts/
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Overview

| config/ | Project and experiment configuration |
| docs/ | Project documentation |
| notebooks/ | Data preparation, training, and evaluation experiments |
| scripts/ | Utility and execution scripts |
| src/ | Reusable project source code |

---

## Reproducibility

The experiments were designed as a controlled comparison.

The following were kept consistent across the evaluation configurations:

- Base model architecture
- Representative test set
- Evaluation methodology
- Generation settings
- Parser
- Evaluation metrics
- Test examples
- Random seed where applicable

This allows the effect of parameter-efficient fine-tuning to be evaluated under consistent conditions.

---

## Hardware

Experiments were conducted using:

| Component | Specification |
|-----------|---------------|
| **GPU**   | T4     |
| **VRAM**  | ~14.56 GB     |

The QLoRA configuration was particularly useful for reducing GPU memory requirements.

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL
- PEFT
- BitsAndBytes
- Qwen2.5
- MLflow
- Google Colab
- NVIDIA T4

---

## Limitations

- Evaluation was performed on a representative 1,000-example test set rather than the complete 6,000-example test split
- The benchmark evaluates tool-call generation rather than actual downstream tool execution
- Exact-match evaluation is strict and may penalize semantically equivalent outputs
- The 4+ tool-call category contains only 24 examples
- Inference latency depends on the hardware and inference environment
- MLflow tracking is local rather than hosted
- The current benchmark focuses on structured function-call generation rather than complete end-to-end agent performance

---

## Future Work

Potential extensions include:

- Evaluation on the complete test split
- Tool execution success rate
- Schema-validity evaluation
- More granular argument-level evaluation
- Batched inference benchmarking
- Standardized CUDA-synchronized latency benchmarking
- Evaluation using production inference engines
- Testing additional parameter-efficient fine-tuning approaches
- Evaluation on larger and more diverse tool sets

---

## Conclusion

This project demonstrates that parameter-efficient fine-tuning can improve structured tool-calling behavior in a relatively small instruction-tuned language model.

Starting from the pretrained Qwen2.5-1.5B-Instruct baseline:

| Configuration | Exact Match Accuracy |
|---------------|----------------------|
| **Baseline**  | 67.20%               |
| **LoRA**      | 74.10%               |
| **QLoRA**     | 76.90%               |

on the same 1,000-example representative test set.

QLoRA achieved the best overall quality across the evaluated metrics while using substantially less GPU memory than the FP16 LoRA configuration.

### Quality-Resource Trade-off Summary

| Configuration | Quality | Latency | Memory 
|---------------|---------|---------|--------
| **Baseline**  | Lowest  | Lowest  | Highest in measured inference 
| **LoRA**      | Improved| Moderate| Moderate 
| **QLoRA**     | Best    | Highest | Lowest among fine-tuned models 
 
**Overall, the results show that task-specific fine-tuning improved structured tool-calling reliability, with QLoRA providing the best observed quality-to-memory trade-off in this experiment.**