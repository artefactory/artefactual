# Question Answering Data Generation & Evaluation

This folder contains the scripts used to generate and evaluate answers for Question-Answering (QA) tasks using Large Language Models (LLMs). These scripts are designed for reproducibility of the results presented in our paper.

> **Note**: These scripts are standalone versions optimized for high-throughput batch processing. Functionally similar implementations are available in our Python package under [`src/artefactual/calibration`](../../../src/artefactual/calibration).

## Scripts Overview

### 1. `QA_data_generation.py`
This script handles the generation of answers from an LLM for a given dataset of questions. It leverages **vLLM** and **Ray** for efficient, multi-GPU parallel processing.

**Features:**
*   **Parallel Generation**: Distributes queries across multiple GPUs.
*   **Sampling Support**: Generates multiple samples (iterations) per query.
*   **Rich Output**: Captures generated text, log-probabilities, and token-level details.
*   **Configurable**: Supports temperature scaling, top-k/top-p sampling, and custom prompt templates.

**Usage:**
```bash
python QA_data_generation.py \
    --model_checkpoint "mistralai/Ministral-8B-Instruct-2410" \
    --QA_dataset_path "path/to/trivia_qa.json" \
    --iterations 5 \
    --temperature 0.7 \
    --num_gpus_to_use 4
```

### 2. `rates_answers.py`
This script implements an **LLM-as-a-Judge** pipeline to evaluate the correctness of generated answers. It compares the model's output against a ground truth (expected answer) and determines if they are semantically compatible.

**Features:**
*   **Structured Output**: Uses `pydantic` and `vLLM`'s guided decoding to ensure the judge outputs valid JSON (`judgment` boolean and `explanation` string).
*   **Robust Prompting**: Includes a specialized prompt for semantic equivalence checking (handling aliases, specificity, etc.).
*   **Resume Capability**: Can resume interrupted runs without reprocessing existing judgments.
*   **Configuration**: deeply configurable via command line flags or YAML config files (using `simple-parsing`).

**Usage:**
```bash
# Basic usage
python rates_answers.py \
    --source "generated_results.json" \
    --model=gemma-3 \
    --model.model=google/gemma-3-12b-it \ # Link to the Judge Model you want
    --sampling_params=gemma
```
