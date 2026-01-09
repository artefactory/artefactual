# SelfCheckGPT Benchmark Scripts

This folder contains the scripts used to evaluate **SelfCheckGPT** as a hallucination detection method for our research.

## Pipeline Overview

The evaluation pipeline consists of four main steps:
1.  **Generation**: Using `vllm` to generate multiple samples for each query.
2.  **Scoring**: Calculating SelfCheckGPT (BERTScore) scores for the generated responses.
3.  **Data Fusion**: Merging scores with ground-truth judgments.
4.  **Evaluation**: Running bootstrap analysis to compute ROC-AUC / PR-AUC.

## Requirements

- Python 3.10+
- `vllm`
- `bert_score`
- `scikit-learn`
- `torch`

## Usage

### 1. Generation
Generate the main response and stochastic samples.

```bash
# For generic models (e.g., Falcon3)
python SCGPT_generate.py \
    --model_checkpoint "tiiuae/Falcon3-10B-Instruct" \
    --input_file "path/to/trivia_qa.json" \
    --output_file "outputs/falcon_gen.json" \
    --iterations 11 # 1 main to compare with 10 other samples

```
```bash

# For Mistral models (specific tokenizer config)
python SCGPT_generate_mistral.py \
    --model_checkpoint "mistralai/Ministral-8B-Instruct-2410" \
    --input_file "path/to/trivia_qa.json" \
    --output_file "outputs/mistral_gen.json"

```

### 2. Scoring (SelfCheckGPT-BERTScore)
Compute the stochastic inconsistency scores.

```bash
python SCGPT_bertscore.py \
    --input_file "outputs/mistral_gen.json"
# Output will be saved as "outputs/mistral_gen_Bertscored.json"
```

### 3. Data Fusion
Combine the scored data with ground truth judgments for evaluation.

*Note: You may need to use `SCGPT_mod_json.py` first if you have updates/patches to apply to your answers. In our case, we needed to make sure that the judged/labeled answers were the one used in the `judged_data.jsonl`*

```bash
python fuse_score_judgement.py \
    "outputs/mistral_gen_Bertscored.json" \
    "path/to/judged_data.jsonl" \
    "outputs/mistral_final.json"
```

### 4. Evaluation
Run bootstrap evaluation to get AUC metrics. This scripts performs the evaluation on all files following the `'*judged.json'` naming pattern in a directory.

```bash
python bootstrap_evaluation.py \
    --input_dir "outputs/" \
    --output_file "results_bootstrap.txt"
```
