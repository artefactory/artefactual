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
The generated `json` file should have entries formatted like :

```json
...
        {
            "query": "What is Bruce Willis' real first name?",
            "query_id": "tc_16",
            "expected_answer": "walter",
            "answer_aliases": [
                "Walter (TV Series)",
                "Walter",
                "Walter (disambiguation)",
                "Walter (TV series)"
            ],
            "main_answer": "William",
            "sampled_answers": [
                "Bruce Willis' real first name is Walter.",
                "Walter.",
                "Walter",
                "Walter",
                "Walter",
                "Bruce Willis's real first name is Walter.",
                "Bruce Willis' real first name is Walter.",
                "Bruce Willis' real first name is William.",
                "Walter",
                "Walter Bruce Willis"
            ]
        },
...

```

### 2. Scoring (SelfCheckGPT-BERTScore)
Compute the stochastic inconsistency scores.

```bash
python SCGPT_bertscore.py \
    --input_file "outputs/mistral_gen.json"
# Output will be saved as "outputs/mistral_gen_Bertscored.json"
```

### 3. Data Fusion
Combine the scored data with ground truth judgments/labels for evaluation.

*Note: You may need to use `SCGPT_mod_json.py` first if you have updates/patches to apply to your answers. In our case, we needed to make sure that the judged/labeled answers were the one used in the `judged_data.jsonl`*

```bash
python fuse_score_judgement.py \
    "outputs/mistral_gen_Bertscored.json" \
    "path/to/judged_data.jsonl" \
    "outputs/mistral_gen_judged.json"
```

Your end `json` file elements should look like :

```json
...
    {
        "query_id": "7228cd94-7e7e-4dbd-9d06-bc507bd31647",
        "selfcheck_bertscore": 0.5798528790473938,
        "judgment": false
    },
    {
        "query_id": "4fb844e0-c327-4a12-9f8b-127dbf59c98a",
        "selfcheck_bertscore": 0.015364468097686768,
        "judgment": true
    },
    {
        "query_id": "de50eb32-3e33-4f9a-8389-fe910c056ce4",
        "selfcheck_bertscore": 0.07880479097366333,
        "judgment": true
    },
...
```

In our case, we used `scripts/paper_scripts/common/rates_answers.py` that sets-up a LLM-as-a-judge approach in order to get labels, with very high agreement with human annotators.

### 4. Evaluation
Run bootstrap evaluation to get AUC metrics. This scripts performs the evaluation on all files following the `'*judged.json'` naming pattern in a directory.

```bash
python bootstrap_evaluation.py \
    --input_dir "outputs/" \
    --output_file "results_bootstrap.txt"
```
