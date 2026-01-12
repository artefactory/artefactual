# Entropic Hallucination Detection (EPR & WEPR)

This folder contains scripts to evaluate hallucination detection methods based on **Entropic Prioritization** of token ranks.

## Pipeline Overview

1.  **Feature Calculation**: Compute entropic contribution statistics for the top ranked tokens.
2.  **Merging**: Merge features with ground-truth judgments (using the common utility).
3.  **Evaluation**:
    *   **EPR (Entropic Prioritization in Rank)**: Uses a simple aggregation (mean) of entropies.
    *   **WEPR (Weighted EPR)**: Learns weighted combinations of rank-specific entropies using Logistic Regression.

## Scripts

### 1. `compute_entropic_features.py`
Extracts entropy-based features from generation logs. For each query, it computes the mean, max, and min entropic contributions for tokens at ranks 1 through 15.

**Usage:**
```bash
python compute_entropic_features.py \
    "path/to/generation_with_logprobs.json" \
    "outputs/entropic_features.json"
```

### 2. Merging Data
*Prerequisite*: Use the common script to merge the features with your labels.

```bash
python ../common/merge_features_proba.py \
    "outputs/entropic_features.json" \
    "path/to/judged_data.jsonl" \
    "outputs/training_data_entropy.json"
```

### 3. `eval_EPR.py` (Baseline)
Evaluates the standard **EPR** method. It aggregates the rank-based features into a single scalar (mean entropy over ranks) and evaluates its predictive power.

**Usage:**
```bash
python eval_EPR.py \
    "outputs/training_data_entropy.json" \
    --output_model_file "outputs/model_epr.json"
```

### 4. `eval_WEPR.py` (Weighted)
Evaluates the **WEPR** method. It uses Logistic Regression to learn optimal weights for different ranks (`mean_rank_i`, `max_rank_i`, etc.) to maximize detection performance.

**Usage:**
```bash
python eval_WEPR.py \
    "outputs/training_data_entropy.json" \
    --output_model_file "outputs/model_wepr.json" \
    --repetitions 1000
```
