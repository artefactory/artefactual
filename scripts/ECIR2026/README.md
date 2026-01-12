# Reproducibility Scripts

This directory contains the scripts required to reproduce the results presented in our paper:

**"Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate"**

## Directory Structure

### 1. General Utilities (`common/`)
This folder contains the shared pipeline scripts used to create the base datasets:
- **Data Generation**: Generating answers to Question-Answering tasks with detailed log-probabilities.
- **Labeling**: Applying an LLM-as-a-judge approach to rate the correctness of answers and provide ground-truth labels.
- **Utilities**: Helper scripts for merging features with labels.

### 2. Entropic Methods (`Experiments/`)
Scripts implementing our proposed methods:
- **EPR**: Assessing hallucination using the Entropic Production Rate.
- **WEPR**: A weighted, learned variant of EPR.

### 3. Benchmarks
We provide the scripts used to compare our approach against other methods:
- **`SelfCheckGPT/`**: Implementation of the SelfCheckGPT (BERTScore) method.
- **`Halludetect/`**: Implementation of uncertainty-based features (MTP, AvgTP, etc.) for detection.

Each subfolder contains a detailed `README.md` with specific instructions on how to run the scripts.
