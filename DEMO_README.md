# UncertaintyDetector Demo

This demo script showcases the `UncertaintyDetector` class for computing Entropy Production Rate (EPR) scores on language model outputs.

## Quick Start

### Basic Usage

Run the demo with a model checkpoint:

```bash
python demo_uncertainty.py \
    --model_checkpoint "mistralai/Mistral-7B-Instruct-v0.2" \
    --n_queries 5 \
    --iterations 10 \
    --number_logprobs 15
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_checkpoint` | str | **required** | HuggingFace model checkpoint path |
| `--n_queries` | int | 5 | Number of queries to process |
| `--iterations` | int | 10 | Number of sampling iterations per query |
| `--number_logprobs` | int | 15 | Number of top-K logprobs (K for UncertaintyDetector) |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--top_k_sampling` | int | 50 | Top-K for sampling |
| `--tensor_parallel_size` | int | 2 | Tensor parallelism size |
| `--gpu_memory_utilization` | float | 0.90 | GPU memory utilization (0.0-1.0) |
| `--data_path` | str | (see below) | Path to JSON data file |
| `--output_dir` | str | `generation_test` | Output directory for results |

<!-- Default data path: `/data/workspace/charles/artefactual_data/web_question_qa.json` -->

### Example Commands

**Quick test (5 queries):**
```bash
python demo_uncertainty.py \
    --model_checkpoint "mistralai/Mistral-7B-Instruct-v0.2" \
    --n_queries 5 \
    --iterations 5
```

**Full run with custom output:**
```bash
python demo_uncertainty.py \
    --model_checkpoint "meta-llama/Llama-2-7b-chat-hf" \
    --n_queries 50 \
    --iterations 10 \
    --number_logprobs 20 \
    --output_dir "my_results"
```

**Ministral model (specific config):**
```bash
python demo_uncertainty.py \
    --model_checkpoint "mistralai/Ministral-8B-Instruct-2410" \
    --n_queries 10
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "generator_model": "model_name",
    "date": "timestamp",
    "temperature": 1.0,
    "iterations": 10,
    "number_logprobs": 15,
    "uncertainty_detector_K": 15
  },
  "results": [
    {
      "query": "What is...",
      "query_id": "...",
      "expected_answer": "...",
      "answer_aliases": [...],
      "generated_answers": [...],
      "full_info": [
        {
          "answer_text": "...",
          "cumulative_logprob": -12.34,
          "detailed_logprobs": [...],
          "epr_score": 2.45,
          "token_epr": [2.1, 2.3, 2.6, ...]
        }
      ],
      "epr_statistics": {
        "mean": 2.45,
        "std": 0.12,
        "min": 2.20,
        "max": 2.70,
        "median": 2.45
      }
    }
  ]
}
```

## Understanding EPR Scores

- **EPR (Entropy Production Rate)**: Measures uncertainty in token generation
- **Higher EPR** = More uncertain/less confident predictions
- **Lower EPR** = More confident predictions
- **Per-token EPR**: Shows uncertainty for each token in the sequence
- **Sequence EPR**: Average uncertainty across all tokens

## Logs

Logs are saved to `logs/demo_uncertainty_YYYYMMDD_HHMMSS.log` with:
- Model initialization details
- Generation progress
- EPR computation times
- Summary statistics per query

## Requirements

- Python 3.8+
- vLLM
- PyTorch with CUDA
- transformers
- numpy

Install dependencies:
```bash
pip install vllm torch transformers numpy
```

## Advanced Usage

### Custom Data Format

The input JSON should have this structure:
```json
[
  {
    "question": "Your question here",
    "question_id": "unique_id",
    "short_answer": "expected answer",
    "answer_aliases": ["alias1", "alias2"]
  }
]
```

### Integration with Your Code

```python
from artefactual.models import UncertaintyDetector

# Initialize
detector = UncertaintyDetector(K=15)

# Compute EPR scores (with vLLM outputs)
epr_scores, token_scores = detector.compute_epr(vllm_outputs, return_tokens=True)

# Get statistics
stats = detector.compute_epr_stats(vllm_outputs)
print(f"Mean EPR: {stats['mean']:.4f}")
```

## Troubleshooting

**Out of memory?**
- Reduce `--tensor_parallel_size`
- Reduce `--gpu_memory_utilization`
- Use a smaller model

**Model loading fails?**
- Check you have access to the model (some require HF tokens)
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

**Slow generation?**
- Increase `--tensor_parallel_size` if you have multiple GPUs
- Reduce `--iterations` for faster testing

## Contact

For issues or questions, please open an issue on the repository.
