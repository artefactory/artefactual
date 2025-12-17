
**Artefactual**

Artefactual is a lightweight Python package for measuring model hallucination risk using entropy-based metrics. It provides two primary uncertainty detectors:

- **EPR (Entropy Production Rate)**: a token- and sequence-level entropy-based metric exposing raw and (optionally) calibrated probabilities.
- **WEPR (Weighted EPR)**: a calibrated, learned weighted combination of entropy contributions yielding sequence- and token-level probabilities of hallucination.

The library includes pre-computed calibration coefficients and weights for a set of popular models so data scientists can use EPR/WEPR out-of-the-box without running a calibration pipeline.

**Installation**

- **Minimal (core) install** — For most users who only want to compute EPR/WEPR using the precomputed files shipped in the package:

```bash
uv sync
# or for editable development install:
uv pip install -e .
```

- **With calibration (full) install** — If you plan to run the calibration pipeline or train WEPR/EPR coefficients, install the `calibration` extra to pull heavier ML tooling and platform-specific dependencies:

```bash
uv pip install -e '.[calibration]'
# or non-editable:
uv pip install '.[calibration]'
```

Notes:
- The `calibration` extra is defined in `pyproject.toml` under `[project.optional-dependencies]` and includes packages such as `scikit-learn`, `ray`, `matplotlib`, `vllm` (platform-gated), and `torch`.
- If you only need the included precomputed calibration/weights files in `src/artefactual/data`, the minimal install is sufficient and much lighter.

**Why Artefactual?**

- **Practical**: Precomputed calibration for several model families is included in `src/artefactual/data` and can be used by model name.
- **Flexible**: Works with vLLM, OpenAI Chat Completions, and the OpenAI Responses API formats.
- **Detailed outputs**: Compute both sequence-level and token-level uncertainty scores to power downstream pipelines (e.g., answer filtering, reranking, human-in-the-loop triggers).

**Contents**

- **`src/artefactual/scoring/entropy_methods/epr.py`** — EPR implementation with `EPR.compute(...)` and `EPR.compute_token_scores(...)`.
- **`src/artefactual/scoring/entropy_methods/wepr.py`** — WEPR implementation with `WEPR.compute(...)` and `WEPR.compute_token_scores(...)`.
- **`src/artefactual/utils/io.py`** — convenience loaders `load_weights(...)` and `load_calibration(...)` and internal registry mapping model ids to JSON files.
- **Precomputed data**: see `src/artefactual/data/*.json` (e.g. `weights_ministral.json`, `calibration_ministral.json`).
- **Examples**: `examples/epr_usage_demo.py` and `examples/calibration_script.py` demonstrate usage and the calibration pipeline.

**Quick Start**

1) Install

```bash
uv sync
# or (for development):
uv pip install -e .
```

2) Basic usage (sequence-level scores)

```python
from artefactual.scoring import EPR, WEPR

# Use precomputed calibration (model keys are defined in the registry)
epr = EPR(pretrained_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")
wepr = WEPR("mistralai/Ministral-8B-Instruct-2410")

# Example: using an OpenAI Responses-like structure (minimal illustrative example)
fake_responses = {
	"object": "response",
	"output": [
		{
			"content": [
				{
					"logprobs": [
						{"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.3}]},
						{"top_logprobs": [{"logprob": -0.05}, {"logprob": -3.1}]}
					]
				}
			]
		}
	]
}

# Compute sequence-level calibrated probabilities (list of floats)
seq_scores_epr = epr.compute(fake_responses)
seq_scores_wepr = wepr.compute(fake_responses)

# Compute token-level scores (list of numpy arrays)
token_scores_epr = epr.compute_token_scores(fake_responses)
token_scores_wepr = wepr.compute_token_scores(fake_responses)

print("EPR sequence scores:", seq_scores_epr)
print("WEPR sequence scores:", seq_scores_wepr)
```

Notes:
- `EPR(pretrained_model_name_or_path=...)` attempts to load calibration coefficients via `artefactual.utils.io.load_calibration`.
- `WEPR(pretrained_model_name_or_path)` requires a weight source (either a known model key from the registry or a local JSON file) and will raise a `ValueError` if weights cannot be found.

**Registry / Precomputed files**

Artefactual ships a small registry which maps canonical model identifiers to precomputed JSON files. These mappings are available in `src/artefactual/utils/io.py` under `MODEL_WEIGHT_MAP` and `MODEL_CALIBRATION_MAP`.

You can pass one of those strings directly to `EPR` or `WEPR` constructors (e.g., `EPR(pretrained_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")`). Under the hood the package reads `src/artefactual/data/<file>.json` via `importlib.resources`.

If you prefer to provide a custom calibration or weight file, pass a filesystem path (e.g., `WEPR('/path/to/my_weights.json')`). See `artefactual.utils.io.load_weights` and `load_calibration` for the exact behavior.

**API Reference (minimal)**

- `EPR(pretrained_model_name_or_path: str, k: int = 15)`
  - `compute(outputs) -> list[float]` — Returns sequence-level EPR scores (calibrated to probabilities).
  - `compute_token_scores(outputs) -> list[np.ndarray]` — Token-level EPR contributions.

- `WEPR(pretrained_model_name_or_path: str)`
  - `compute(outputs) -> list[float]` — Sequence-level calibrated WEPR probabilities.
  - `compute_token_scores(outputs) -> list[np.ndarray]` — Token-level calibrated probabilities.

Current supported input formats (the library will auto-detect):

- vLLM `RequestOutput` lists
- OpenAI classic ChatCompletion (`choices` with `logprobs`)
- OpenAI Responses API (`object: "response"`, `output` with `content` parts containing `logprobs`)

For exact parsing rules see `src/artefactual/preprocessing/openai_parser.py` and `src/artefactual/preprocessing/vllm_parser.py`.

**Examples and Convenience Files**

- `examples/epr_usage_demo.py`: a small demonstration of computing EPR on sample inputs.
- `sample_qa_data.json` and `outputs/sample_qa_data_Ministral-8B-Instruct-2410_entropy.json`: example data and exported results used for demonstrations and tests.

**Advanced: Calibration pipeline (for deep usage)**

The package includes a calibration pipeline for generating the `weights_*.json` and `calibration_*.json` files. This is intended for users who want to train new WEPR/EPR coefficients for a custom model or dataset. The pipeline is non-trivial and is documented here at a high level; see `scripts/calibration_llm` for runnable scripts.

High-level steps:

1. Generate a dataset of outputs with full top-K logprobs per token from your target model (vLLM or OpenAI Responses API). The scripts expect a structure that preserves, for each generated token, the top-K `logprob` values.
2. Run the `scripts/calibration_llm/generate_responses.py` script to create evaluation outputs for a calibration dataset.
3. Extract entropy contributions and dataset-level features using `scripts/calibration_llm/entropic_scoring.py`.
4. Fit logistic/linear calibration models using `scripts/calibration_llm/calibration.py` and `scripts/calibration_llm/score_responses.py`. This will produce JSON weight files of the same shape as the ones stored under `src/artefactual/data`.
5. Add the produced `weights_*.json` or `calibration_*.json` to the package data registry and optionally create a PR to include it upstream.

Important notes for calibration:

- The calibration pipeline assumes access to ground-truth labels (e.g., binary hallucination labels at sequence or token level) to learn the mapping from entropy features to probabilities.
- WEPR training learns two sets of coefficients: `mean_rank_i` (for per-token contributions) and `max_rank_i` (for sequence-level max contributions). EPR calibration is a simpler scalar intercept + single coefficient for mean entropy.
- See `scripts/calibration_llm/plot_calibration.py` and `scripts/calibration_llm/plot_entropy.py` for visualization utilities used during development.

**Testing**

Run the package tests using pytest:

```bash
pytest -q
```
