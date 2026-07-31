# Artefactual

Artefactual is a lightweight Python package for measuring model hallucination risk using entropy-based metrics. It is:

- **Practical**: Precomputed calibration for several model families is included in `src/artefactual/data` and can be used by model name.
- **Flexible**: Works with OpenAI Chat Completions and OpenAI Responses payloads, as objects or plain mappings, one at a time or as a batch.
- **Detailed outputs**: Compute both sequence-level and token-level uncertainty scores to power downstream pipelines (e.g., answer filtering, reranking, human-in-the-loop triggers).

The package provides two primary uncertainty detectors:

- **EPR (Entropy Production Rate)**: a token- and sequence-level entropy-based metric exposing raw and (optionally) calibrated probabilities.
- **WEPR (Weighted EPR)**: a calibrated, learned weighted combination of entropy contributions yielding sequence- and token-level probabilities of hallucination.

The library includes pre-computed calibration coefficients and weights for a set of popular models so data scientists can use EPR/WEPR out-of-the-box without running a calibration pipeline.

## Installation

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
*Mac users:* the calibration extra omits the GPU generation backend (no Darwin wheels). Running the full calibration pipeline on Mac requires an alternative generation backend.

*Note*: Typical packages included in this installation method are `scikit-learn` (training), a local generation backend, `ray` (optional distributed processing), `pandas`, `numpy`, and `tqdm`. Installing these may require system-level libraries or CUDA support depending on your environment.

## Quickstart

To try the examples instantly (no GPU or model downloads required) run the following commands :

```bash
uv sync
uv run jupyter lab examples/epr_usage_demo.ipynb
uv run jupyter lab examples/wepr_usage_demo.ipynb
```

## Basic usage (sequence-level scores)

### EPR example:

```python
from artefactual.preprocessing import parse_top_logprobs
from artefactual.scoring import EPR

# Use precomputed calibration (model keys are defined in the registry)
epr = EPR(pretrained_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")

# Compute sequence-level calibrated probabilities (list of floats)
parsed_logprobs = parse_top_logprobs(response)  # an OpenAI completion response, or a list of them
seq_scores_epr = epr.compute(parsed_logprobs)

# Compute token-level scores (list of numpy arrays)
token_scores_epr = epr.compute_token_scores(parsed_logprobs)

print("EPR sequence scores:", seq_scores_epr)
```

For a runnable demo run `examples/epr_usage_demo.ipynb`.

### WEPR example:

```python
from artefactual.scoring import WEPR

# WEPR requires a weight source (model key or local weights file)
wepr = WEPR(pretrained_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")

# Compute sequence-level calibrated probabilities (list of floats)
parsed_logprobs = parse_top_logprobs(response)  # an OpenAI completion response, or a list of them
seq_scores_wepr = wepr.compute(parsed_logprobs)

# Compute token-level scores (list of numpy arrays)
token_scores_wepr = wepr.compute_token_scores(parsed_logprobs)

print("WEPR sequence scores:", seq_scores_wepr)
```
For a runnable demo run `examples/wepr_usage_demo.ipynb`.

In both examples, the `response` object can have the following structure :

```python
# Example: using an OpenAI Responses-like structure (minimal illustrative example, not runnable)
response = {
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
```

*Notes*:
- `EPR(pretrained_model_name_or_path=...)` attempts to load calibration coefficients via `artefactual.utils.io.load_calibration` and will silently fall back to uncalibrated raw EPR scores if calibration is not found.
- `WEPR(pretrained_model_name_or_path)` requires a weight source (either a known model key from the registry or a local JSON file) and will raise a `ValueError` if weights cannot be found.
- Both `EPR.compute(...)` and `WEPR.compute(...)` return lists because the methods accept batch-style inputs (the top-level structure may contain multiple response objects). If you pass a single response object you'll receive a single-element list — index the first element (for example, `seq_scores_epr[0]` or `seq_scores_wepr[0]`) to obtain a single float probability.

### Further Examples

Some examples and dummy scripts are available :
* `examples/epr_usage_demo.ipynb` — EPR usage demo
* `examples/wepr_usage_demo.ipynb` — WEPR usage demo
* `examples/langfuse_integration_demo.ipynb` — scoring Langfuse traces

## Calibration logic

When possible, we strongly recommend to use calibrated detectors, so that outputs can be interpreted as probabilities. We describe below how to load existing weights, or to run the full pipeline on a new model and/or corpus.

### Registry / Precomputed files

Artefactual ships a small registry which maps canonical model identifiers to precomputed JSON files. These mappings are available in `src/artefactual/utils/io.py` under `MODEL_WEIGHT_MAP` and `MODEL_CALIBRATION_MAP`.

You can pass one of those strings directly to `EPR` or `WEPR` constructors (e.g., `EPR(pretrained_model_name_or_path="mistralai/Ministral-8B-Instruct-2410")`). Under the hood the package reads `src/artefactual/data/<file>.json` via `importlib.resources`.

If you prefer to provide a custom calibration or weight file, pass a filesystem path (e.g., `WEPR('/path/to/my_weights.json')`). See `artefactual.utils.io.load_weights` and `load_calibration` for the exact behavior.

### Advanced: Calibration pipeline (for deep usage)

The calibration pipeline in this package produces the `weights_*.json` and `calibration_*.json` files used to turn raw entropy scores into calibrated probabilities.

The implemented flow (the generation and rating modules live under `src/artefactual/calibration`; training lives in `scripts/`) is:

1. Prepare a QA dataset of question/answers (e.g., `web_question_qa.json`) containing entries like:

   {
	   "question": "where is roswell area 51?",
	   "question_id": "d204f08c-fbcb-41cb-8e55-ee3879d68eea",
	   "short_answer": "Roswell",
	   "answer_aliases": []
   }

2. Run the generation utility `src/artefactual/calibration/outputs_entropy.py` to produce a JSON dataset that includes EPR/WEPR scores for each generated answer (this JSON contains `generated_answers` entries with an `epr_score`/`wepr_score` field).

3. Use `src/artefactual/calibration/rates_answers.py` to have a judge LLM label each generated answer as `True`/`False` (correct/incorrect). This script produces a pandas DataFrame (or CSV) where each row contains `uncertainty_score` (EPR/WEPR) and `judgment` (the target).

4. Train a calibration model with `scripts/train_calibration.py`:

```bash
uv run python scripts/train_calibration.py \
    --responses responses.json --labels labels.json --reduction epr
```

   It takes the raw completion responses and one 0/1 label per generated sequence (1 for hallucination), and fits the same parser/entropy/classifier pipeline `epr()` and `wepr()` build — with the logistic regression fitted on your data instead of loaded from weights. The fitted intercept and coefficients are logged.

*Important notes for calibration*:

- The pipeline requires to use a LLM-as-a-judge, which can be chosen by the user (default is "mistralai/Ministral-8B-Instruct-2410").
- WEPR training learns multiple coefficient groups (e.g., `mean_rank_i` and `max_rank_i`) while EPR calibration is a single-intercept plus mean-entropy coefficient.
- See the modules under `src/artefactual/calibration` for implementation details and plotting utilities.

## Citation

If you consider `artefactual` or any of its feature useful for your research, consider citing our paper, accepted for publication at ECIR 2026:

```
@misc{moslonka2025learnedhallucinationdetectionblackbox,
      title={Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate},
      author={Charles Moslonka and Hicham Randrianarivo and Arthur Garnier and Emmanuel Malherbe},
      year={2025},
      eprint={2509.04492},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04492},
}
```

## License

The use of this software is under the MIT license, with no limitation of usage, including for commercial applications.
