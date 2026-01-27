# Artefactual

**Artefactual** is a lightweight Python package for hallucination detection and robustness in LLM responses.

## Features

- **Practical**: Precomputed calibration for several model families
- **Flexible**: Works with vLLM, OpenAI Chat Completions, and the OpenAI Responses API formats
- **Detailed outputs**: Compute both sequence-level and token-level uncertainty scores

## Uncertainty Detectors

- **EPR (Entropy Production Rate)**: Token- and sequence-level entropy-based metric exposing raw and calibrated probabilities
- **WEPR (Weighted EPR)**: Calibrated, learned weighted combination of entropy contributions

```{toctree}
:maxdepth: 2

api
```
