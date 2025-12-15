# Artefactual

Artefactual is a library for LLM response calibration and analysis, focusing on entropy-based uncertainty detection methods like EPR (Entropy Production Rate) and WEPR (Weighted EPR).

## Installation

Artefactual is designed to be lightweight by default. You can install the core package (scoring methods only) or include heavy dependencies for calibration and model execution.

### Core Installation (Lightweight)
For using the scoring metrics (`EPR`, `WEPR`) with pre-computed logprobs or OpenAI API responses:

```bash
pip install artefactual
```

### Full Installation (Calibration & vLLM)
If you need to run local models (via vLLM), perform calibration, or retrain weights:

```bash
pip install "artefactual[calibration]"
```

## Usage

### Basic Scoring

```python
from artefactual.scoring import EPR, WEPR

# Initialize scorers
epr = EPR()
wepr = WEPR(model="mistralai/Ministral-8B-Instruct-2410")

# Compute scores on your model outputs
# Supports OpenAI format, vLLM RequestOutput, etc.
# scores = wepr.compute(outputs)
```
