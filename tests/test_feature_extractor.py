from pathlib import Path

import numpy as np

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from examples.mock_vllm import load_json

RESPONSES = load_json(Path(__file__).parent.parent / "examples/wepr_demo_responses.json")


PARSED = LogProbParser().transform(RESPONSES)
# print(f"Parsed logprobs: {PARSED}")
assert isinstance(PARSED, np.ndarray), "Parsed output should be a list"
assert len(PARSED) == 1, "Parsed output should contain one item"
for item in PARSED:
    assert isinstance(item, np.ndarray), "Each item in parsed output should be a dictionary"


EPR_FEATURES = EntropyTransformer(k=5, reduction="epr").transform(PARSED)
assert 0 < EPR_FEATURES.shape[0] < 15, f"Expected between 0 and 15 rows, but got {EPR_FEATURES.shape[0]}"
assert EPR_FEATURES.shape[1] == 1, f"Expected 1 column (one per output), but got {EPR_FEATURES.shape[1]}"
# print(f"EPR features: {EPR_FEATURES}")


WEPR_FEATURES = EntropyTransformer(k=5, reduction="wepr").transform(PARSED)
# print(f"WEPR features: {WEPR_FEATURES}")
# print(f"WEPR features shape: {WEPR_FEATURES.shape}")
