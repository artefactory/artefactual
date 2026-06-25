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


EPR_SEQUENCE_FEATURES = EntropyTransformer(k=5, reduction="epr").transform(PARSED)
assert 0 < EPR_SEQUENCE_FEATURES.shape[0] < 15, "Expected between 0 and 15 rows"
assert EPR_SEQUENCE_FEATURES.shape[1] == 1, "Expected 1 column (one per output)"
# print(f"EPR sequence level features: {EPR_SEQUENCE_FEATURES}")

EPR_TOKEN_FEATURES = EntropyTransformer(k=5, reduction="epr").transform_tokens(PARSED)
# print(f"EPR token level features: {EPR_TOKEN_FEATURES}")


WEPR_SEQUENCE_FEATURES = EntropyTransformer(k=5, reduction="wepr").transform(PARSED)
# print(f"WEPR sequence level features: {WEPR_SEQUENCE_FEATURES}")
# print(f"WEPR features shape: {WEPR_SEQUENCE_FEATURES.shape}")

WEPR_TOKEN_FEATURES = EntropyTransformer(k=5, reduction="wepr").transform_tokens(PARSED)
# print(f"WEPR token level features: {WEPR_TOKEN_FEATURES}")
