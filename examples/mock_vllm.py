# import modules
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

# These dataclasses mimic the vLLM RequestOutput structure
# based on the JSON schema in 'examples/wepr_demo_responses.json'.


@dataclass(frozen=True)
class DemoLogprob:
    decoded_token: str
    logprob: float
    rank: int


@dataclass(frozen=True)
class DemoCompletionOutput:
    text: str
    token_ids: list[int]
    logprobs: list[dict[int, DemoLogprob]]  # one dict per generated position, keyed by token id


@dataclass(frozen=True)
class DemoRequestOutput:
    prompt: str
    outputs: list[DemoCompletionOutput]

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        # walks the JSON dict into the dataclass tree
        outputs = []
        for output_dict in payload["outputs"]:
            completion_logprobs = []
            for position_dict in output_dict["logprobs"]:
                candidates_at_this_pos = {}
                for token_id, details in position_dict.items():
                    candidate = DemoLogprob(
                        decoded_token=details["decoded_token"], logprob=details["logprob"], rank=details["rank"]
                    )
                    candidates_at_this_pos[int(token_id)] = candidate
                completion_logprobs.append(candidates_at_this_pos)
            completion_output = DemoCompletionOutput(
                text=output_dict["text"], token_ids=output_dict["token_ids"], logprobs=completion_logprobs
            )
            outputs.append(completion_output)
        return cls(prompt=payload["prompt"], outputs=outputs)


def load_json(path: Path) -> list[DemoRequestOutput]:
    """Read the fixture JSON file and return the list of mock RequestOutput objects."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [DemoRequestOutput.from_dict(item) for item in data["responses"]]
