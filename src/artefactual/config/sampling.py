"""Sampling configuration classes."""

import dataclasses
from typing import Annotated

from beartype.vale import Is
from etils import edc
from vllm.sampling_params import RequestOutputKind  # pytype: disable=import-error

MIN_VAL = 1e-2


@edc.dataclass
@dataclasses.dataclass
class SamplingConfig:
    """Configuration for sampling parameters in text generation."""

    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    ignore_eos: bool = False
    max_tokens: int = 1024
    min_tokens: int = 0
    logprobs: int | None = 0
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE


@edc.dataclass
@dataclasses.dataclass
class MultipleGenerationConfig:
    """Configuration for multiple generation runs with varying parameters."""

    min_val: Annotated[float, Is[lambda x: x > MIN_VAL]] = 1.0
    max_val: float = 1.0
    n_samples: int = 1
    seed: int = 42

    def __post_init__(self):
        if self.min_val > self.max_val:
            msg = "min_val can't be larger than max_val"
            raise ValueError(msg)
