# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "absl-py",
#     "etils[eapp,edc,epath,etqdm]",
#     "beartype",
#     "jinja2",
#     "toolz",
#     "clu",
#     "huggingface-hub",
#     "grain",
#     "wandb",
#     "safetensors",
#     "numpy",
#     "vllm>=0.7",
#     "polars",
#     "tensorflow-datasets",
#     "orjson",
#     "outlines[vllm]",
#     "pydantic",
# ]
# ///


import dataclasses
from typing import Any

import orjson as json
import polars as pl
import toolz as tlz
from absl import app, logging
from etils import eapp, edc, epath, etqdm
from jinja2 import Environment, Template
from outlines import generate, models
from pydantic import BaseModel, ValidationError
from simple_parsing import field, subgroups
from vllm import LLM, SamplingParams  # pytype: disable=import-error
from vllm.sampling_params import RequestOutputKind  # pytype: disable=import-error


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
    seed: int | None = 42
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    ignore_eos: bool = False
    max_tokens: int = 1024
    min_tokens: int = 0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE


@edc.dataclass
@dataclasses.dataclass
class ModelConfig:
    """Configuration for a language model."""

    model: str
    task: str = "generate"
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    max_model_len: int = 8192


@edc.dataclass
@dataclasses.dataclass
class MistralModelConfig(ModelConfig):
    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"


@edc.dataclass
@dataclasses.dataclass
class LLamaModelConfig(ModelConfig):
    model: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
    tensor_parallel_size: int = 2


TRUE_FALSE_PROMPT = """
You are an expert evaluator tasked with determining if a provided answer correctly responds to a query based on an expected answer. Your task is to make a binary True/False judgment.

Query:
{{ query }}

Expected Answer:
{{ expected_answer }}

Provided Answer:
{{ provided_answer }}

Instructions:
1. Focus solely on comparing the core factual content of the provided answer against the expected answer.
2. Key facts (dates, names, numbers, relationships, etc.) must match the expected answer. Significant deviations in critical details should result in a "False" judgment.
3. Minor formatting or stylistic differences that do not alter meaning can be ignored.
4. The judgment should be "True" only if the provided answer conveys the same essential information as the expected answer.
5. The judgment should be "False" if:
   - The provided answer contradicts the expected answer
   - The provided answer is significantly incomplete
   - The provided answer includes major factual errors
   - The provided answer fails to address the query directly

After careful analysis, provide:
1. A binary judgment: True or False
2. A brief explanation (1-3 sentences) justifying your judgment

Remember: Your goal is to determine factual correctness, not stylistic similarity. Focus on whether the provided answer would give a user the correct information they need.
"""


@edc.dataclass
@dataclasses.dataclass
class PromptConfig:
    prompt: str | None = None
    template: Template = field(init=False)

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = TRUE_FALSE_PROMPT

        env = Environment(autoescape=True)
        self.template = env.from_string(self.prompt)

    def render(self, query: str, expected_answer: str, provided_answer: str) -> str:
        return self.template.render(query=query, expected_answer=expected_answer, provided_answer=provided_answer)


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    source: epath.Path
    model: ModelConfig = subgroups(
        {
            "default": ModelConfig,
            "mistral": MistralModelConfig,
            "llama-3.1-70b": LLamaModelConfig,
        },
        default="default",
    )
    prompt: PromptConfig = edc.field(default_factory=PromptConfig)
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")
    sampling_params: SamplingConfig = field(default_factory=SamplingConfig)
    limit: int | None = None
    batch_size: int = 64
    resume: bool = False


class Response(BaseModel):
    """Model for the LLM's judgment response."""

    judgment: bool  # True or False judgment
    explanation: str  # Explanation of the judgment


def expand_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each sample by creating one entry per generated answer."""
    expanded_samples = []

    for sample in samples:
        query_id = sample.get("query_id", sample.get("id"))
        query = sample["query"]

        # Handle expected_answer as string or list
        expected_answer = sample["expected_answer"]
        if isinstance(expected_answer, list):
            expected_answer = " ".join(expected_answer)

        # Process each generated answer
        for gen_answer_dict in sample["generated_answers"]:
            # Extract the answer text from the dictionary
            gen_key = next(iter(gen_answer_dict.keys()))
            provided_answer = gen_answer_dict[gen_key]

            # Create new sample with the sub id
            expanded_sample = {
                "query_id": query_id,
                "sub_id": gen_key,
                "query": query,
                "expected_answer": expected_answer,
                "provided_answer": provided_answer,
            }
            expanded_samples.append(expanded_sample)

    return expanded_samples


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sampling_params = SamplingParams(**dataclasses.asdict(cfg.sampling_params))

    logging.debug("Make Embedder")
    llm = LLM(**dataclasses.asdict(cfg.model))
    model = models.VLLM(llm)

    output_file = cfg.output_dir / f"judgments_{cfg.model.model}_{cfg.source.name}".replace("/", "_")
    logging.debug("Output file %s", output_file)

    if output_file.exists():
        if not cfg.resume:
            msg = f"Output file {output_file} already exists"
            raise ValueError(msg)
        else:
            with output_file.open("r") as src:
                samples = map(json.loads, src)
                # Track both query_id and sub_id to avoid duplicates
                existing_ids = {(s["query_id"], s["sub_id"]) for s in samples}
    else:
        existing_ids = set()

    generator = generate.json(model, Response)

    with cfg.source.open("r") as src:
        with output_file.open("a") as dst:
            # Top level iterator with tqdm for tracking progress
            for _ in etqdm.tqdm(range(cfg.limit or float("inf"))):
                lines = list(tlz.take(cfg.batch_size, src))
                if not lines:
                    break

                # Parse raw samples
                raw_samples = list(map(json.loads, lines))

                # Expand samples - one entry per generated answer
                expanded_samples = expand_samples(raw_samples)

                # Filter out already processed samples
                expanded_samples = [s for s in expanded_samples if (s["query_id"], s["sub_id"]) not in existing_ids]

                if not expanded_samples:
                    continue

                # Prepare data for batch processing
                prompts = [
                    cfg.prompt.render(
                        query=sample["query"],
                        expected_answer=sample["expected_answer"],
                        provided_answer=sample["provided_answer"],
                    )
                    for sample in expanded_samples
                ]

                # Get model judgments
                try:
                    responses = generator(prompts, sampling_params=sampling_params)

                    # Create results
                    results = []
                    for sample, response in zip(expanded_samples, responses, strict=True):
                        result = {
                            "query_id": sample["query_id"],
                            "sub_id": sample["sub_id"],
                            "judgment": response.judgment,
                            "explanation": response.explanation,
                        }
                        results.append(result)
                        existing_ids.add((sample["query_id"], sample["sub_id"]))

                    # Write results
                    if results:
                        df = pl.DataFrame(results)
                        df.write_ndjson(dst)

                except ValidationError:
                    msg = "Error validating generated output"
                    logging.exception(msg)
                    continue
                except Exception as e:
                    logging.exception(f"Unexpected error: {e}")
                    continue

    logging.info("Wrote results in %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
