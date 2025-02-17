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
import re

import orjson as json
import polars as pl
import toolz as tlz
from absl import app, logging
from beartype import beartype
from etils import eapp, edc, epath, etqdm
from jinja2 import Environment
from outlines import generate, models
from pydantic import BaseModel, Field, ValidationError, conint
from simple_parsing import field, subgroups
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


@edc.dataclass
@dataclasses.dataclass
class SamplingConfig:
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
class AppConfig:
    source: epath.Path
    model: ModelConfig = subgroups({"default": ModelConfig, "mistral": MistralModelConfig}, default="default")
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")
    sampling_params: SamplingConfig = field(default_factory=SamplingConfig)
    limit: int | None = None
    batch_size: int = 64
    resume: bool = False


RATING_PATTERN = re.compile(r"<rating>(.*?)</rating>")
EXPLANATION_PATTERN = re.compile(r"<explanation>(.*?)</explanation>")


class Response(BaseModel):
    explanation: str
    rating: int


def main(cfg: AppConfig):
    # TODO: Add logging
    logging.info("\n%s", cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sampling_params = SamplingParams(**dataclasses.asdict(cfg.sampling_params))
    logging.debug("Make Embedder")
    llm = LLM(**dataclasses.asdict(cfg.model))
    model = models.VLLM(llm)
    output_file = cfg.output_dir / f"scores_{cfg.model.model}_{cfg.source.name}.json".replace("/", "_")
    logging.debug("Output file %s", output_file)
    if output_file.exists():
        msg = f"{output_file} already exists"
        raise FileExistsError(msg)

    env = Environment(autoescape=True)
    prompt = env.from_string("""You will be given a question, a golden_answer and a generated_answer to read and understand.
    Your task is to evaluate how similar the generated_answer and the golden_answer are.
    You must analyze each element of the golden_answer and rates if the generated_answer provides the same elements.
    Answers are similar if the dates, the names, the location, the facts etc. are exactly the same between the answers.
    You must grade how similar the generated_answer is to the golden_answer on an integer scale from 0 to 5.
    0 means the generated_answer are differents the golden_answer, and 5 means the generated_answer and the golden_answer provides the same answer to the question.
    To help you in your task, you must know the golden_answer is the true reponse to the question and the generated_answer is a candidate answer to the question.
    You must first provide a short reasoning about your rating then rate the similarity between the golden_answer and the generated_answer.

    Here the question, golden_answer and the generated_answer.

    <question>{{ question }}</question>
    <golden_answer>{{ golden_answer }}</golden_answer>
    <generated_answer>{{ generated_answer }}</generated_answer>

    <think>\n""")

    generator = generate.json(model, Response)

    output_file = cfg.source.parent / f"ratings_{cfg.model.model}_{cfg.source.name}".replace("/", "_")
    if output_file.exists():
        if not cfg.resume:
            msg = f"Output file {output_file} already exists"
            raise ValueError(msg)
        else:
            with output_file.open("r") as src:
                samples = map(json.loads, src)
                existing_ids = set(tlz.pluck("id", samples))
    else:
        existing_ids = {}

    with cfg.source.open("r") as src:
        with output_file.open("a") as dst:
            while etqdm.tqdm(src):
                lines = tlz.take(cfg.batch_size, src)
                samples = map(json.loads, lines)
                samples = tlz.filter(lambda s: s["id"] not in existing_ids, samples)
                try:
                    _, samples = tlz.peek(samples)
                except StopIteration:
                    continue

                ids, questions, responses, answers = zip(*tlz.pluck(["id", "question", "response", "answer"], samples))
                prompts = [
                    prompt.render(question=question, golden_answer=golden_answer, generated_answer=generated_answer)
                    for question, golden_answer, generated_answer in zip(questions, answers, responses)
                ]
                try:
                    requests = generator(prompts, sampling_params=sampling_params)
                except ValidationError as e:
                    msg = "Error validating generated output"
                    logging.exception(msg)
                    continue
                ratings, explanations = zip(*((req.rating, req.explanation) for req in requests))
                samples = {"id": ids, "rating": ratings, "explanation": explanations}
                df = pl.DataFrame(samples)
                df.write_ndjson(dst)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
