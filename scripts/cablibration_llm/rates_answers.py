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


DEFAULT_PROMPT = """
You are tasked with evaluating how closely a generated answer matches the provided golden answer:

1. Evaluate Core Factual Content Only:
- Focus solely on comparing the core factual content of the generated answer against the golden answer.
- Key facts (dates, names, numbers, relationships, etc.) must exactly match the golden answer. Any deviation in these critical details is unacceptable.
- Extra explanatory details are acceptable only if they do not contradict or obscure the core answer.

2. Identify and Classify Errors:
    - Major Error:
        - Occurs when the generated answer contradicts any key fact or core detail of the golden answer (e.g., an incorrect date, name, number, or relationship).
        - Occurs when the generated answer fails to provide the required core answer—such as asking questions or offering irrelevant commentary—instead of stating the answer.
    - Minor Error:
        - Occurs when the generated answer has slight wording differences, includes extra correct details, or omits non-critical elements without changing the overall correctness.
        - Minor formatting or stylistic differences (e.g., punctuation or extra adjectives) that do not alter meaning should be ignored.

3. Specific Considerations:
    - Numeric and Date Accuracy: Any numerical values or dates must match exactly. Even a one-year or several-day discrepancy is a major error.
    - Entity and Name Precision: All key names or entities provided in the golden answer must appear correctly in the generated answer. Substituting or omitting a key entity is a major error.
    - Irrelevant Content: If the generated answer includes unrelated content (e.g., asking for more details or providing instructions on contacting someone) rather than delivering the required answer, this is a major error.

4. Scoring Guidelines:
    - Assign a similarity rating on a scale from 0 to 5:
        - 0: Completely dissimilar to the golden answer.
        - 1: Mostly dissimilar, with only minimal overlapping elements.
        - 2: Partially similar but with significant discrepancies.
        - 3: Mostly similar, with some noticeable discrepancies.
        - 4: Similar with only minor issues or omissions.
        - 5: Perfectly similar and fully consistent with the golden answer.

5. Provide a Clear, Concise Explanation:
    - Detail the specific similarities and discrepancies between the generated answer and the golden answer.
    - Clearly indicate any major or minor errors (e.g., incorrect dates, names, numbers, or inclusion of irrelevant content).
    - Keep your explanation focused solely on the factual similarity between the generated answer and the golden answer.

Your evaluation must be objective, detailed, and exclusively focused on ensuring that all key details in the generated answer exactly match those in the golden answer while dismissing any non-essential or unrelated information.

Golden Answer:
{{ golden_answer }}

Generated Answer:
{{ generated_answer }}
"""


@edc.dataclass
@dataclasses.dataclass
class PromptConfig:
    prompt: str | None = None
    template: Template = field(init=False)

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = DEFAULT_PROMPT

        env = Environment(autoescape=True)
        self.template = env.from_string(self.prompt)

    def render(self, question: str, golden_answer: str, generated_answer: str) -> str:
        return self.template.render(question=question, golden_answer=golden_answer, generated_answer=generated_answer)


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
    explanation: str
    rating: int


def main(cfg: AppConfig):
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

    generator = generate.json(model, Response)

    output_file = cfg.output_dir / f"ratings_{cfg.model.model}_{cfg.source.name}".replace("/", "_")
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
                    break

                ids, questions, responses, answers = zip(
                    *tlz.pluck(["id", "question", "response", "answer"], samples), strict=True
                )
                prompts = [
                    cfg.prompt.render(question=question, golden_answer=golden_answer, generated_answer=generated_answer)
                    for question, golden_answer, generated_answer in zip(questions, answers, responses, strict=True)
                ]
                try:
                    requests = generator(prompts, sampling_params=sampling_params)
                except ValidationError:
                    msg = "Error validating generated output"
                    logging.exception(msg)
                    continue
                ratings, explanations = zip(*((req.rating, req.explanation) for req in requests), strict=False)
                samples = {"id": ids, "rating": ratings, "explanation": explanations}
                df = pl.DataFrame(samples)
                df.write_ndjson(dst)
    logging.info("Wrote results in %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
