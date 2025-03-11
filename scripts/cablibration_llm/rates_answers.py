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
    """Model configuration for Mistral models."""

    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"


@edc.dataclass
@dataclasses.dataclass
class LLamaModelConfig(ModelConfig):
    """Model configuration for Llama models."""

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
    """Configuration for prompt templating."""

    prompt: str | None = None
    template: Template = field(init=False)

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = TRUE_FALSE_PROMPT

        env = Environment(autoescape=True)
        self.template = env.from_string(self.prompt)

    def render(self, query: str, expected_answer: str, provided_answer: str) -> str:
        """Render the prompt template with the given parameters.

        Args:
            query: The question or query to evaluate
            expected_answer: The reference/correct answer
            provided_answer: The generated answer to evaluate

        Returns:
            Formatted prompt string
        """
        return self.template.render(query=query, expected_answer=expected_answer, provided_answer=provided_answer)


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    """Application configuration for the judgment pipeline.

    This class defines all the parameters needed to run the judgment pipeline,
    including input/output paths, model configuration, and processing parameters.
    """

    # Input/output configuration
    source: epath.Path  # Path to input data file
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")  # Directory for output files
    resume: bool = False  # Whether to resume from existing output

    # Model configuration
    model: ModelConfig = subgroups(
        {
            "default": ModelConfig,
            "mistral": MistralModelConfig,
            "llama-3.1-70b": LLamaModelConfig,
        },
        default="default",
    )

    # Prompt and sampling configuration
    prompt: PromptConfig = edc.field(default_factory=PromptConfig)
    sampling_params: SamplingConfig = field(default_factory=SamplingConfig)

    # Processing parameters
    limit: int | None = None  # Maximum number of batches to process
    batch_size: int = 64  # Number of samples to process in each batch


class Response(BaseModel):
    """Model for the LLM's judgment response."""

    judgment: bool  # True or False judgment
    explanation: str  # Explanation of the judgment


# === Data Processing Functions ===


def expand_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each sample by creating one entry per generated answer.

    Takes samples with multiple generated answers and creates individual samples
    for each answer, adding a sub_id to track the specific generation.

    Args:
        samples: List of samples with query, expected_answer, and generated_answers

    Returns:
        List of expanded samples with one entry per generated answer
    """
    expanded_samples = []

    for sample in samples:
        # Get sample identification info
        query_id = sample.get("query_id", -1)
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


# === Helper Functions for Main Application ===


def setup_output_file(cfg: AppConfig) -> tuple[epath.Path, set[tuple[Any, str]]]:
    """Set up the output file and load existing IDs if resuming.

    Args:
        cfg: Application configuration

    Returns:
        Tuple containing:
        - Path to the output file
        - Set of existing IDs to avoid duplicates
    """
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file path
    output_file = cfg.output_dir / f"judgments_{cfg.model.model}_{cfg.source.name}".replace("/", "_")
    logging.debug("Output file: %s", output_file)

    # Check if output file exists
    if output_file.exists():
        if not cfg.resume:
            response = input(f"Output file {output_file} already exists. Continue? (y/n): ")
            if response.lower() != "y":
                msg = f"Output file {output_file} already exists"
                raise ValueError(msg)
            else:
                # Continue with existing output file
                logging.info(f"Continuing with existing output file {output_file}")

        # Load existing IDs to avoid duplicates when resuming
        with output_file.open("r") as src:
            samples = list(map(json.loads, src))
            # Track both query_id and sub_id to avoid duplicates
            existing_ids = {(s["query_id"], s["sub_id"]) for s in samples}
            logging.info(f"Loaded {len(existing_ids)} existing sample IDs to avoid duplicates")
    else:
        existing_ids = set()

    return output_file, existing_ids


def initialize_model(cfg: AppConfig) -> tuple[SamplingParams, Any]:
    """Initialize the model and sampling parameters.

    Args:
        cfg: Application configuration

    Returns:
        Tuple containing:
        - Sampling parameters
        - Model generator for structured output
    """
    logging.debug("Initializing model")
    sampling_params = SamplingParams(**dataclasses.asdict(cfg.sampling_params))
    llm = LLM(**dataclasses.asdict(cfg.model))
    model = models.VLLM(llm)
    generator = generate.json(model, Response)

    return sampling_params, generator


def load_input_data(source_path: epath.Path) -> list[dict[str, Any]]:
    """Load input data from the source file.

    Args:
        source_path: Path to the input file

    Returns:
        List of samples to process
    """
    with source_path.open("r") as src:
        # Load the full JSON file
        data = json.loads(src.read())

        # Extract the results list from the JSON structure
        if "results" not in data:
            msg = "Input file does not contain 'results' key"
            raise ValueError(msg)

        raw_samples = data["results"]
        logging.info(f"Loaded {len(raw_samples)} samples from input file")

    return raw_samples


def process_batch(
    batch: list[dict[str, Any]],
    existing_ids: set[tuple[Any, str]],
    cfg: AppConfig,
    sampling_params: SamplingParams,
    generator: Any,
) -> tuple[list[dict[str, Any]], set[tuple[Any, str]]]:
    """Process a batch of samples.

    Args:
        batch: Batch of samples to process
        existing_ids: Set of existing IDs to avoid duplicates
        cfg: Application configuration
        sampling_params: Sampling parameters
        generator: Model generator

    Returns:
        Tuple containing:
        - Results from processing the batch
        - Updated set of existing IDs
    """
    # Expand samples - create one entry per generated answer
    expanded_samples = expand_samples(batch)

    # Filter out already processed samples
    expanded_samples = [s for s in expanded_samples if (s["query_id"], s["sub_id"]) not in existing_ids]
    if not expanded_samples:
        return [], existing_ids

    # Prepare prompts for batch processing
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
        logging.debug(f"Processing {len(expanded_samples)} samples")
        responses = generator(prompts, sampling_params=sampling_params)

        # Create results from responses
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

        return results, existing_ids

    except ValidationError:
        logging.exception("Error validating generated output")
        return [], existing_ids
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return [], existing_ids


def write_results(results: list[dict[str, Any]], dst_file) -> None:
    """Write batch results to the output file.

    Args:
        results: List of result dictionaries to write
        dst_file: Destination file object
    """
    if results:
        df = pl.DataFrame(results)
        df.write_ndjson(dst_file)


# === Main Application Function ===


def main(cfg: AppConfig):
    """Main function that runs the True/False judgment pipeline.

    This function:
    1. Initializes the model and output files
    2. Processes samples in batches
    3. Generates True/False judgments using the LLM
    4. Saves results to the output file

    Args:
        cfg: Application configuration
    """
    logging.info("\n%s", cfg)

    # Setup output file and load existing IDs
    output_file, existing_ids = setup_output_file(cfg)

    # Initialize model
    sampling_params, generator = initialize_model(cfg)

    # Load input data
    raw_samples = load_input_data(cfg.source)

    # Process samples in batches
    batches = list(tlz.partition_all(cfg.batch_size, raw_samples))

    with output_file.open("a") as dst:
        # Process batches with progress tracking
        for batch_idx, batch in enumerate(etqdm.tqdm(batches)):
            if cfg.limit is not None and batch_idx >= cfg.limit:
                break

            # Process the batch
            results, existing_ids = process_batch(batch, existing_ids, cfg, sampling_params, generator)

            # Write results to output file
            write_results(results, dst)

    logging.info("Completed processing. Results saved to %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
