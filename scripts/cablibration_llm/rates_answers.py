# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "absl-py",
#     "etils[eapp,edc,epath,etqdm]",
#     "beartype",
#     "jinja2",
#     "toolz",
#     "huggingface-hub",
#     "safetensors",
#     "vllm>=0.7",
#     "polars",
#     "orjson",
#     "outlines[vllm]",
#     "pydantic",
#     "triton==3.1",
#     "torch==2.5.1",
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
from pydantic import BaseModel
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
class MistralSmallSamplingConfig(SamplingConfig):
    max_tokens: int = 8092


@edc.dataclass
@dataclasses.dataclass
class GemmaSamplingConfig(SamplingConfig):
    """Configuration for Gemma model sampling parameters."""

    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


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

    @classmethod
    def make_generator(cls, model: Any) -> Any:
        """Create a generator that produces structured output for this model.

        Args:
            model: The model to use for generation

        Returns:
            A generator that produces structured output
        """
        return generate.json(model, Response)


@edc.dataclass
@dataclasses.dataclass
class MistralModelConfig(ModelConfig):
    """Model configuration for Mistral models."""

    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.65


@edc.dataclass
@dataclasses.dataclass
class MistralSmallModelConfig(MistralModelConfig):
    """
    Note: this model does not seem to support guided decoding
    """

    model: str = "mistralai/Mistral-Small-24B-Instruct-2501"


@edc.dataclass
@dataclasses.dataclass
class LLamaModelConfig(ModelConfig):
    """Model configuration for Llama models."""

    model: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
    tensor_parallel_size: int = 2


@edc.dataclass
@dataclasses.dataclass
class GemmaModelConfig(ModelConfig):
    """Model configuration for Gemma models."""

    model: str = "neuralmagic/gemma-2-9b-it-FP8"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    trust_remote_code: bool = True
    dtype: str = "auto"


TRUE_FALSE_PROMPT = """
You are an expert evaluator tasked with determining if a provided answer correctly responds to a query based on an expected answer. Your task is to make a binary True/False judgment.

Query:
{{ query }}

Expected Answer:
{{ expected_answer }}

Provided Answer:
{{ generated_answer }}

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

    def render(self, query: str, expected_answer: str, generated_answer: str) -> str:
        """Render the prompt template with the given parameters.

        Args:
            query: The question or query to evaluate
            expected_answer: The reference/correct answer
            generated_answer: The generated answer to evaluate

        Returns:
            Formatted prompt string
        """
        return self.template.render(query=query, expected_answer=expected_answer, generated_answer=generated_answer)


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
            "mistral-small": MistralSmallModelConfig,
            "llama-3.1-70b": LLamaModelConfig,
            "gemma-2-9b": GemmaModelConfig,
        },
        default="default",
    )

    # Prompt and sampling configuration
    prompt: PromptConfig = subgroups(
        {
            "default": PromptConfig,
        },
        default="default",
    )
    sampling_params: SamplingConfig = subgroups(
        {
            "default": SamplingConfig,
            "mistral-small": MistralSmallSamplingConfig,
            "gemma-2-9b": GemmaSamplingConfig,
        },
        default="default",
    )

    # Processing parameters
    limit: int | None = None  # Maximum number of batches to process
    batch_size: int = 4  # Number of samples to process in each batch


class Response(BaseModel):
    """Model for the LLM's judgment response."""

    judgment: bool  # True or False judgment
    explanation: str  # Explanation of the judgment


# === Utility Functions ===


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
        # Normalize expected_answer to string
        expected_answer = sample["expected_answer"]
        if isinstance(expected_answer, list):
            expected_answer = " ".join(expected_answer)

        # Extract common fields once
        base_sample = {
            "query_id": sample.get("query_id", -1),
            "query": sample["query"],
            "expected_answer": expected_answer,
        }

        # Flatten the generated answers
        for gen_answer_dict in sample["generated_answers"]:
            gen_key = next(iter(gen_answer_dict.keys()))
            expanded_samples.append({
                **base_sample,
                "sub_id": gen_key,
                "generated_answer": gen_answer_dict[gen_key],
            })

    return expanded_samples


def setup_output_file(
    output_dir: epath.Path, model_name: str, source_name: str, *, resume: bool = False
) -> tuple[epath.Path, set[tuple[Any, str]]]:
    """Set up the output file and load existing IDs if resuming.

    Args:
        output_dir: Directory to store output files
        model_name: Name of the model
        source_name: Name of the source file
        resume: Whether to resume from existing output

    Returns:
        Tuple containing:
        - Path to the output file
        - Set of existing IDs to avoid duplicates
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file path - use "judgements" in the filename
    output_file = output_dir / f"judgements_{model_name}_{source_name}".replace("/", "_")
    logging.debug("Output file: %s", output_file)

    # Check if output file exists
    if output_file.exists():
        if not resume:
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


def load_input_data(source_path: epath.Path) -> list[dict[str, Any]]:
    """Load input data from the source file.

    Args:
        source_path: Path to the input file

    Returns:
        List of samples to process
    """
    with source_path.open("r") as src:
        # Load the JSON file and extract results
        data = json.loads(src.read())

        # Extract the results list or raise a descriptive error
        if not (raw_samples := data.get("results")):
            available_keys = list(data.keys())
            msg = f"Input file does not contain 'results' key: {available_keys}"
            raise ValueError(msg)

        logging.info(f"Loaded {len(raw_samples)} samples from input file")
        return raw_samples


def initialize_model(model_config: ModelConfig, sampling_config: SamplingConfig) -> tuple[SamplingParams, Any]:
    """Initialize the model and sampling parameters.

    Args:
        model_config: Configuration for the model
        sampling_config: Configuration for sampling parameters

    Returns:
        Tuple containing:
        - Sampling parameters
        - Model generator for structured output
    """
    logging.debug(f"Initializing model: {model_config.model}")
    sampling_params = SamplingParams(**dataclasses.asdict(sampling_config))

    # Disable progress bar for model initialization and inference
    llm = LLM(**dataclasses.asdict(model_config), distributed_executor_backend="mp")
    vllm_model = models.VLLM(llm)

    # Use the model-specific generator creation method with its class
    model_class = type(model_config)
    generator = model_class.make_generator(vllm_model)

    return sampling_params, generator


def write_results(results: list[dict[str, Any]], dst_file) -> None:
    """Write batch results to the output file.

    Args:
        results: List of result dictionaries to write
        dst_file: Destination file object
    """
    if results:
        df = pl.DataFrame(results)
        df.write_ndjson(dst_file)
        dst_file.flush()  # Ensure data is flushed to disk after each batch


def filter_processed_samples(
    expanded_samples: list[dict[str, Any]], processed_ids: set[tuple[Any, str]]
) -> list[dict[str, Any]]:
    """Filter out samples that have already been processed.

    Args:
        expanded_samples: Expanded samples to filter
        processed_ids: Set of already processed IDs

    Returns:
        Filtered list of samples
    """
    # Keep samples with ID -1 (default) and filter out others that are already processed
    filtered_samples = [
        s for s in expanded_samples if s["query_id"] == -1 or (s["query_id"], s["sub_id"]) not in processed_ids
    ]
    return filtered_samples


def create_result_from_response(sample: dict[str, Any], response: Any) -> dict[str, Any] | None:
    """Create a result dictionary from a sample and its corresponding response.

    Args:
        sample: The sample that was processed
        response: The model's response

    Returns:
        Result dictionary or None if there was an error
    """
    try:
        return {
            "query_id": sample["query_id"],
            "sub_id": sample["sub_id"],
            **response.model_dump(mode="json"),
        }
    except Exception as e:
        logging.error(f"Error creating result for sample {sample['query_id']}-{sample['sub_id']}: {e}")
        return None


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

    # Step 1: Set up output file and get existing processed samples
    output_file, existing_ids = setup_output_file(
        output_dir=cfg.output_dir, model_name=cfg.model.model, source_name=cfg.source.name, resume=cfg.resume
    )

    # Step 2: Initialize model
    sampling_params, generator = initialize_model(model_config=cfg.model, sampling_config=cfg.sampling_params)

    # Step 3: Load input data
    raw_samples = load_input_data(cfg.source)

    # Step 4: Divide samples into batches
    batches = list(tlz.partition_all(cfg.batch_size, raw_samples))

    # Step 5: Process each batch and write results
    processed_ids = set(existing_ids)  # Create a copy for tracking

    with output_file.open("a") as dst:
        # Process batches with progress tracking
        for batch_idx, batch in enumerate(etqdm.tqdm(batches, desc="Processing batches")):
            # Check if we've reached the batch limit
            if cfg.limit is not None and batch_idx >= cfg.limit:
                logging.info(f"Reached limit of {cfg.limit} batches, stopping")
                break

            try:
                # Step 5.1: Expand samples (one entry per generated answer)
                expanded_samples = expand_samples(batch)
                logging.debug(f"Batch {batch_idx}: Expanded to {len(expanded_samples)} individual samples")

                # Step 5.2: Filter out already processed samples (keep samples with ID -1)
                to_process = filter_processed_samples(expanded_samples, processed_ids)
                logging.debug(f"Batch {batch_idx}: {len(to_process)} samples to process after filtering")

                if not to_process:
                    logging.warning(f"Batch {batch_idx}: All samples already processed, skipping")
                    continue

                # Step 5.3: Prepare prompts for generation
                prompts = [
                    cfg.prompt.render(
                        query=sample["query"],
                        expected_answer=sample["expected_answer"],
                        generated_answer=sample["generated_answer"],
                    )
                    for sample in to_process
                ]

                # Step 5.4: Get model judgments
                logging.debug(f"Processing {len(to_process)} samples")
                responses = generator(prompts, sampling_params=sampling_params, use_tqdm=False)

                # Step 5.5: Process responses into results
                results = (
                    create_result_from_response(sample, response)
                    for sample, response in zip(to_process, responses, strict=True)
                )
                results = [res for res in results if res is not None]

                logging.debug(f"Batch {batch_idx}: Generated {len(results)} results")

                # Step 5.6: Write results to output file
                if results:
                    write_results(results, dst)
                    logging.debug(f"Batch {batch_idx}: Wrote {len(results)} results to output file")

                # Step 5.7: Update processed IDs set (except for default IDs)
                processed_ids |= {
                    (sample["query_id"], sample["sub_id"]) for sample in to_process if sample["query_id"] != -1
                }

            except (ValueError, RuntimeError, json.JSONDecodeError) as e:
                # Log error but continue processing remaining batches
                logging.error(f"Error processing batch {batch_idx}: {e}")
                logging.debug("Error details", exc_info=True)

    logging.info("Completed processing. Results saved to %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
