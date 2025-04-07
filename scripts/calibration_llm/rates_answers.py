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
from beartype.door import die_if_unbearable
from etils import eapp, edc, epath, etqdm
from jinja2 import Environment, Template
from outlines import generate, models
from pydantic import BaseModel
from simple_parsing import field, subgroups
from simple_parsing.helpers import Serializable
from vllm import LLM, SamplingParams  # pytype: disable=import-error
from vllm.sampling_params import RequestOutputKind  # pytype: disable=import-error


@edc.dataclass
@dataclasses.dataclass
class SamplingConfig(Serializable):
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
class MistralSamplingConfig(SamplingConfig):
    """Configuration for Mistral models sampling parameters."""

    max_tokens: int = 8092
    temperature: float = 0.8
    top_p: float = 0.95


@edc.dataclass
@dataclasses.dataclass
class MistralSmallSamplingConfig(MistralSamplingConfig):
    """Configuration for Mistral Small model sampling parameters."""

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
class LlamaSamplingConfig(SamplingConfig):
    """Configuration for Llama model sampling parameters."""

    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1


@edc.dataclass
@dataclasses.dataclass
class ModelConfig(Serializable):
    """Configuration for a language model."""

    model: str | None = None
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
    dtype: str = "bfloat16"


@edc.dataclass
@dataclasses.dataclass
class MistralSmallModelConfig(MistralModelConfig):
    """
    Note: this model does not seem to support guided decoding
    """

    model: str | None = "mistralai/Mistral-Small-24B-Instruct-2501"


@edc.dataclass
@dataclasses.dataclass
class LlamaModelConfig(ModelConfig):
    """Base configuration for Llama models."""

    model: str | None = None
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.8
    trust_remote_code: bool = True
    max_model_len: int = 8192


@edc.dataclass
@dataclasses.dataclass
class Llama31ModelConfig(LlamaModelConfig):
    """Configuration for Llama 3.1 models."""

    model: str | None = "meta-llama/Llama-3.1-70B-Instruct"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9


@edc.dataclass
@dataclasses.dataclass
class Llama32ModelConfig(LlamaModelConfig):
    """Configuration for Llama 3.2 models."""

    model: str | None = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85


@edc.dataclass
@dataclasses.dataclass
class Llama33ModelConfig(LlamaModelConfig):
    """Configuration for Llama 3.3 models."""

    model: str | None = "nvidia/Llama-3.3-70B-Instruct-FP4"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.8


@edc.dataclass
@dataclasses.dataclass
class GemmaModelConfig(ModelConfig):
    """Base model configuration for Gemma models."""

    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    trust_remote_code: bool = True
    dtype: str = "bfloat16"


@edc.dataclass
@dataclasses.dataclass
class Gemma2ModelConfig(GemmaModelConfig):
    """Model configuration for Gemma 2 models."""

    model: str | None = "google/gemma-2-27b-it"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@edc.dataclass
@dataclasses.dataclass
class Gemma3ModelConfig(GemmaModelConfig):
    """Model configuration for Gemma 3 models."""

    model: str | None = "google/gemma-3-27b-it"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"


TRUE_FALSE_PROMPT = """
You are an expert evaluator tasked with determining if two answers convey compatible information. Your task is to make a binary True/False judgment on whether the answers are SEMANTICALLY COMPATIBLE.

Query:
{{ query }}

Expected Answer:
{{ expected_answer }}

Generated Answer:
{{ generated_answer }}

CRITICAL INSTRUCTIONS:
1. FIRST, perform a simple VERBATIM TEXT COMPARISON:
   - If the expected answer and generated answer are IDENTICAL (exact same text), your judgment MUST be TRUE
   - If not identical, proceed to semantic comparison

2. For SEMANTIC COMPARISON, use these MANDATORY RULES:
   - Judge "True" WHENEVER the general meaning or core concept is the same
   - Judge "True" if one answer is GENERAL and one is SPECIFIC about the same thing
   - Judge "True" if one answer names a CATEGORY (e.g., "missionaries") and the other provides SPECIFIC INSTANCES of that category (e.g., "Augustine was sent by Pope Gregory")
   - Judge "True" if one answer gives a BRIEF fact and the other ELABORATES with more details
   - Judge "True" if one answer is more detailed but does NOT contradict the other
   - Judge "False" ONLY if the answers directly CONTRADICT each other or discuss ENTIRELY different topics

3. EXTREMELY IMPORTANT RULES ABOUT SPECIFICITY:
   - When one answer is general and one is specific → TRUE
   - When one uses a category term and one gives examples → TRUE
   - When one gives "who/what" and the other adds "when/where/how/why" → TRUE
   - When one gives a person's role and the other gives their name → TRUE
   - When one refers to a group and the other names individuals → TRUE

4. Always check if the specific answer is an INSTANCE or EXAMPLE of the general answer
   - If it is, the judgment MUST be TRUE regardless of how detailed the specific answer is

5. The query is provided ONLY for context - do NOT use it in your judgment

FINAL CHECK BEFORE SUBMITTING:
- If one answer could reasonably be considered a more detailed version of the other → TRUE
- If after reading both answers, they feel like they're talking about the same basic concept → TRUE
- If you think "these answers are not contradicting each other" → TRUE

Your response MUST follow this format:
{
  "judgment": true/false,
  "explanation": "One clear sentence explaining why the answers are compatible or contradictory."
}
"""


@edc.dataclass
@dataclasses.dataclass
class PromptConfig(Serializable):
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
class AppConfig(Serializable):
    """Application configuration for the judgment pipeline.

    This class defines all the parameters needed to run the judgment pipeline,
    including input/output paths, model configuration, and processing parameters.
    """

    # Input/output configuration
    source: epath.Path  # Path to input data file
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")  # Directory for output files
    resume: bool = False  # Whether to resume from existing output
    config_file: epath.Path | None = None  # Path to a YAML configuration file to load

    # Model configuration
    model: ModelConfig = subgroups(
        {
            "default": ModelConfig,
            "mistral": MistralModelConfig,
            "mistral-small": MistralSmallModelConfig,
            "llama": LlamaModelConfig,
            "llama-3.1": Llama31ModelConfig,
            "llama-3.2": Llama32ModelConfig,
            "llama-3.3": Llama33ModelConfig,
            "gemma-2": Gemma2ModelConfig,
            "gemma-3": Gemma3ModelConfig,
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
            "mistral": MistralSamplingConfig,
            "mistral-small": MistralSmallSamplingConfig,
            "llama": LlamaSamplingConfig,
            "gemma": GemmaSamplingConfig,
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
    # Validate input types
    die_if_unbearable(model_config, ModelConfig)
    die_if_unbearable(sampling_config, SamplingConfig)

    logging.debug(f"Initializing model: {model_config.model}")
    logging.debug(f"Using sampling config: {type(sampling_config).__name__}")

    # Convert sampling config to VLLM SamplingParams
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
            "query": sample["query"],
            "expected_answer": sample["expected_answer"],
            "generated_answer": sample["generated_answer"],
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
    # Check if we should load the configuration from a file
    if cfg.config_file is not None:
        if not cfg.config_file.exists():
            msg = f"Config file not found: {cfg.config_file}"
            raise ValueError(msg)

        # Load the config file and replace our configuration
        logging.info("Loading configuration from %s", cfg.config_file)
        cfg = AppConfig.load(cfg.config_file)
        logging.info("Configuration loaded from file")

    # Validate model
    if cfg.model.model is None:
        msg = "Model path must be specified either via command line or in the config file"
        raise ValueError(msg)

    logging.info("\n%s", cfg)

    # Create output directory if it doesn't exist
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config to the output directory
    # Replace special characters in model name with underscores
    model_name = cfg.model.model.split("/")[-1].replace("/", "_").replace(".", "_")
    config_file = cfg.output_dir / f"{model_name}_config.yaml"
    cfg.save(config_file)
    logging.info(f"Saved configuration to {config_file}")

    # Step 1: Set up output file and get existing processed samples
    output_file, existing_ids = setup_output_file(
        output_dir=cfg.output_dir, model_name=cfg.model.model, source_name=cfg.source.name, resume=cfg.resume
    )

    # Step 2: Initialize model - sampling_config can be None for auto-selection
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
