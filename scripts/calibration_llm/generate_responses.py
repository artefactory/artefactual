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
#     "safetensors",
#     "numpy",
#     "vllm",
#     "datasets>=3.2.0",
#     "polars",
# ]
# ///


import abc
import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from itertools import chain
from typing import Annotated, Any

import numpy as np
import polars as pl
import toolz as tlz
from absl import app, logging
from beartype import beartype
from beartype.vale import Is
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from etils import eapp, edc, epath, etqdm
from simple_parsing import Serializable, field, subgroups
from typing_extensions import override
from vllm import LLM, SamplingParams  # pytype: disable=import-error
from vllm.sampling_params import RequestOutputKind  # pytype: disable=import-error

# Import from artefactual library
from artefactual.scoring import extract_logprobs

HFDataset = DatasetDict | Dataset | IterableDataset | IterableDatasetDict

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


@dataclasses.dataclass
class SplitDatasetConfig(abc.ABC, Serializable):
    """Configuration for a dataset split."""

    path: str
    split: str
    batch_size: int = 1


@edc.dataclass
@dataclasses.dataclass
class TrainSplitDatasetConfig(SplitDatasetConfig):
    """Configuration for a training dataset split."""

    split: str = "train"


@edc.dataclass
@dataclasses.dataclass
class TestSplitDatasetConfig(SplitDatasetConfig):
    """Configuration for a test dataset split."""

    split: str = "test"


@edc.dataclass
@dataclasses.dataclass
class ValSplitDatasetConfig(SplitDatasetConfig):
    """Configuration for a validation dataset split."""

    split: str = "val"


@dataclasses.dataclass
class DatasetConfig(abc.ABC, Serializable):
    """Base configuration for a dataset."""

    name: str
    train: TrainSplitDatasetConfig
    val: ValSplitDatasetConfig | None = None
    test: TestSplitDatasetConfig | None = None
    num_proc: int | None = None

    @abc.abstractmethod
    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert a dataset sample to a standardized format."""
        pass


@edc.dataclass
@dataclasses.dataclass
class MrQaDatasetConfig(DatasetConfig):
    name: str = "mrqua"
    train: TrainSplitDatasetConfig = field(default_factory=lambda: TrainSplitDatasetConfig(path="mrqa"))

    @override
    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Implementation for MRQA dataset processing
        return {"question": None, "answer": None}


@edc.dataclass
@dataclasses.dataclass
class SquadDatasetConfig(DatasetConfig):
    name: str = "squad"
    train: TrainSplitDatasetConfig = field(default_factory=lambda: TrainSplitDatasetConfig(path="squad"))

    @override
    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Implementation for SQuAD dataset processing
        return {"question": None, "answer": None}


@edc.dataclass
@dataclasses.dataclass
class NaturalQADatasetConfig(DatasetConfig):
    name: str = "naturalqa"
    train: TrainSplitDatasetConfig = field(
        default_factory=lambda: TrainSplitDatasetConfig(path="google-research-datasets/natural_questions")
    )

    @override
    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        question = tlz.get_in(["question", "text"], sample, no_default=True)
        short_answers = tlz.get_in(["annotations", "short_answers"], sample, no_default=True)
        texts = tlz.pluck("text", short_answers)
        all_texts = chain.from_iterable(texts)
        answer = " ".join(all_texts)
        if not answer:
            return {"question": None, "answer": None}
        return {"question": question, "answer": answer}


@edc.dataclass
@dataclasses.dataclass
class Gemma2SamplingConfig(SamplingConfig):
    temperature: float = 0.6
    top_p: float = 0.9
    add_generation_prompt: bool = True


@edc.dataclass
@dataclasses.dataclass
class Gemma2ModelConfig(ModelConfig):
    model: str = "neuralmagic/gemma-2-9b-it-FP8"


@edc.dataclass
@dataclasses.dataclass
class MistralModelConfig(ModelConfig):
    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    # TODO: Could be an experiments config
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")
    model: ModelConfig = subgroups(
        {"default": ModelConfig, "mistral": MistralModelConfig, "gemma2": Gemma2ModelConfig}, default="default"
    )
    dataset: DatasetConfig = subgroups(
        {"mrqa": MrQaDatasetConfig, "squad": SquadDatasetConfig, "naturalqa": NaturalQADatasetConfig}, default="squad"
    )
    sampling_params: SamplingConfig = subgroups(
        {"default": SamplingConfig, "gemma": Gemma2SamplingConfig}, default="default"
    )
    repeat: MultipleGenerationConfig = field(default_factory=MultipleGenerationConfig)

    def __post_init__(self):
        if self.repeat.n_samples == 1:
            self.repeat = dataclasses.replace(
                self.repeat, min_val=self.sampling_params.temperature, max_val=self.sampling_params.temperature
            )


@beartype
def make_dataset(
    config: SplitDatasetConfig, sample_fn: Callable[[dict[str, Any]], dict[str, Any]], num_proc: int | None
) -> HFDataset:
    ds = load_dataset(config.path, split=config.split)
    ds = ds.map(sample_fn, num_proc=num_proc)
    ds = ds.filter(lambda x: x["answer"] is not None, num_proc=num_proc)
    if config.batch_size > 1:
        ds = ds.batch(config.batch_size, num_proc=num_proc)
    return ds


def sampling_params_temperature(config: MultipleGenerationConfig, sp: SamplingParams):
    temperature = 10 ** np.random.uniform(np.log10(config.min_val), np.log10(config.max_val))
    new_sp = sp.clone()
    new_sp.temperature = temperature
    return new_sp


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)
    output_dir = cfg.output_dir / f"{cfg.model.model}_{cfg.dataset.name}".replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.debug("Make dataset")
    ds = make_dataset(cfg.dataset.train, cfg.dataset.sample_fn, num_proc=cfg.dataset.num_proc)
    logging.debug("Make llm")
    sampling_params = SamplingParams.from_optional(**dataclasses.asdict(cfg.sampling_params))

    # Write n_samples files
    # Draw temperature randomly at each batch but be careful to write each document only once

    sps = [sampling_params_temperature(cfg.repeat, sampling_params) for _ in range(cfg.repeat.n_samples)]
    output_files = [output_dir / f"responses_temperature_{sp.temperature}.json" for sp in sps]

    llm = LLM(**dataclasses.asdict(cfg.model))
    total = cfg.repeat.n_samples * len(ds)
    with ExitStack() as stack:
        opened_files = [stack.enter_context(path.open("a")) for path in output_files]
        with etqdm.tqdm(total=total, desc="batch") as pbar:
            for batch in ds:
                for opened_file, sp in zip(opened_files, sps, strict=True):
                    requests = llm.generate(batch["question"], sampling_params=sp, use_tqdm=False)
                    responses, logprobs = zip(
                        *((req.outputs[0].text, req.outputs[0].logprobs) for req in requests), strict=True
                    )
                    tokens, processed = zip(*map(extract_logprobs, logprobs), strict=True)
                    sample = {
                        "id": batch["id"],
                        "response": responses,
                        "question": batch["question"],
                        "answer": batch["answer"],
                        "tokens": tokens,
                        "logprobs": processed,
                    }
                    df = pl.DataFrame(sample)
                    df.write_ndjson(opened_file)
                    pbar.update(len(df))


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
