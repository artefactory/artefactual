"""Dataset configuration classes."""

import abc
import dataclasses
from typing import Any

from etils import edc
from simple_parsing import Serializable


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
