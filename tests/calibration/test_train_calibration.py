import tempfile
from pathlib import Path

import pandas as pd
import pytest

from artefactual.calibration import train_calibration


@pytest.mark.parametrize(
    ("scores", "judgments"),
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["true", "true", "false", "false", "true", "false"]),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["True", "True", "False", "False", "True", "False"]),
        ([0.1, 0.2, 0.3, 0.4], ["TRUE", "false", "True", "FALSE"]),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["true", None, "false", "false", "true", None]),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["true", "invalid", "false", "maybe", "true", "false"]),
    ],
)
def test_train_calibration_success_cases(scores, judgments):
    """Test successful calibration training across supported judgment variations."""
    df = pd.DataFrame({"uncertainty_score": scores, "judgment": judgments})

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)
        train_calibration(input_file, output_file)

        assert output_file.exists()


def test_train_calibration_insufficient_classes():
    """Test error when only one class is present."""
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3],
        "judgment": ["true", "true", "true"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)

        with pytest.raises(ValueError, match=r"Need both positive.*and negative.*judgments"):
            train_calibration(input_file, output_file)


def test_train_calibration_all_none():
    """Test error when all judgments are None."""
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3],
        "judgment": [None, None, None],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)

        with pytest.raises(ValueError, match="No valid data found"):
            train_calibration(input_file, output_file)
