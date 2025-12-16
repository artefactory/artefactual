import tempfile
from pathlib import Path

import pandas as pd
import pytest

from artefactual.calibration.train_calibration import train_calibration


def test_train_calibration_basic():
    """Test basic calibration training with valid data."""
    # Create sample data
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "judgment": ["true", "true", "false", "false", "true", "false"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)
        train_calibration(input_file, output_file)

        # Check output exists
        assert output_file.exists()


def test_train_calibration_numeric_labels():
    """Test that target labels are properly converted to numeric int type."""
    # Create sample data with string judgments
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "judgment": ["True", "True", "False", "False", "True", "False"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)
        train_calibration(input_file, output_file)

        # Verify model was trained successfully (no type errors)
        assert output_file.exists()


def test_train_calibration_mixed_case_judgments():
    """Test handling of mixed case judgment values."""
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3, 0.4],
        "judgment": ["TRUE", "false", "True", "FALSE"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)
        train_calibration(input_file, output_file)

        assert output_file.exists()


def test_train_calibration_with_none_judgments():
    """Test that None judgments are properly filtered out."""
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "judgment": ["true", None, "false", "false", "true", None],
    }
    df = pd.DataFrame(data)

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


def test_train_calibration_invalid_judgments():
    """Test that invalid judgment values are filtered out."""
    data = {
        "uncertainty_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "judgment": ["true", "invalid", "false", "maybe", "true", "false"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.csv"
        output_file = Path(tmpdir) / "output.json"

        df.to_csv(input_file, index=False)
        train_calibration(input_file, output_file)

        # Should succeed with only valid judgments
        assert output_file.exists()
