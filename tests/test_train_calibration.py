"""Tests for train_calibration.py script."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from artefactual.calibration.train_calibration import train_calibration


def test_train_calibration_valid_data():
    """Test train_calibration with valid data."""
    # Create temporary CSV with valid data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        f.write("0.1,True\n")
        f.write("0.2,True\n")
        f.write("0.3,False\n")
        f.write("0.8,False\n")
        f.write("0.9,False\n")
        input_file = f.name

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should succeed
        train_calibration(input_file, output_file)

        # Verify output file exists and has the expected format
        output_path = Path(output_file)
        assert output_path.exists()

        with open(output_file) as f:
            weights = json.load(f)

        assert "intercept" in weights
        assert "coefficients" in weights
        assert "mean_entropy" in weights["coefficients"]
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_with_malformed_judgments():
    """Test train_calibration with mix of valid and malformed judgments."""
    # Create temporary CSV with mix of valid and malformed data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        f.write("0.1,True\n")
        f.write("0.2,True\n")
        f.write("0.3,False\n")
        f.write("0.4,False\n")
        f.write("0.9,maybe\n")  # malformed
        f.write("0.8,invalid\n")  # malformed
        f.write("0.7,\n")  # empty
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should succeed and drop malformed rows
        train_calibration(input_file, output_file)

        # Verify output file exists
        output_path = Path(output_file)
        assert output_path.exists()
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_all_malformed():
    """Test train_calibration with all malformed judgments."""
    # Create temporary CSV with all malformed data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        f.write("0.1,maybe\n")
        f.write("0.9,invalid\n")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should raise ValueError
        with pytest.raises(ValueError, match="No valid judgments found after parsing"):
            train_calibration(input_file, output_file)
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_missing_columns():
    """Test train_calibration with missing required columns."""
    # Create temporary CSV without required columns
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("score,result\n")
        f.write("0.1,True\n")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should raise ValueError
        with pytest.raises(ValueError, match="must contain 'uncertainty_score' and 'judgment' columns"):
            train_calibration(input_file, output_file)
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_single_class():
    """Test train_calibration with only one class."""
    # Create temporary CSV with only True judgments
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        f.write("0.1,True\n")
        f.write("0.2,True\n")
        f.write("0.3,True\n")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should raise ValueError
        with pytest.raises(ValueError, match="Need both positive .* and negative .* judgments"):
            train_calibration(input_file, output_file)
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_empty_file():
    """Test train_calibration with empty file."""
    # Create temporary empty CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should raise ValueError
        with pytest.raises(ValueError, match="No valid data found"):
            train_calibration(input_file, output_file)
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_train_calibration_case_insensitive():
    """Test train_calibration with different case variations."""
    # Create temporary CSV with different case variations
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("uncertainty_score,judgment\n")
        f.write("0.1,true\n")  # lowercase
        f.write("0.2,TRUE\n")  # uppercase
        f.write("0.3,False\n")  # normal case
        f.write("0.4,FALSE\n")  # uppercase
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        # Should succeed
        train_calibration(input_file, output_file)

        # Verify output file exists
        output_path = Path(output_file)
        assert output_path.exists()
    finally:
        # Cleanup
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
