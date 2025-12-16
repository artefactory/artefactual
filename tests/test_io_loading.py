"""Tests for asset loading from the new location."""

import json
from pathlib import Path

import pytest

from artefactual.utils.io import load_calibration, load_weights


def test_load_weights_from_built_in_model():
    """Test loading weights using a built-in model name."""
    weights = load_weights("mistralai/Ministral-8B-Instruct-2410")
    
    assert isinstance(weights, dict)
    assert "intercept" in weights
    assert "coefficients" in weights
    assert isinstance(weights["coefficients"], dict)


def test_load_calibration_from_built_in_model():
    """Test loading calibration using a built-in model name."""
    calibration = load_calibration("mistralai/Ministral-8B-Instruct-2410")
    
    assert isinstance(calibration, dict)
    # Calibration files may have different structure, just verify it's a dict
    assert len(calibration) > 0


def test_load_weights_all_models():
    """Test that all registered models have weights files."""
    from artefactual.utils.io import MODEL_WEIGHT_MAP
    
    for model_name in MODEL_WEIGHT_MAP.keys():
        weights = load_weights(model_name)
        assert isinstance(weights, dict)


def test_load_calibration_all_models():
    """Test that all registered models have calibration files."""
    from artefactual.utils.io import MODEL_CALIBRATION_MAP
    
    for model_name in MODEL_CALIBRATION_MAP.keys():
        calibration = load_calibration(model_name)
        assert isinstance(calibration, dict)


def test_load_weights_from_custom_path(tmp_path):
    """Test loading weights from a custom file path."""
    # Create a temporary weights file
    custom_weights = {
        "intercept": 0.5,
        "coefficients": {"feature1": 1.0, "feature2": 2.0}
    }
    weights_file = tmp_path / "custom_weights.json"
    with open(weights_file, "w") as f:
        json.dump(custom_weights, f)
    
    # Load from custom path
    loaded_weights = load_weights(str(weights_file))
    assert loaded_weights == custom_weights


def test_load_calibration_from_custom_path(tmp_path):
    """Test loading calibration from a custom file path."""
    # Create a temporary calibration file
    custom_calibration = {
        "intercept": -1.5,
        "coefficients": {"coef1": 0.8, "coef2": 1.2}
    }
    calibration_file = tmp_path / "custom_calibration.json"
    with open(calibration_file, "w") as f:
        json.dump(custom_calibration, f)
    
    # Load from custom path
    loaded_calibration = load_calibration(str(calibration_file))
    assert loaded_calibration == custom_calibration


def test_load_weights_invalid_model():
    """Test that loading weights with invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Could not find weights"):
        load_weights("invalid/model-name")


def test_load_calibration_invalid_model():
    """Test that loading calibration with invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Could not find calibration"):
        load_calibration("invalid/model-name")


def test_load_weights_invalid_json(tmp_path):
    """Test that loading weights from invalid JSON raises ValueError."""
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("not valid json{")
    
    with pytest.raises(ValueError, match="not valid JSON"):
        load_weights(str(invalid_file))


def test_assets_directory_exists():
    """Test that the assets/data directory exists and contains the expected files."""
    from artefactual.utils.io import _get_assets_dir
    
    assets_dir = _get_assets_dir()
    assert assets_dir.exists(), f"Assets directory not found: {assets_dir}"
    assert assets_dir.is_dir()
    
    # Check that at least some JSON files exist
    json_files = list(assets_dir.glob("*.json"))
    assert len(json_files) > 0, "No JSON files found in assets directory"
