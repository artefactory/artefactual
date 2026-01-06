"""Tests for artefactual.utils.io module."""

import json
import tempfile
from pathlib import Path

from artefactual.utils import save_to_json


def test_save_to_json_with_list():
    """Test save_to_json with a list of dicts."""
    data = [
        {"id": 1, "name": "Alice", "score": 0.95},
        {"id": 2, "name": "Bob", "score": 0.87},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_list.json"
        save_to_json(data, str(output_file))

        # Verify file was created and contains correct data
        assert output_file.exists()
        with Path(output_file).open(encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == data


def test_save_to_json_with_dict():
    """Test save_to_json with a single dict (dataset-level output)."""
    data = {
        "metadata": {
            "model": "test-model",
            "temperature": 0.6,
            "date": "2024-01-01",
        },
        "results": [
            {"query_id": "q1", "query": "What is AI?", "generated_answers": []},
            {"query_id": "q2", "query": "What is ML?", "generated_answers": []},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_dict.json"
        save_to_json(data, str(output_file))

        # Verify file was created and contains correct data
        assert output_file.exists()
        with Path(output_file).open(encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == data


def test_save_to_json_entropy_output_format():
    """Test save_to_json with the expected entropy dataset format."""
    # Simulate the structure created by generate_entropy_dataset
    entropy_data = {
        "metadata": {
            "generator_model": "mistralai/Ministral-8B-Instruct-2410",
            "retriever": "NONE",
            "date": "2024-01-01T00:00:00+00:00",
            "temperature": 0.6,
            "top_k_sampling": 50,
            "top_p": 1,
            "n_queries": 2,
            "iterations": 1,
            "number_logprobs": 15,
        },
        "results": [
            {
                "query_id": "q1",
                "query": "What is the capital of France?",
                "expected_answers": ["Paris"],
                "generated_answers": [{"0": "Paris", "epr_score": 0.95}],
            },
            {
                "query_id": "q2",
                "query": "What is 2+2?",
                "expected_answers": ["4", "four"],
                "generated_answers": [{"0": "4", "epr_score": 0.98}],
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "entropy_output.json"
        save_to_json(entropy_data, str(output_file))

        # Verify file was created and contains correct data
        assert output_file.exists()
        with Path(output_file).open(encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Verify structure
        assert "metadata" in loaded_data
        assert "results" in loaded_data
        assert loaded_data["metadata"]["generator_model"] == "mistralai/Ministral-8B-Instruct-2410"
        assert len(loaded_data["results"]) == 2
        assert loaded_data == entropy_data
