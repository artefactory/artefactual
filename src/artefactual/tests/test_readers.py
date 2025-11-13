"""Tests for the data.readers module."""

import json
import tempfile
from typing import Any

from absl.testing import absltest, parameterized
from etils import epath
from hypothesis import given
from hypothesis import strategies as st

from artefactual.data.readers import join_samples, read_file


class ReadersTest(parameterized.TestCase):
    """Test cases for the readers module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = epath.Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_file(self, samples: list[dict[str, Any]]) -> epath.Path:
        """Create a test file with the given samples."""
        path = self.temp_path / "test.jsonl"
        with path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        return path

    def test_read_file_empty(self):
        """Test reading an empty file."""
        path = self._create_test_file([])
        samples = read_file(path)
        self.assertEmpty(samples)

    def test_read_file_single(self):
        """Test reading a file with a single sample."""
        sample = {"id": "1", "question": "Why?", "answer": "Because"}
        path = self._create_test_file([sample])
        samples = read_file(path)
        self.assertLen(samples, 1)
        self.assertEqual(samples[0], sample)

    def test_read_file_multiple(self):
        """Test reading a file with multiple samples."""
        samples = [
            {"id": "1", "question": "Why?", "answer": "Because"},
            {"id": "2", "question": "How?", "answer": "Somehow"},
        ]
        path = self._create_test_file(samples)
        read_samples = read_file(path)
        self.assertLen(read_samples, 2)
        self.assertEqual(read_samples, samples)

    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        path = self.temp_path / "nonexistent.jsonl"
        with self.assertRaises(FileNotFoundError):
            read_file(path)

    def test_read_file_invalid_json(self):
        """Test reading a file with invalid JSON."""
        path = self.temp_path / "invalid.jsonl"
        with path.open("w") as f:
            f.write("not json\n")
        with self.assertRaises(json.JSONDecodeError):
            read_file(path)

    @given(st.lists(st.dictionaries(st.text(), st.text(), min_size=1).map(lambda d: {"id": "1", **d})))
    def test_read_file_property(self, samples):
        """Test reading a file with property-based testing."""
        path = self._create_test_file(samples)
        read_samples = read_file(path)
        self.assertEqual(read_samples, samples)

    def test_join_samples_empty(self):
        """Test joining empty samples."""
        result = join_samples([], [])
        self.assertEmpty(result)

    def test_join_samples_left_empty(self):
        """Test joining with empty left samples."""
        result = join_samples([], [{"id": "1"}])
        self.assertEmpty(result)

    def test_join_samples_right_empty(self):
        """Test joining with empty right samples."""
        result = join_samples([{"id": "1"}], [])
        self.assertEmpty(result)

    def test_join_samples_simple(self):
        """Test joining simple samples."""
        left = [{"id": "1", "rating": 5}]
        right = [{"id": "1", "response": "answer"}]
        result = join_samples(left, right)
        self.assertLen(result, 1)
        self.assertEqual(result[0]["id"], "1")
        self.assertEqual(result[0]["rating"], 5)
        self.assertEqual(result[0]["response"], "answer")

    def test_join_samples_multiple(self):
        """Test joining multiple samples."""
        left = [
            {"id": "1", "rating": 5},
            {"id": "2", "rating": 3},
            {"id": "3", "rating": 1},
        ]
        right = [
            {"id": "1", "response": "answer1"},
            {"id": "2", "response": "answer2"},
        ]
        result = join_samples(left, right)
        self.assertLen(result, 2)
        self.assertEqual(result[0]["id"], "1")
        self.assertEqual(result[0]["rating"], 5)
        self.assertEqual(result[0]["response"], "answer1")
        self.assertEqual(result[1]["id"], "2")
        self.assertEqual(result[1]["rating"], 3)
        self.assertEqual(result[1]["response"], "answer2")

    def test_join_samples_custom_key(self):
        """Test joining samples with a custom key function."""
        left = [
            {"id": "1", "key": "a", "rating": 5},
            {"id": "2", "key": "b", "rating": 3},
        ]
        right = [
            {"id": "3", "key": "a", "response": "answer1"},
            {"id": "4", "key": "b", "response": "answer2"},
        ]
        result = join_samples(left, right, key_fn=lambda x: x["key"])
        self.assertLen(result, 2)
        self.assertEqual(result[0]["key"], "a")
        self.assertEqual(result[0]["rating"], 5)
        self.assertEqual(result[0]["response"], "answer1")
        self.assertEqual(result[1]["key"], "b")
        self.assertEqual(result[1]["rating"], 3)
        self.assertEqual(result[1]["response"], "answer2")

    def test_join_samples_key_error(self):
        """Test joining samples with a key that doesn't exist."""
        left = [{"id": "1", "rating": 5}]
        right = [{"id": "1", "response": "answer"}]
        with self.assertRaises(KeyError):
            join_samples(left, right, key_fn=lambda x: x["nonexistent"])

    @given(
        st.lists(st.dictionaries(st.text(), st.text(), min_size=1).map(lambda d: {"id": "1", **d})),
        st.lists(st.dictionaries(st.text(), st.text(), min_size=1).map(lambda d: {"id": "1", **d})),
    )
    def test_join_samples_property(self, left, right):
        """Test joining samples with property-based testing."""
        if not left or not right:
            result = join_samples(left, right)
            self.assertEmpty(result)
        else:
            left[0]["id"] = "1"
            right[0]["id"] = "1"
            result = join_samples(left, right)
            if result:
                self.assertEqual(result[0]["id"], "1")


if __name__ == "__main__":
    absltest.main()
