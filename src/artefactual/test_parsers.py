"""Tests for the parsing layer."""

from absl.testing import absltest, parameterized

from artefactual.parsers import (
    OpenAIParsedCompletionV1,
    parsed_token_logprobs,
    parsed_top_logprobs_tensor,
)


class ParseCompletionTest(parameterized.TestCase):
    """Unit tests for OpenAIParsedCompletionV1 parsing."""

    def test_openai_new_schema(self):
        response = {
            "id": "cmpl-123",
            "model": "gpt-test",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello",
                                "logprobs": {
                                    "tokens": ["Hello", "!"],
                                    "token_logprobs": [-0.1, -0.2],
                                    "top_logprobs": [
                                        {"Hello": -0.1, "Hi": -0.5},
                                        {"!": -0.2, ".": -1.0},
                                    ],
                                },
                            },
                        ],
                    },
                }
            ],
        }

        parsed = OpenAIParsedCompletionV1.from_response(response)

        self.assertIsInstance(parsed, OpenAIParsedCompletionV1)
        self.assertEqual(parsed.text, "Hello")
        self.assertEqual([ti.token for ti in parsed.token_infos], ["Hello", "!"])
        self.assertEqual([ti.logprob for ti in parsed.token_infos], [-0.1, -0.2])
        self.assertLen(parsed.token_infos, 2)
        self.assertIsNotNone(parsed.metadata)
        metadata = parsed.metadata or {}
        self.assertEqual(metadata.get("finish_reason"), "stop")
        self.assertEqual(metadata.get("model"), "gpt-test")

    def test_legacy_logprobs(self):
        response = {
            "choices": [
                {
                    "text": "Hi!",
                    "logprobs": {
                        "tokens": ["Hi", "!"],
                        "token_logprobs": [-0.4, -0.6, -9.9],  # extra logprob should be truncated
                        "top_logprobs": [
                            {"Hi": -0.4},
                            {"!": -0.6},
                            {"?": -1.0},
                        ],
                    },
                }
            ]
        }

        parsed = OpenAIParsedCompletionV1.from_response(response)

        self.assertEqual(parsed.text, "Hi!")
        self.assertEqual([ti.token for ti in parsed.token_infos], ["Hi", "!"])
        self.assertEqual([ti.logprob for ti in parsed.token_infos], [-0.4, -0.6])
        self.assertLen(parsed.token_infos, 2)

    def test_missing_logprobs(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "No logprobs"}}]}
        parsed = OpenAIParsedCompletionV1.from_response(response)

        self.assertEqual(parsed.text, "No logprobs")
        self.assertEqual(parsed.token_infos, [])

    def test_parse_completions_batch(self):
        responses = [
            {"choices": [{"text": "First", "logprobs": {"tokens": ["First"], "token_logprobs": [-0.3]}}]},
            {"choices": [{"text": "Second"}]},
        ]
        parsed_batch = [OpenAIParsedCompletionV1.from_response(resp) for resp in responses]

        self.assertLen(parsed_batch, 2)
        self.assertEqual(parsed_batch[0].text, "First")
        self.assertEqual(parsed_batch[1].text, "Second")
        self.assertEqual([ti.token for ti in parsed_batch[0].token_infos], ["First"])
        self.assertEqual(parsed_batch[1].token_infos, [])

    def test_parsed_token_logprobs(self):
        responses = [
            {"choices": [{"text": "One", "logprobs": {"tokens": ["One"], "token_logprobs": [-0.5]}}]},
            {"choices": [{"text": "Two", "logprobs": {"tokens": ["Two"], "token_logprobs": [-1.0]}}]},
        ]
        parsed_batch = [OpenAIParsedCompletionV1.from_response(resp) for resp in responses]
        arr = parsed_token_logprobs(parsed_batch)  # type: ignore[arg-type]
        self.assertEqual(arr.shape, (2,))
        self.assertAlmostEqual(arr[0], -0.5)
        self.assertAlmostEqual(arr[1], -1.0)

    def test_top_logprobs_padding_and_truncation(self):
        responses = [
            {
                "choices": [
                    {
                        "text": "Hi",
                        "logprobs": {
                            "tokens": ["Hi"],
                            "token_logprobs": [-0.4],
                            "top_logprobs": [
                                {"Hi": -0.4, "Hello": -0.8},  # only 2 entries
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "text": "Yo",
                        "logprobs": {
                            "tokens": ["Yo"],
                            "token_logprobs": [-0.2],
                            "top_logprobs": [
                                {"Yo": -0.2, "Y": -0.3, "Yup": -0.5, "Yolo": -1.0},  # 4 entries
                            ],
                        },
                    }
                ]
            },
        ]
        parsed_batch = [OpenAIParsedCompletionV1.from_response(resp) for resp in responses]
        tensor = parsed_top_logprobs_tensor(parsed_batch, top_k=3, padding_value=0.0)  # type: ignore[arg-type]
        self.assertEqual(tensor.shape, (2, 1, 3))
        # First sequence: two entries padded to 3
        self.assertAlmostEqual(tensor[0, 0, 0], -0.4)
        self.assertAlmostEqual(tensor[0, 0, 1], -0.8)
        self.assertAlmostEqual(tensor[0, 0, 2], 0.0)
        # Second sequence: truncated to top 3 by logprob value order
        self.assertTrue(tensor[1, 0, 0] >= tensor[1, 0, 1] >= tensor[1, 0, 2])

    def test_top_logprobs_fallback_to_single_logprob(self):
        responses = [
            {"choices": [{"text": "OnlyOne", "logprobs": {"tokens": ["OnlyOne"], "token_logprobs": [-0.7]}}]},
        ]
        parsed_batch = [OpenAIParsedCompletionV1.from_response(resp) for resp in responses]
        tensor = parsed_top_logprobs_tensor(parsed_batch, top_k=2, padding_value=0.0)  # type: ignore[arg-type]
        self.assertEqual(tensor.shape, (1, 1, 2))
        self.assertAlmostEqual(tensor[0, 0, 0], -0.7)
        self.assertAlmostEqual(tensor[0, 0, 1], 0.0)


if __name__ == "__main__":
    absltest.main()
