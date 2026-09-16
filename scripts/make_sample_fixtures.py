"""Write the two sample batch files the offline notebooks train on.

`docs/examples/questions_sample.json` in, `responses_sample.jsonl` and
`judgments_sample.jsonl` out, in the OpenAI Batch output shape every reader in this
repository already understands.

The three files are joined on `custom_id`, so regenerating the question pack without
regenerating these leaves the notebooks silently training on whatever half still matches.
They were hand-written before, which is how that drift became possible; this script exists
so the pair can be rebuilt from the pack in one command:

    uv run python scripts/make_sample_fixtures.py

The responses are synthetic and say so -- `"model": "synthetic-stand-in"`. Nothing here was
sampled from a language model, and the log-probabilities are drawn to carry the property the
notebooks demonstrate rather than measured: a hallucinated answer is generated from a
flatter distribution than a correct one, so the entropy features separate the two classes
and the fit has something to find. Real numbers for a real model are what
`train_wepr_pipeline` produces; these exist so the other two notebooks run with no API key.

The per-source accuracies are the point of the mixed pack and are reproduced here: a
current model answers most of TriviaQA correctly and most of SimpleQA wrongly. Sampling
both halves at the same rate would hide exactly the effect the mix was introduced to
expose.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
QUESTIONS = EXAMPLES / "questions_sample.json"
RESPONSES = EXAMPLES / "responses_sample.jsonl"
JUDGMENTS = EXAMPLES / "judgments_sample.jsonl"

# Every published detector is fitted at 15, and a response carrying fewer is refused rather
# than zero-filled, so a fixture narrower than this fails inside the notebook's own `fit`.
K = 15

SEED = 20260916

# Share of answers that come back correct, per source. Not a guess about any particular
# model -- the shape of the published SimpleQA results, which is why the pack mixes the two.
ACCURACY = {"triviaqa": 0.70, "simpleqa": 0.24}

# Two requests fail outright. `read_batch` emits one row per line and refuses to skip a
# failed one, so the notebooks have to show a reader what that looks like; with none in the
# file, the branch that reports them is never exercised.
FAILURES = 2


def source_of(question_id: str) -> str:
    """Which dataset a row came from. SimpleQA ids are the only ones with a stable prefix.

    TriviaQA's own ids carry a dozen different ones -- `tc_`, `sfq_`, `bb_`, `odql_` -- so
    they are identified by not being SimpleQA's.
    """
    return "simpleqa" if question_id.startswith("sq-") else "triviaqa"


def tokens_of(answer: str) -> list[str]:
    """The answer as the two to four pieces a tokenizer would plausibly cut it into.

    Word boundaries rather than a real tokenizer: the point of the fixture is the *shape* of
    the log-probability array, and no reader learns anything from byte-pair splits of
    "Radcliffe College".
    """
    pieces = answer.split()[:4]
    return pieces or [answer[:12] or "?"]


def logprobs_for(token: str, uncertain: bool, rng: random.Random) -> dict:
    """One token's chosen log-probability and its `K` candidates, most likely first.

    `uncertain` flattens the distribution. `-p*log(p)` peaks at `p = 1/e`, so spreading mass
    into the middle ranks is what raises the entropy the detector reads -- pushing the top
    rank down alone would not, since a near-zero candidate contributes almost nothing.
    """
    # The top candidate's probability: confident answers concentrate, uncertain ones do not.
    top = rng.uniform(0.30, 0.55) if uncertain else rng.uniform(0.72, 0.95)
    remaining = 1.0 - top

    # The tail is drawn then normalised to whatever mass the top rank left, so the K
    # probabilities always sum to at most 1 and `logprob` is never positive -- which the
    # parser rejects by name.
    tail = sorted((rng.uniform(0.2, 1.0) for _ in range(K - 1)), reverse=True)
    scale = remaining / sum(tail)
    probabilities = [top, *(value * scale for value in tail)]

    candidates = [
        {"token": token if rank == 0 else f"«alt{rank}»", "logprob": round(math.log(p), 4)}
        for rank, p in enumerate(probabilities)
    ]
    return {"token": token, "logprob": candidates[0]["logprob"], "top_logprobs": candidates}


def completion(custom_id: str, text: str, uncertain: bool, rng: random.Random) -> dict:
    content = [logprobs_for(token, uncertain, rng) for token in tokens_of(text)]
    return {
        "id": f"chatcmpl-{custom_id}",
        "object": "chat.completion",
        "created": 0,
        "model": "synthetic-stand-in",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
                "logprobs": {"content": content},
            }
        ],
    }


def envelope(custom_id: str, body: dict, prefix: str = "batch") -> dict:
    return {
        "id": body["id"],
        "custom_id": custom_id,
        "response": {"status_code": 200, "request_id": f"{prefix}-{custom_id}", "body": body},
        "error": None,
    }


def verdict(custom_id: str, correct: bool, gold: str) -> dict:
    """A judge's reply, kept whole the way the pipeline writes it.

    `judgment` is in the judge's direction -- `true` means the answer was correct -- because
    that is what `read_judgment` returns and what `scripts/train_detector.py` negates. The
    label the detector trains on is `int(not verdict)`.
    """
    body = {
        "id": f"chatcmpl-judge-{custom_id}",
        "object": "chat.completion",
        "created": 0,
        "model": "synthetic-judge",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "judgment": correct,
                            "explanation": (
                                f"the answer agrees with {gold!r}"
                                if correct
                                else f"the answer contradicts {gold!r}"
                            ),
                        }
                    ),
                },
            }
        ],
    }
    return envelope(custom_id, body, prefix="batch-judge")


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    # A wrong answer is another question's gold answer, from the same source, so it reads as
    # a confusion rather than as noise -- and it is text, which is what the judge fixture
    # has to explain away.
    golds = {source: [] for source in ACCURACY}
    for question in questions:
        golds[source_of(question["question_id"])].append(question["short_answer"])

    # The failures are chosen up front so they are spread across both sources rather than
    # landing wherever the loop happens to be.
    failed = set(rng.sample([question["question_id"] for question in questions], FAILURES))

    responses, judgments = [], []
    for question in questions:
        custom_id = question["question_id"]
        if custom_id in failed:
            responses.append(
                {
                    "id": f"chatcmpl-{custom_id}",
                    "custom_id": custom_id,
                    "response": None,
                    "error": {"code": "rate_limit_exceeded", "message": "request timed out"},
                }
            )
            continue

        source = source_of(custom_id)
        correct = rng.random() < ACCURACY[source]
        gold = question["short_answer"]
        if correct:
            answer = gold
        else:
            others = [other for other in golds[source] if other != gold]
            answer = rng.choice(others)

        responses.append(envelope(custom_id, completion(custom_id, answer, not correct, rng)))
        judgments.append(verdict(custom_id, correct, gold))

    for path, records in ((RESPONSES, responses), (JUDGMENTS, judgments)):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        print(f"{path.relative_to(EXAMPLES.parents[1])}: {len(records)} lines")


if __name__ == "__main__":
    main()
