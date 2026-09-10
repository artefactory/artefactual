"""Reading an LLM-as-a-judge verdict out of the completion that carries it.

A detector is fitted on labels, and the label a batch run produces is a judge's reply: a
second completion, keyed to the answer by the same `custom_id`. Reading it is the one step
between two batch files and a training set, and it is not a `json.loads` -- which is why it
lives here rather than in every caller.
"""

import contextlib
import json
from typing import Any

from artefactual.preprocessing.response_models import read_message

# What the judge prompt asks for, and what a reply that ignored the format is scanned for.
_JUDGMENT = "judgment"
_BARE = {"true": True, "true.": True, "false": False, "false.": False}


def _from_json(content: str) -> bool | None:
    """The verdict as the judge was asked to write it: a JSON object with a boolean.

    Only a real boolean counts. `bool("false")` is True, so a judge emitting the value as a
    string -- which a loose schema invites -- would mark every wrong answer correct,
    silently, leaving a label distribution that still looks plausible. Such a reply falls
    through to the scan instead.
    """
    with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError):
        verdict = json.loads(content)[_JUDGMENT]
        if verdict is True or verdict is False:
            return verdict
    return None


def _from_fence(content: str) -> bool | None:
    """The verdict from a reply that wrapped the object in a Markdown code fence.

    Common enough to be worth its own step: the fenced object is valid JSON once the fence
    is gone, so this recovers the boolean rather than falling through to a substring scan
    that cannot tell `"judgment": true` in the object from the same text quoted in prose.
    """
    if "```" not in content:
        return None
    _, _, after = content.partition("```")
    body, _, _ = after.partition("```")
    return _from_json(body.removeprefix("json").strip())


def _from_scan(content: str) -> bool | None:
    """The verdict from a reply that is prose, or a bare `true` / `false`.

    Last resort, and deliberately narrow: the key spelled as the prompt asks for it, or a
    reply that is nothing but the word. Anything looser reads a verdict out of a sentence
    that merely discusses one.
    """
    lowered = content.lower()
    if lowered.strip() in _BARE:
        return _BARE[lowered.strip()]
    if f'"{_JUDGMENT}": true' in lowered:
        return True
    if f'"{_JUDGMENT}": false' in lowered:
        return False
    return None


def read_judgment(completion: Any) -> bool | None:
    """Whether the judge said the answer was CORRECT, or `None` if its reply cannot be read.

    Note the polarity: `judgment: true` means the answer was correct, which is the opposite
    of the class a detector predicts, so a caller building labels wants `not read_judgment(
    ...)`. The convention is the judge prompt's, not this package's, and renaming it here
    would put the two out of step.

    The judge is asked for `{"judgment": true|false, "explanation": "..."}`. Models wrap
    that in prose or a Markdown fence often enough that a bare `json.loads` is unsafe, so
    three readings are tried in decreasing order of trust: the reply as JSON, the reply's
    fenced block as JSON, then a scan for the key. A reply none of them can read returns
    `None` rather than a guess -- an unreadable verdict is a row to drop and count, and a
    guessed one is a mislabelled row that nothing downstream can notice.

    Args:
        completion: A chat completion, as a mapping or an attribute-style object. A batch
            line's `completion` is exactly this, including the `None` a failed line yields.

    Returns:
        `True` when the judge said the answer was correct, `False` when it said it was not,
        `None` when there is no readable verdict -- an unreadable reply, a completion with
        no message, or no completion at all.
    """
    # A failed batch line, a rejected request's error object and a reply with no text are
    # all `None` from `read_message`, and none of them is worth a different answer than a
    # reply that simply could not be read: all of them are a row to drop and count. Raising
    # would make the caller branch on which kind of nothing it got.
    content = read_message(completion)
    if content is None:
        return None
    for reading in (_from_json, _from_fence, _from_scan):
        verdict = reading(content)
        if verdict is not None:
            return verdict
    return None
