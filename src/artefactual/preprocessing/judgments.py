"""Reading an LLM-as-a-judge verdict out of the completion that carries it.

A detector is fitted on labels, and the label a batch run produces is a judge's reply: a
second completion, keyed to the answer by the same `custom_id`. Reading it is the one step
between two batch files and a training set, and it is not a `json.loads` -- which is why it
lives here rather than in every caller.
"""

import contextlib
import json
from collections.abc import Iterable
from typing import Any

from beartype.door import is_bearable
from pydantic import BaseModel, StrictBool, TypeAdapter, ValidationError

from artefactual.preprocessing.response_models import BatchRequestOutput, read_message

# Reads a JSON boolean and nothing else: no `"true"` string, no `1`.
BARE_VERDICT = TypeAdapter(StrictBool)


class Judgment(BaseModel):
    """What a judge said about an answer: the verdict, and why.

    This is the schema the judge prompt asks for, `{"judgment": true|false, "explanation":
    "..."}`, so the class can be passed as a structured-output `response_format` and
    validated like any other model. The constructors read the replies that ignored it.

    Note the polarity: `judgment=True` means the answer was CORRECT, which is the opposite
    of the class a detector predicts, so a caller building labels wants `not judgment`.
    The convention is the judge prompt's, not this package's, and renaming it here would
    put the two out of step.

    Parameters
    ----------
    judgment : bool
        `True` when the judge said the answer was correct. Only a real boolean is
        accepted: `bool("false")` is True, so a judge emitting the value as a string --
        which a loose schema invites -- would mark every wrong answer correct, silently,
        leaving a label distribution that still looks plausible.
    explanation : str, default=""
        The judge's reasoning. The verdict never depends on it.

    Examples
    --------
    >>> Judgment.from_text('{"judgment": true, "explanation": "matches"}').judgment
    True
    >>> Judgment.from_text("I cannot decide.") is None
    True
    """

    judgment: StrictBool
    explanation: str = ""

    @classmethod
    def from_text(cls, reply: str) -> "Judgment | None":
        """Read a judge's reply, or `None` if it carries no verdict.

        Models wrap the requested object in prose or a Markdown fence often enough that a
        bare `json.loads` is unsafe, so three readings are tried in decreasing order of
        trust:

        1. the reply as JSON, an object with a boolean verdict;
        2. the reply's fenced block as JSON. Common enough to be worth its own step: the
           fenced object is valid JSON once the fence is gone, so this recovers the
           boolean rather than falling through to a substring scan that cannot tell
           `"judgment": true` in the object from the same text quoted in prose;
        3. a scan, deliberately narrow: the key spelled as the prompt asks for it, or a
           reply that is nothing but `true` / `false`. Anything looser reads a verdict out
           of a sentence that merely discusses one.

        A reply none of them can read returns `None` rather than a guess -- an unreadable
        verdict is a row to drop and count, and a guessed one is a mislabelled row that
        nothing downstream can notice.

        Parameters
        ----------
        reply : str
            The text the judge wrote.

        Returns
        -------
        Judgment or None
            The verdict, with the `explanation` when the JSON object has one as a string;
            a missing, null or non-string explanation is `""`. `None` when no reading
            finds a verdict.
        """
        readable = [reply]
        if "```" in reply:
            _, _, after = reply.partition("```")
            body, _, _ = after.partition("```")
            readable.append(body.removeprefix("json").strip())
        for text in readable:
            with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError):
                fields = json.loads(text)
                verdict = fields["judgment"]
                if verdict is True or verdict is False:
                    explanation = fields.get("explanation")
                    return cls(judgment=verdict, explanation=explanation if is_bearable(explanation, str) else "")

        lowered = reply.lower()
        with contextlib.suppress(ValidationError):
            return cls(judgment=BARE_VERDICT.validate_json(lowered.strip().removesuffix(".")))
        for word, verdict in (("true", True), ("false", False)):
            if f'"judgment": {word}' in lowered:
                return cls(judgment=verdict)
        return None

    @classmethod
    def from_completion(cls, completion: Any) -> "Judgment | None":
        """Read the verdict out of the completion that carries the judge's reply.

        A failed batch line, a rejected request's error object and a reply with no text
        are all `None` from `read_message`, and none of them is worth a different answer
        than a reply that simply could not be read: all of them are a row to drop and
        count. Raising would make the caller branch on which kind of nothing it got.

        Parameters
        ----------
        completion : Any
            A chat completion, as a mapping or an attribute-style object. A batch line's
            `completion` is exactly this, including the `None` a failed line yields.

        Returns
        -------
        Judgment or None
            `None` for an unreadable reply, a completion with no message, or no
            completion at all.
        """
        reply = read_message(completion)
        return None if reply is None else cls.from_text(reply)

    @classmethod
    def from_batch(cls, rows: Iterable[BatchRequestOutput]) -> "list[Judgment | None]":
        """Read the verdict of every line of a judge's Batch output, in line order.

        One entry per row, so the result is aligned with `read_batch` and a caller joins
        on `custom_id` with `zip(rows, Judgment.from_batch(rows))`. `None` marks a line
        that failed or whose reply is unreadable: they are kept so the caller can count
        them, and dropping them here would silently shift every later verdict against its
        row.

        Parameters
        ----------
        rows : iterable of BatchRequestOutput
            Lines as `read_batch` returns them. Iterated once.

        Returns
        -------
        list of Judgment or None
        """
        return [cls.from_completion(row.completion) for row in rows]


def read_judgment(completion: Any) -> bool | None:
    """Whether the judge said the answer was CORRECT, or `None` if its reply cannot be read.

    Note the polarity: `True` means the answer was correct, which is the opposite of the
    class a detector predicts, so a caller building labels wants `not read_judgment(...)`.
    This is `Judgment.from_completion(completion).judgment`, which also keeps the
    explanation.

    Args:
        completion: A chat completion, as a mapping or an attribute-style object. A batch
            line's `completion` is exactly this, including the `None` a failed line yields.

    Returns:
        `True` when the judge said the answer was correct, `False` when it said it was not,
        `None` when there is no readable verdict -- an unreadable reply, a completion with
        no message, or no completion at all.
    """
    judgment = Judgment.from_completion(completion)
    return None if judgment is None else judgment.judgment
