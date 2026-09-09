"""Reading an OpenAI Batch output file into the model that owns its shape.

Every stage of a training run reads one of these files -- the generations, then the
verdicts -- and the reading is the same each time: one JSON object per line, validated into
`BatchRequestOutput`, joined on `custom_id`. It lives here so that a caller writes the join
rather than the parser.

Lines are returned as they were read, failures included, because whether a failed request
is worth reporting is the caller's decision and `failure` says why for the ones that are.
"""

from pathlib import Path

from artefactual.preprocessing.response_models import BatchRequestOutput


def read_batch(path: Path | str) -> list[BatchRequestOutput]:
    """Every line of a Batch output file, in file order.

    Blank lines are skipped; a line that is not a batch record raises through pydantic,
    naming the field that was wrong.

    A repeated `custom_id` raises, whether or not either line carries a completion. It is
    the key responses are paired to their verdicts by, so a duplicate is an ambiguous join
    -- and the ambiguity surfaces later as a mislabelled row rather than as an error.
    """
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = BatchRequestOutput.model_validate_json(line)
        if row.custom_id in seen:
            msg = (
                f"{path} line {number}: custom_id {row.custom_id!r} appears more than once; "
                f"the join would be ambiguous."
            )
            raise ValueError(msg)
        seen.add(row.custom_id)
        rows.append(row)
    return rows


def index_by_custom_id(rows: list[BatchRequestOutput]) -> dict[str, BatchRequestOutput]:
    """The rows that carry a completion, keyed by the id every stage joins on."""
    return {row.custom_id: row for row in rows if row.completion}


__all__ = ["index_by_custom_id", "read_batch"]
