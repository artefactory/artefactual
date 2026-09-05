"""Structural checks on the example notebooks, of the kind no reader would spot.

Not for the documentation site: docutils renormalises section levels, so a notebook whose
markdown starts at `##` still renders `<h1>` there, and a `#` to `###` jump still renders
`<h1>` then `<h2>`. Two of these notebooks had exactly those defects and both pages looked
correct.

It matters everywhere the notebook is rendered *without* that normalisation -- GitHub's
notebook preview, nbviewer, Colab -- which is how most readers meet a notebook in a pull
request. There a skipped level shows as a skipped level and a `##` title shows undersized.

Cheap to check and impossible to notice by eye, which is the case for a test.

Each check is a function returning the complaints it found, so the same code that runs
against the shipped notebooks also runs against notebooks written here to carry a defect.
A check nothing has ever been seen to fail is a check that might not be able to.
"""

import json
from itertools import pairwise
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

# CommonMark, which is what the renderers this file is about implement. Configured once:
# parsing is per cell, and a parser holds no state between calls.
_COMMONMARK = MarkdownIt("commonmark")

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
NOTEBOOKS = sorted(path.stem for path in EXAMPLES.glob("*.ipynb"))


def load(name):
    return json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))


def notebook_of(*cells):
    """A notebook holding exactly `cells`, for driving a check against a known defect."""
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(execution_count, outputs=()):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": list(outputs),
        "source": ["pass\n"],
    }


def result(execution_count):
    """An `execute_result` output, which carries a second copy of its cell's run count."""
    return {
        "data": {"text/plain": ["1"]},
        "execution_count": execution_count,
        "metadata": {},
        "output_type": "execute_result",
    }


def headings(notebook):
    """Every markdown heading in the notebook, as (level, text), in document order.

    Parsed rather than scanned for `#`. The question this file asks is what a renderer
    shows, so anything short of a CommonMark parser disagrees with the answer somewhere:
    `#hashtag` is not a heading and seven hashes are a paragraph; `~~~` fences a block and
    ```` ``` ```` inside a four-space indent does not; three leading spaces still make a
    heading; `Title` over `=====` is an h1 with no `#` in sight.

    One parse per cell, because a renderer treats each cell as its own document -- a fence
    left open in one cell does not swallow the next, which a single scan over the whole
    notebook cannot know.
    """
    return [
        (int(opening.tag.removeprefix("h")), inline.content)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        for opening, inline in pairwise(_COMMONMARK.parse("".join(cell["source"])))
        if opening.type == "heading_open"
    ]


def title_complaints(found):
    """Why the notebook's headings do not open on exactly one H1, if they do not."""
    levels = [level for level, _ in found]
    if not levels:
        return ["has no headings, so its page has no title"]

    complaints = []
    if levels[0] != 1:
        complaints.append(f"starts at H{levels[0]}; a raw render shows that as a section, not a title")
    if levels.count(1) != 1:
        complaints.append(f"has {levels.count(1)} H1s; a page has one title")
    return complaints


def skipped_levels(found):
    """Every place the heading level jumps by more than one, as a readable pair."""
    return [
        f"H{previous_level} {previous!r} -> H{current_level} {current!r}"
        for (previous_level, previous), (current_level, current) in pairwise(found)
        if current_level - previous_level > 1
    ]


def stale_run_complaints(notebook):
    """Why the committed outputs cannot have come from one top-to-bottom run, if they cannot.

    All-absent counts are the deliberate state of the notebooks that ship without outputs
    because they need a live endpoint -- so that is an escape hatch only when the outputs
    are absent too. Counts stripped from a notebook that kept its outputs is the same
    stale-output defect wearing the exemption, and a jupytext or nbstripout round-trip
    produces exactly that.

    An output carries the count of the run that produced it, which is a second copy of the
    same fact and disagrees when a single cell was re-run in place.
    """
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code]
    ran = [count for count in counts if count is not None]

    if not ran:
        carrying = [cell for cell in code if cell.get("outputs")]
        if carrying:
            complaint = (
                f"carries outputs on {len(carrying)} cell(s) but no execution counts, so "
                f"nothing says the outputs came from the code beside them"
            )
            return [complaint]
        return []

    if ran != list(range(1, len(counts) + 1)):
        return [f"has execution counts {counts}"]

    disagree = [
        (cell["execution_count"], output["execution_count"])
        for cell in code
        for output in cell.get("outputs", [])
        if output.get("execution_count") not in (None, cell["execution_count"])
    ]
    if disagree:
        return [f"has cells whose output was produced by a different run (cell, output): {disagree}"]
    return []


# --- the shipped notebooks -------------------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_starts_at_a_single_h1(name):
    """One top-level heading, first, so a raw render shows a title rather than a section."""
    assert not title_complaints(headings(load(name))), f"{name}: {title_complaints(headings(load(name)))}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_never_skips_a_heading_level(name):
    """H1 to H3 reads as a missing section wherever levels are taken literally."""
    skips = skipped_levels(headings(load(name)))

    assert not skips, (
        f"{name} jumps more than one heading level at: {skips}. Promote the second, or give "
        f"it a parent one level below the first."
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_outputs_come_from_one_top_to_bottom_run(name):
    """Execution counts read 1..N, or are absent throughout.

    Editing one cell and re-running only that cell leaves its neighbours' outputs
    describing the code that used to be above them. Every other check still passes --
    the outputs are there and none of them is an error -- and the published page shows
    results that never came from the code beside them. The counts are the only trace of
    it left in the file.

    This catches the partial re-run, which is the accident that happens. It cannot catch
    an edit followed by no run at all; nothing short of executing the notebook can, and
    two of these download published weights to run, so executing them here would compare
    against numbers this suite deliberately does not reproduce.
    """
    complaints = stale_run_complaints(load(name))

    assert not complaints, f"{name} {complaints[0]}; re-run it top to bottom before committing"


def test_every_notebook_is_classified_by_the_execution_tests():
    """`test_examples.py` names its notebooks; this file globs the directory.

    Two consequences, both silent. A notebook added to `docs/examples/` is picked up here
    and by the docs build, but `test_examples.py` runs only the names in its two lists, so
    a new notebook ships un-executed with the suite still green. And an empty glob makes
    every parametrised test in this file *skip* -- pytest's default for an empty parameter
    set -- which is a pass, and is reachable: the sdist ships `/tests` without `/docs`.

    Comparing the two settles both at once.
    """
    from test_examples import ALL_NOTEBOOKS

    assert sorted(ALL_NOTEBOOKS) == NOTEBOOKS, (
        f"docs/examples holds {NOTEBOOKS}; test_examples.py runs {sorted(ALL_NOTEBOOKS)}. "
        f"Add the new notebook to OFFLINE_NOTEBOOKS or NETWORKED_NOTEBOOKS there."
    )


# --- the checks themselves, against markdown a `#` scanner reads wrong -------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("# Title\n", [(1, "Title")]),
        ("   # Title\n", [(1, "Title")]),  # up to three leading spaces still opens a heading
        ("Title\n=====\n", [(1, "Title")]),  # setext, with no `#` to find
        ("Section\n-------\n", [(2, "Section")]),
        ("#hashtag no space\n", []),  # ATX needs the space; this is a paragraph
        ("####### seven hashes\n", []),  # six is the deepest heading there is
        ("```\n# fenced\n```\n", []),
        ("~~~\n# also fenced\n~~~\n", []),  # tildes fence too
        ("    ```\n    # indented, so not a fence\n", []),  # a code block, not a heading
        ("> # quoted\n", [(1, "quoted")]),  # still a heading, still rendered
    ],
)
def test_the_heading_parser_agrees_with_commonmark(text, expected):
    """A `#` scanner disagrees with a renderer on every one of these.

    The check is only worth its failure message if it reads the markdown the way the
    renderers it is about do, so the disagreements are pinned rather than assumed.
    """
    assert headings(notebook_of(markdown_cell(text))) == expected


def test_a_fence_left_open_in_one_cell_does_not_hide_the_next_cell_s_headings():
    """Each cell is its own document, which is what makes the per-cell parse necessary.

    A single pass over the concatenated notebook carries the open fence forward and stops
    seeing headings from there on -- so the file quietly stops being checked at the first
    cell with unbalanced backticks, and still passes.
    """
    found = headings(notebook_of(markdown_cell("```\nnever closed\n"), markdown_cell("# Title\n")))

    assert found == [(1, "Title")]


# --- the checks themselves, against notebooks built to fail them -------------------------


@pytest.mark.parametrize(
    ("cells", "complaint"),
    [
        pytest.param(["## Section\n"], "starts at H2", id="opens-below-h1"),
        pytest.param(["# One\n", "# Two\n"], "has 2 H1s", id="two-titles"),
        pytest.param(["no heading here\n"], "has no headings", id="untitled"),
    ],
)
def test_a_notebook_without_one_leading_h1_is_refused(cells, complaint):
    found = headings(notebook_of(*(markdown_cell(text) for text in cells)))

    assert any(complaint in line for line in title_complaints(found)), title_complaints(found)


def test_a_skipped_heading_level_is_refused():
    found = headings(notebook_of(markdown_cell("# Title\n"), markdown_cell("### Buried\n")))

    assert skipped_levels(found) == ["H1 'Title' -> H3 'Buried'"]


def test_returning_to_a_shallower_level_is_not_a_skip():
    """`#` `##` `###` `#` is an ordinary document: only descending too fast is a defect."""
    found = headings(
        notebook_of(
            markdown_cell("# Title\n"),
            markdown_cell("## Section\n"),
            markdown_cell("### Detail\n"),
            markdown_cell("## Next\n"),
        )
    )

    assert skipped_levels(found) == []


@pytest.mark.parametrize(
    ("cells", "complaint"),
    [
        pytest.param(
            [code_cell(1), code_cell(3), code_cell(2)],
            "execution counts [1, 3, 2]",
            id="re-run-out-of-order",
        ),
        pytest.param(
            [code_cell(1), code_cell(None), code_cell(3)],
            "execution counts [1, None, 3]",
            id="one-cell-never-run",
        ),
        pytest.param(
            [code_cell(1), code_cell(2, [result(1)])],
            "produced by a different run",
            id="output-from-an-earlier-run",
        ),
        pytest.param(
            [code_cell(None, [result(None)])],
            "no execution counts",
            id="counts-stripped-outputs-kept",
        ),
    ],
)
def test_outputs_that_cannot_come_from_one_run_are_refused(cells, complaint):
    complaints = stale_run_complaints(notebook_of(*cells))

    assert any(complaint in line for line in complaints), complaints


def test_a_notebook_that_ships_without_outputs_is_accepted():
    """The deliberate state of the notebooks that need a live endpoint to run."""
    assert stale_run_complaints(notebook_of(code_cell(None), code_cell(None))) == []
