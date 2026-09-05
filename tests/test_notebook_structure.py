"""Structural checks on the example notebooks, of the kind no reader would spot.

Not for the documentation site: docutils renormalises section levels, so a notebook whose
markdown starts at `##` still renders `<h1>` there, and a `#` to `###` jump still renders
`<h1>` then `<h2>`. Two of these notebooks had exactly those defects and both pages looked
correct.

It matters everywhere the notebook is rendered *without* that normalisation -- GitHub's
notebook preview, nbviewer, Colab -- which is how most readers meet a notebook in a pull
request. There a skipped level shows as a skipped level and a `##` title shows undersized.

Cheap to check and impossible to notice by eye, which is the case for a test.
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


def headings(name):
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
    notebook = json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))
    return [
        (int(opening.tag.removeprefix("h")), inline.content)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        for opening, inline in pairwise(_COMMONMARK.parse("".join(cell["source"])))
        if opening.type == "heading_open"
    ]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_starts_at_a_single_h1(name):
    """One top-level heading, first, so a raw render shows a title rather than a section."""
    levels = [level for level, _ in headings(name)]

    assert levels, f"{name} has no headings, so its page has no title"
    assert levels[0] == 1, f"{name} starts at H{levels[0]}; a raw render shows that as a section, not a title"
    assert levels.count(1) == 1, f"{name} has {levels.count(1)} H1s; a page has one title"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_never_skips_a_heading_level(name):
    """H1 to H3 reads as a missing section wherever levels are taken literally."""
    found = headings(name)

    skips = [
        f"H{previous_level} {previous!r} -> H{current_level} {current!r}"
        for (previous_level, previous), (current_level, current) in pairwise(found)
        if current_level - previous_level > 1
    ]
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

    All-absent is the deliberate state of the notebooks that ship without outputs because
    they need a live endpoint -- so it is an escape hatch only when the outputs are absent
    too. Counts stripped from a notebook that kept its outputs is the same stale-output
    defect wearing the exemption, and a jupytext or nbstripout round-trip produces exactly
    that.

    An output carries the count of the run that produced it, which is a second copy of the
    same fact and disagrees when a single cell was re-run in place.

    This catches the partial re-run, which is the accident that happens. It cannot catch
    an edit followed by no run at all; nothing short of executing the notebook can, and
    two of these download published weights to run, so executing them here would compare
    against numbers this suite deliberately does not reproduce.
    """
    notebook = json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code]
    ran = [count for count in counts if count is not None]

    if not ran:
        outputs = [cell for cell in code if cell.get("outputs")]
        assert not outputs, (
            f"{name} carries outputs on {len(outputs)} cell(s) but no execution counts, so "
            f"nothing says the outputs came from the code beside them; re-run it top to bottom"
        )
        return

    assert ran == list(range(1, len(counts) + 1)), (
        f"{name} has execution counts {counts}; re-run it top to bottom before committing"
    )

    disagree = [
        (cell["execution_count"], output["execution_count"])
        for cell in code
        for output in cell.get("outputs", [])
        if output.get("execution_count") not in (None, cell["execution_count"])
    ]
    assert not disagree, (
        f"{name} has cells whose output was produced by a different run (cell, output): "
        f"{disagree}; re-run it top to bottom before committing"
    )


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
