"""Heading structure of the example notebooks.

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

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
NOTEBOOKS = sorted(path.stem for path in EXAMPLES.glob("*.ipynb"))


def headings(name):
    """Every markdown heading in the notebook, as (level, text), in document order.

    Only headings at the start of a line count: `#` inside a fenced block is a Python
    comment, and inside prose it is a character.
    """
    notebook = json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))
    found, fenced = [], False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced and line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                found.append((level, line.lstrip("# ").strip()))
    return found


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
        (previous, current)
        for (previous_level, previous), (current_level, current) in pairwise(found)
        if current_level - previous_level > 1
    ]
    assert not skips, f"{name} jumps more than one level at: {skips}"
