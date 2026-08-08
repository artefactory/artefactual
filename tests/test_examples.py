"""Execute the example notebooks against the current source.

The documentation ships the notebooks with their committed outputs and does not
re-execute them (`nbsphinx_execute = "never"`), which keeps the site reproducible offline
but means nothing would notice if the API drifted out from under them. These tests are
what makes that trade safe: they run the notebooks for real and fail when the published
examples stop working.

The Langfuse notebook generates against a live endpoint, so it is checked statically --
imports resolve, names are defined -- rather than executed.
"""

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"

OFFLINE_NOTEBOOKS = ["epr_usage_demo", "wepr_usage_demo"]
NETWORKED_NOTEBOOKS = ["langfuse_integration_demo"]
ALL_NOTEBOOKS = OFFLINE_NOTEBOOKS + NETWORKED_NOTEBOOKS


def load(name):
    return json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))


def code_of(notebook):
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_the_notebook_is_shipped_with_the_docs(name):
    # nbsphinx resolves them relative to the Sphinx source dir; outside it they vanish
    assert (EXAMPLES / f"{name}.ipynb").is_file()


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_the_notebook_is_listed_in_the_toctree(name):
    # a notebook not in the toctree builds to an orphan page Sphinx warns about
    assert name in (EXAMPLES / "index.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_the_notebook_parses(name):
    compile(code_of(load(name)), name, "exec")


@pytest.fixture
def _detectors_resolve_locally(monkeypatch, tmp_path):
    """Resolve a published detector name to an estimator written here, not fetched.

    The notebooks name a Hugging Face repository, which is what a reader should copy, so
    running them as written reaches the network -- and a runner blip then fails a build
    that has nothing to do with the Hub. Only the resolution step is replaced: the
    notebook's own code, `read_estimator`, and the whole pipeline still run for real.

    The width follows the reduction the name carries, because that is what the classifier
    is checked against: EPR pools to a single coefficient, WEPR keeps `2k`.
    """
    import skops.io as sio
    from conftest import fitted_logistic

    from artefactual.scoring.base_detector import BaseDetector

    def resolve(identifier, *_args, **_kwargs):
        n_features = 1 if "-epr-" in str(identifier) else 2 * 15
        path = tmp_path / f"{n_features}.skops"
        if not path.exists():
            sio.dump(fitted_logistic(-0.5, [0.1] * n_features), path)
        return path

    monkeypatch.setattr(BaseDetector, "resolve_estimator", staticmethod(resolve))


@pytest.mark.parametrize("name", OFFLINE_NOTEBOOKS)
def test_the_notebook_runs_against_the_current_source(name, monkeypatch, _detectors_resolve_locally):
    """Execute every code cell in order, from the notebook's own directory.

    Run in-process rather than through nbconvert: the failure surfaces as an ordinary
    traceback pointing at the offending cell, and there is no kernel to install.
    """
    monkeypatch.chdir(EXAMPLES)
    namespace = {"__name__": "__main__"}

    exec(compile(code_of(load(name)), name, "exec"), namespace)


@pytest.mark.parametrize("name", OFFLINE_NOTEBOOKS)
def test_the_committed_outputs_are_not_empty(name):
    """A notebook stripped of outputs renders as a blank page on the docs site."""
    notebook = load(name)
    executed = [c for c in notebook["cells"] if c["cell_type"] == "code" and c.get("outputs")]

    assert executed, f"{name} carries no cell outputs; re-run it before committing"


@pytest.mark.parametrize("name", OFFLINE_NOTEBOOKS)
def test_the_committed_outputs_carry_no_errors(name):
    notebook = load(name)
    errors = [o for c in notebook["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]

    assert not errors, f"{name} was committed with an error output: {errors[:1]}"
