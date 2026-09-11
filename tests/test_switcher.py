"""The version switcher is generated during the deploy, where nothing else checks it.

A wrong answer here is silent: the site still builds and still deploys, it just offers the
wrong version as `stable` or drops a release out of the dropdown.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "build_switcher.py"

_spec = importlib.util.spec_from_file_location("build_switcher", _SCRIPT)
build_switcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_switcher)

BASE = "https://artefactory.github.io/artefactual"


def make_site(tmp_path: Path, names: list[str]) -> Path:
    for name in names:
        (tmp_path / name).mkdir()
    return tmp_path


def test_versions_are_ordered_numerically_not_lexically(tmp_path):
    """2026.10.0 is newer than 2026.9.0, and string order says the opposite.

    Sorted as strings, "2026.10.0" < "2026.9.0", so the first CalVer month to reach two
    digits would publish the previous release as `stable`.
    """
    site = make_site(tmp_path, ["2026.8.1", "2026.9.0", "2026.10.0"])

    assert build_switcher.released_versions(site) == ["2026.10.0", "2026.9.0", "2026.8.1"]


def test_only_release_directories_are_offered(tmp_path):
    """The site root also holds `stable`, `_static` and the redirecting index."""
    site = make_site(tmp_path, ["2026.9.0", "stable", "_static", "dev"])
    (site / "index.html").write_text("")

    assert build_switcher.released_versions(site) == ["2026.9.0"]


def test_the_newest_release_is_offered_once_as_stable(tmp_path):
    """Two entries with one version would leave the duplicate unreachable as current."""
    site = make_site(tmp_path, ["2026.8.1", "2026.9.0"])

    entries = build_switcher.switcher_entries(build_switcher.released_versions(site), BASE)

    assert [e["version"] for e in entries] == ["2026.9.0", "2026.8.1"]
    assert entries[0]["url"] == f"{BASE}/stable/"
    assert entries[0]["preferred"] is True
    assert entries[1]["url"] == f"{BASE}/2026.8.1/"


def test_every_entry_carries_the_keys_the_theme_requires(tmp_path):
    """pydata-sphinx-theme reads `version` and `url`; it warns on an entry missing either.

    It would check this itself by fetching the file at build time, which this site cannot
    do -- the file is written by the deploy, after the build.
    """
    site = make_site(tmp_path, ["2026.8.0", "2026.8.1", "2026.9.0"])

    entries = build_switcher.switcher_entries(build_switcher.released_versions(site), BASE)

    assert all({"version", "url"} <= set(e) for e in entries)


def test_an_empty_archive_is_refused(tmp_path):
    """Publishing a switcher with no versions would render an empty dropdown."""
    with pytest.raises(SystemExit):
        build_switcher.write_index(str(tmp_path), BASE)


def test_the_written_file_is_the_entries(tmp_path):
    site = make_site(tmp_path, ["2026.9.0"])

    build_switcher.write_index(str(site), BASE)

    assert json.loads((site / "switcher.json").read_text()) == build_switcher.switcher_entries(["2026.9.0"], BASE)


@pytest.mark.parametrize(
    ("tagged", "installed"),
    [("2026.09.0", "2026.9.0"), ("2026.08.1", "2026.8.1"), ("2026.10.0", "2026.10.0")],
)
def test_the_padded_tag_becomes_the_version_the_package_reports(tagged, installed):
    """The tag and the installed package spell the same release differently.

    `calver_format = "{YYYY}.{0M}"` pads the month, so the tag is v2026.09.0, while PEP 440
    normalisation strips the zero and `artefactual.__version__` is 2026.9.0. conf.py takes
    `version_match` from the package, so a directory named after the tag would never be
    matched and the dropdown would mark nothing as current.
    """
    assert build_switcher.canonical_version(tagged) == installed


def test_the_directory_name_matches_what_conf_py_compares_against(tmp_path):
    """End to end: the gate's string in, an entry conf.py's version_match can match out."""
    from_the_gate = "2026.09.0"

    published_as = build_switcher.canonical_version(from_the_gate)
    site = make_site(tmp_path, [published_as])
    entries = build_switcher.switcher_entries(build_switcher.released_versions(site), BASE)

    # conf.py sets version_match = artefactual.__version__, which for this release is:
    version_match = "2026.9.0"
    assert any(e["version"] == version_match for e in entries)
