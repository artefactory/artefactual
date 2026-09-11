"""Sphinx configuration for Artefactual documentation."""

import os
from pathlib import Path, PurePosixPath

import artefactual

# Project information
project = "Artefactual"
copyright = "2025, Artefact Research Center"  # noqa: A001
author = "Hicham Randrianarivo, Gauthier Jeannin, Charles Moslonka"

# Read from the installed package, which hatch-vcs derives from the git tag, so the site
# names a version that exists rather than one restated here and left to drift. A build
# from an untagged commit carries a `.dev` segment, which is what marks it unreleased.
release = artefactual.__version__
version = ".".join(release.split(".")[:2])
_is_development_build = ".dev" in release

# Where the published site lives. Fixed rather than derived, because the version switcher
# has to name one URL that every version of the site agrees on: each release ships a copy
# of this file, and they must all point at the same switcher.json, or an old page cannot
# offer the versions released after it.
SITE_URL = "https://artefactory.github.io/artefactual"

# The site is one directory per release, so a page has to say which release it belongs to.
html_baseurl = f"{SITE_URL}/{release}/" if not _is_development_build else SITE_URL

# Just the project. The version belongs in the switcher, which names it once and makes it
# navigable; repeating it here would also put the full local version
# (2026.8.1.post1.dev36+j772a71f36.d20260911) in the site's most prominent label on every
# build that is not a release.
html_title = f"{project} documentation"

# Extensions
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_llms_txt",
    "sphinxcontrib.mermaid",
]

# Optional extras autodoc must not need installed to document the modules that guard
# their imports behind TYPE_CHECKING.
autodoc_mock_imports = ["langfuse"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True

# Napoleon settings. The codebase documents with Google-style Args:/Returns:/Raises:
# sections; NumPy style stays on so a contributor using it is still parsed.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # The detectors subclass Pipeline and LogisticRegression, so their inherited
    # docstrings reference sklearn's glossary and labels.
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# MyST settings
myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# nbsphinx settings
#
# The published site runs its notebooks. Committed outputs are whatever the last person to
# open the notebook happened to produce -- which has already put an absolute path from a
# contributor's home directory on the public site -- so the release build executes them
# and publishes what it got. Every other build reads the committed outputs instead, because
# executing needs the network and would make a pull request wait on the Hugging Face Hub.
#
# A notebook that cannot run unattended opts out in its own metadata
# (`"nbsphinx": {"execute": "never"}`), which is where the reason belongs: it travels with
# the notebook rather than sitting in a list here that nobody updates.
nbsphinx_execute = "always" if os.environ.get("ARTEFACTUAL_EXECUTE_NOTEBOOKS") else "never"

# A notebook that reaches the Hub has to survive a slow answer; the default is 30 seconds.
nbsphinx_timeout = 600

# Every notebook page carries its own two entry points in the article header: open the
# notebook in Colab, or download the `.ipynb`. They live in a theme component
# (`_templates/notebook-buttons.html`) filled in by the `html-page-context` handler below,
# rather than in `nbsphinx_prolog`, so they render as part of the page furniture instead of
# as the first paragraph of the notebook's own content.
#
# Colab loads a notebook from GitHub by owner, repo, branch and path, and all four have to
# be right or the link 404s -- so the first three come from the environment where CI sets
# them, and fall back to this repository's own values for a local build. A fork's docs then
# link into the fork, and a renamed default branch does not silently break every badge on
# the site. A notebook on an unmerged branch still gets a badge that 404s on a `main`
# build: the file is not there yet, and that resolves itself when the branch lands.
_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "artefactory/artefactual")
_BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
_COLAB = f"https://colab.research.google.com/github/{_REPOSITORY}/blob/{_BRANCH}/docs/"


def _notebook_buttons(app, pagename: str, templatename: str, context: dict, doctree) -> None:  # noqa: ARG001
    """Give a notebook page the two values its header component renders, and others none.

    `env.nbsphinx_notebooks` maps a notebook's docname to the path nbsphinx copies it to in
    the output, next to the page it produced -- so the download href is a bare filename and
    resolves whatever the depth of the page. Same origin, which is what makes the browser
    honour `download` rather than navigating to a screenful of JSON; the theme's own "Show
    Source" link is not this, it serves `_sources/<name>.ipynb.txt`, which renders as text.

    Membership of that mapping is also what identifies a notebook page. Every other page
    leaves `notebook_filename` unset, and the component renders nothing.
    """
    notebook = getattr(app.env, "nbsphinx_notebooks", {}).get(pagename)
    if notebook is None:
        return
    context["notebook_filename"] = PurePosixPath(notebook).name
    context["colab_url"] = _COLAB + notebook


def setup(app) -> None:
    """Register the handler that fills the notebook header buttons."""
    app.connect("html-page-context", _notebook_buttons)


# sphinx-llms-txt settings
#
# The extension reads Sphinx *source* files, not rendered output, which decides everything
# below:
#
#   - `_autosummary/*` stubs are four lines of `.. automodule::` each. Rendered they become
#     the API reference; as source they are empty, so including them puts 19 links to
#     nothing in llms.txt and 18 bare directives in llms-full.txt.
#   - Notebooks are read as raw `.ipynb` JSON, outputs and all. Unfiltered they were 89% of
#     llms-full.txt, and the HTML pages remain the readable form of them.
#
# The API is supplied instead as the source itself, via `llms_txt_code_files`. That is
# strictly more than autodoc would have rendered: the same docstrings, plus the code and
# the comments explaining it.
#
# The notebooks are globbed rather than matched by name. `examples/*_demo` covered the three
# that existed when this was written, and a notebook called anything else silently landed in
# llms-full.txt as its own source -- one of them carrying a fixture of per-token logprob
# arrays. Globbing the directory cannot go stale that way. `examples/index` is prose and
# stays.
_DOCS = Path(__file__).parent
_NOTEBOOKS = [f"examples/{path.stem}" for path in sorted((_DOCS / "examples").glob("*.ipynb"))]
llms_txt_exclude = ["_autosummary/*", *_NOTEBOOKS, "presentations/index"]
# Each file is listed explicitly rather than globbed with `+:../src/artefactual/**/*.py`,
# because the extension's `-:` exclusions do not work for paths outside the source
# directory: it compares resolved include paths against unresolved exclude globs, so
# `-:../src/**/__init__.py` never matches and every package `__init__.py` is pulled in. They
# are dropped here instead, since eight sections all titled `__init__.py` help nobody.
#
# Titles are bare filenames whatever `llms_txt_code_base_path` is set to -- the extension
# derives them with `relative_to(srcdir)`, which raises for anything outside `docs/` and
# falls back to the basename. Every remaining module basename is unique, so that is legible.
# Located through the imported package rather than an assumed repo layout, so the list
# follows the installed source wherever it lives. os.path.relpath rather than
# Path.relative_to(walk_up=True): the latter is 3.12+, and the docs build runs on 3.11.
_PACKAGE = Path(artefactual.__file__).parent
llms_txt_code_files = [
    f"+:{os.path.relpath(path, _DOCS)}" for path in sorted(_PACKAGE.rglob("*.py")) if path.name != "__init__.py"
]

# HTML output
# The published site, which is where the docs are deployed from .github/workflows/docs.yml.
# sphinx-llms-txt needs it to emit absolute links: without it the entries in llms.txt are
# host-less paths like `/_sources/index.md.txt`, which no consumer can fetch.
html_baseurl = "https://artefactory.github.io/artefactual/"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/artefactory/artefactual",
    # The released versions, and which one you are reading. `version_match` is the release
    # string, matching the `version` key the deploy writes into switcher.json; on a
    # development build it matches nothing, which leaves the dropdown listing the releases
    # with none marked current -- correct, because an unreleased build is not among them.
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": f"{SITE_URL}/switcher.json",
        "version_match": release,
    },
    # The theme fetches json_url at build time and *warns* when it cannot be read, which
    # -W turns into a failed build. That fetch would have to succeed before the deploy that
    # publishes the file -- impossible for the release that introduces the switcher, and a
    # network dependency in every build after it. The file's shape is asserted by the deploy
    # that generates it instead.
    "check_switcher": False,
    # The slot is empty by default; the theme puts downloads out of scope, so the component
    # is this repository's own.
    "article_header_end": ["notebook-buttons.html"],
    "show_nav_level": 2,
    "navigation_depth": 3,
}

# The site is published from the release pipeline, so what it documents is a released
# version and the two cannot disagree. A build from anywhere else -- a pull request, a
# working copy -- says so, because the alternative is a page that reads as published while
# describing an API `pip install artefactual` does not give you.
if _is_development_build:
    html_theme_options["announcement"] = (
        "This is the development version of the documentation. It describes unreleased "
        "changes. The released version is on "
        '<a href="https://pypi.org/project/artefactual/">PyPI</a>.'
    )

# Mermaid renders in the browser, so a diagram is live SVG rather than an image: it selects,
# it scales, and with d3 zoom it pans like the ones GitHub renders. The palette it bakes in
# is light-only, which `_static/mermaid-theme.css` restates in the theme's own variables.
mermaid_d3_zoom = True

# General
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["mermaid-theme.css", "notebook-buttons.css"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Quarto renders the decks into docs/_extra; keep nbsphinx off the sources.
    "presentations/**/*.ipynb",
]

# Rendered decks, copied verbatim into the site root, produced by
# `quarto render docs/presentations`. Listed only when that directory is present: a missing
# extra path is a warning, and the build runs with -W, so naming it unconditionally would
# fail any build that skipped Quarto. An empty directory warns about nothing either way,
# so CI asserts the decks reached the site rather than relying on this entry.
html_extra_path = ["_extra"] if (Path(__file__).parent / "_extra").is_dir() else []
