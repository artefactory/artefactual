"""Sphinx configuration for Artefactual documentation."""

import os
from pathlib import Path

import artefactual

# Project information
project = "Artefactual"
copyright = "2025, Artefact Research Center"  # noqa: A001
author = "Hicham Randrianarivo, Gauthier Jeannin, Charles Moslonka"

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
nbsphinx_execute = "never"

# A badge above every notebook page that opens the same file in Colab, so a reader can run
# the example without a checkout. Colab loads a notebook from GitHub by owner, repo, branch
# and path, and all four have to be right or the link 404s -- so the first three come from
# the environment where CI sets them, and fall back to this repository's own values for a
# local build. A fork's docs then link into the fork, and a renamed default branch does not
# silently break every badge on the site.
#
# A notebook on an unmerged branch still gets a badge that 404s on a `main` build: the file
# is not there yet. That resolves itself when the branch lands.
#
# `env.doc2path(..., base=None)` is the source file relative to the source directory,
# extension included (`examples/epr_usage_demo.ipynb`), so prefixing `docs/` gives the
# file's path in the repository. `|string` is required rather than decorative: doc2path
# returns a `_StrPath`, which subclasses `PurePath` and not `str`, so `+` on it raises.
#
# The prolog applies to every notebook nbsphinx renders. The Quarto decks under
# `presentations/` are excluded from the build by `exclude_patterns` below, so they never
# see it.
_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "artefactory/artefactual")
_BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
# One placeholder and one substitution, rather than Jinja variables: the prolog is
# rendered per document with a context nbsphinx owns, so values from here reach it by
# being in the string already.
nbsphinx_prolog = """
{% set docname = "docs/" + env.doc2path(env.docname, base=None)|string %}

.. raw:: html

    <p style="margin-bottom: 1.5rem">
      <a href="COLAB_PREFIX{{ docname }}" target="_blank" rel="noopener">
        <img src="https://colab.research.google.com/assets/colab-badge.svg"
             alt="Open {{ docname }} in Colab" style="vertical-align: middle">
      </a>
    </p>
""".replace("COLAB_PREFIX", f"https://colab.research.google.com/github/{_REPOSITORY}/blob/{_BRANCH}/")

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
llms_txt_exclude = ["_autosummary/*", "examples/*_demo", "presentations/index"]
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
_DOCS = Path(__file__).parent
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
    "show_nav_level": 2,
    "navigation_depth": 3,
}

# General
templates_path = ["_templates"]
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
