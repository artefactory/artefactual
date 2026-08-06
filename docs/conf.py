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
