"""Sphinx configuration for Artefactual documentation."""

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

# HTML output
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

# Rendered decks, copied verbatim into the site root. Populated by
# `quarto render docs/presentations`; a plain Sphinx build without Quarto still
# succeeds, it just warns that the directory is missing.
html_extra_path = ["_extra"]
