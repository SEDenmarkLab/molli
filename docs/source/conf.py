# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import molli.chem
import molli

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Molli"
copyright = "2022, Alexander Shved"
author = "Alexander Shved"
release = molli.config.VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Ensures that Jupyter notebook files area always read through this extension
nbsphinx_execute = "auto"

autodoc_member_order = "bysource"

autoyaml_level = 2

# Separates the class and the constructor signature, leading to the html page looking cleaner
autodoc_class_signature = "separated"

# Removes the type hints from the documentation, this makes the documentation legible
autodoc_typehints = "none"

templates_path = ["_templates"]
exclude_patterns = []

# Mapping for outside references.

intersphinx_mapping = {
    "rdkit": ("https://www.rdkit.org/docs/", None),
}

latex_engine = "xelatex"

# latex_elements = {
#     "preamble": r"\usepackage{pmboxdraw}",
# }
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
