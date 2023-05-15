# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import molli.chem.atom
import molli.chem.bond
import molli.chem.ensemble
import molli.chem.fragment 
import molli.chem.geometry
import molli.chem.molecule
import molli.chem.structure

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Molli'
copyright = '2022, Alexander Shved'
author = 'Alexander Shved'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'myst_parser', "nbsphinx"]

#Ensures that Jupyter notebook files area always read through this extension
nbsphinx_execute = 'always'

autodoc_member_order = 'bysource'

# Separates the class and the constructor signature, leading to the html page looking cleaner
autodoc_class_signature = "separated"

source_suffix = {".rst": "restructuredtext",
                 ".md": "markdown",}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
