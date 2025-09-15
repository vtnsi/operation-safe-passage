import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../src/operation_safe_passage"))

project = 'operation_safe_passage'
copyright = '2025, Sami Saliba, Michael "Alex" Kyer, Dan DeCollo, Dan Sobien, Stephen Adams'
author = 'Sami Saliba, Michael "Alex" Kyer, Dan DeCollo, Dan Sobien, Stephen Adams'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 
    'sphinx.ext.autodoc', 
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    "nbsphinx"
] 

nbsphinx_allow_errors = True

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

