# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DYNATREE'
copyright = '2024, Jan Tippner & DYNATREE team'
author = 'Jan Tippner & DYNATREE team'

import os
import sys

# Přidejte aktuální adresář do sys.path
sys.path.insert(0, os.path.abspath('./dynatree'))
sys.path.insert(0, os.path.abspath('.'))

print(sys.path)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'cloud'
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
#html_theme = 'pydata_sphinx_theme'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
     "sphinx.ext.autosummary", 
     'sphinx.ext.napoleon'
    # Další rozšíření můžete přidat zde...
]

#autosummary_generate = True
#autodoc_default_options = {
#    'members': True,
#}
autodoc_typehints = "description"

