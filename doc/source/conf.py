# mypy: ignore_errors

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Logic1'
copyright = '2023, Thomas Sturm'
author = '<a: href="https://science.thomas-sturm.de">Thomas Sturm</a>'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    # 'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_togglebutton'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

doctest_path = ['~/Documents/Dynamic/src/python/Logic1/logic1/logic1']

autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True
}

intersphinx_mapping = {
    'sympy': ('https://docs.sympy.org/latest', None),
    'python': ('https://docs.python.org/3', None)
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# _extra_footer = """
# """

html_context = {
    # 'default_mode': 'auto'
}

html_css_files = [
    "custom.css"
]

html_last_updated_fmt = ''

html_logo = None

# html_sidebars = {
#     '**': ['sidebar-logo.html',
#            'search-field.html',
#            'sbt-sidebar-nav.html']
# }

html_static_path = ['_static']

html_theme = 'sphinx_book_theme'

html_theme_options = {
    'home_page_in_toc': True,
    'repository_url': 'https://github.com/thomas-sturm/logic1',
    'show_navbar_depth': 1,  # default is 1
    'show_toc_level': 1,  # default is 1
    'use_repository_button': True
}

html_title = 'Logic1'

# pygments_style = 'tango'
