# mypy: ignore_errors

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Logic1'
copyright = '2023-%Y, N. Faroß and T. Sturm'
author = '<a href="https://www.chalmers.se/en/persons/faross/">N. Faroß</a>, <a href="https://science.thomas-sturm.de/">T. Sturm</a>'
release = '0.1'
version = '0.1'

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = ['**/atomlib.rst']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# doctest_path = ['../logic1']

autodoc_class_signature = 'separated'

autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True,
}

autodoc_type_aliases = {}

# _extra_footer = ''

intersphinx_mapping = {
    'gmpy2': ('https://gmpy2.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'sage': ('https://doc.sagemath.org/html/en/reference/', None),
    'sage-polynomial-rings': ('https://doc.sagemath.org/html/en/reference/polynomial_rings/', None),
    'sage-rings-standard': ('https://doc.sagemath.org/html/en/reference/rings_standard/', None),
    # Further Sage intersphinx inventories can be derived from the subdirectory
    # names of the Sage documentation at https://doc.sagemath.org/html/en/reference/
    'sympy': ('https://docs.sympy.org/latest', None),
}

language = 'en'

nitpicky = True

# pygments_style = 'tango'

python_use_unqualified_type_names = True

templates_path = ['_templates']

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    # 'default_mode': 'auto'
}

html_css_files = [
    "custom.css"
]

html_js_files = ["sidebar-toggle.js"]

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
    'collapse_navbar': False,
    'home_page_in_toc': True,
    'max_navbar_depth': 12,
    'repository_url': 'https://github.com/logic1-eu/logic1',
    'show_navbar_depth': 12,  # default is 1
    'show_toc_level': 1,  # default is 1
    'use_repository_button': True
}

html_title = 'Logic1'