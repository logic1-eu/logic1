# mypy: ignore_errors

import html
import os
import subprocess

from docutils import nodes
import setuptools_scm


def redlog_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    url = f"https://www.redlog.eu/documentation/service.php?key={text}"
    node = nodes.reference(
        rawtext,
        "",
        nodes.literal(text, f"{text}()"),
        refuri=url,
        **(options or {}),
    )
    return [node], []


def setup(app):
    app.add_role("redlog", redlog_role)


def subprocess_output(*args: str) -> str | None:
    """Return text output from a subprocess, or ``None``."""
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


git_commit = subprocess_output('git', 'rev-parse', '--short=10', 'HEAD')


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'Logic1'
repository_url = 'https://github.com/logic1-eu/logic1'
copyright = '2023–%Y, N. Faroß and T. Sturm'
author = '<a href="https://www.chalmers.se/en/persons/faross/">N. Faroß</a>, <a href="https://science.thomas-sturm.de/">T. Sturm</a>'
version = setuptools_scm.get_version(root='../..', relative_to=__file__)
release = version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

autodoc_class_signature = 'separated'

autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True,
}

extra_footer = (
    None if version is None else
        f'Corresponds to version <code>{html.escape(version)}</code>.' if git_commit is None else
            f'Corresponds to version <a href="{repository_url}/tree/{git_commit}"><code>{html.escape(version)}</code></a>.'
)

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

nitpicky = True

python_use_unqualified_type_names = True

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    'default_mode': 'auto'
}

html_css_files = [
    "custom.css"
]

html_js_files = ["sidebar-toggle.js"]

html_last_updated_fmt = ''

html_static_path = ['_static']

html_theme = 'sphinx_book_theme'

html_theme_options = {
    'extra_footer': extra_footer,
    'footer_content_items': [
        'extra-footer',
        'last-updated',
        'copyright',
    ],
    'home_page_in_toc': True,
    'max_navbar_depth': 12,
    'repository_url': repository_url,
    'show_navbar_depth': 12,
    'use_repository_button': True
}

html_title = 'Logic1'
