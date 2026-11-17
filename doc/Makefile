# Documentation builds with Sphinx and tests with pytest.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -D graphviz_dot=dot
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.DEFAULT_GOAL := help

.PHONY: all test pytest doctest help Makefile

all: test html

test: pytest

pytest:
	pytest -n 2 --exitfirst source

doctest:
	@echo "make doctest is not supported; documentation tests use pytest." >&2
	@echo "Run 'make test' here, or 'make test-doc' from the repository root." >&2
	@exit 1

help:
	@printf '%s\n' \
		'Usage: make TARGET' \
		'' \
		'  help       Show this help (default)' \
		'  all        Run documentation tests and build HTML' \
		'  test       Run documentation tests with pytest' \
		'  pytest     Alias for test' \
		'  doctest    Unsupported; use test or pytest instead' \
		'  html       Build standalone HTML documentation' \
		'  linkcheck  Check external links' \
		'  clean      Remove the Sphinx build directory' \
		'' \
		'Other Sphinx builders, such as dirhtml, epub and latexpdf, are also available.' \
		'From the repository root, use make test-doc to run documentation tests.'

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
