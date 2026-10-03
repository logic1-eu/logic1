ESC   := $(shell printf '\033')
BOLD  := $(ESC)[1m
RESET := $(ESC)[0m

EXT_SUFFIX := $(shell python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# List active .pyx modules as whitespace-separated paths without the suffix.
CYTHON_BASES   := logic1/theories/RCF/range
# CYTHON_BASES += logic1/theories/RCF/substitution
CYTHON_CS      := $(addsuffix .c, $(CYTHON_BASES))
CYTHON_HTMLS   := $(addsuffix .html, $(CYTHON_BASES))
CYTHON_SOS     := $(addsuffix $(EXT_SUFFIX), $(CYTHON_BASES))
CYTHON_SO_GLOBS := $(addsuffix .so, $(CYTHON_BASES)) $(addsuffix .*.so, $(CYTHON_BASES))

.DEFAULT_GOAL := test
GOALS := $(if $(MAKECMDGOALS), $(MAKECMDGOALS), $(.DEFAULT_GOAL))

POLYLIB_TARGETS := mypy-run

ifneq ($(filter $(GOALS), $(POLYLIB_TARGETS)),)
  POLYLIB := $(shell PYTHONPATH=. python -c 'from logic1.theories.RCF.term import POLYLIB; print(POLYLIB)')
  $(info Determined POLYLIB == $(BOLD)"$(POLYLIB)"$(RESET) via Python import)

  ifeq ($(POLYLIB), FLINT)
    exclude_re := logic1/theories/RCF/term/term_sage\.py

  else ifeq ($(POLYLIB), SAGE)
    exclude_re := logic1/theories/RCF/term/term_flint\.py

  else
    $(error Could not determine valid POLYLIB)
  endif
endif

ign_redlog := --ignore=logic1/theories/RCF/test_redlog.txt \
              --ignore=logic1/theories/RCF/redlog.py

ignores :=
PYTEST := pytest
PYTEST_OPTIONS := -n 8 --durations=10 --doctest-cython --exitfirst --doctest-modules

REDLOG_TARGETS := test test-all pytest coverage coverage_html

ifneq ($(filter $(GOALS), $(REDLOG_TARGETS)),)
reduce := $(shell echo "quit;" | redcsl -w &>/dev/null; echo $$?)

ifeq ($(reduce), 0)
  $(info Executing Reduce succeeded, will run tests with Redlog)
else
  $(info Executing Reduce failed with exit code $(reduce), will skip tests with Redlog)
  ignores += $(ign_redlog)
endif
endif

.PHONY: cython cython-debug \
        pytest mypy mypy-run \
        test test-all test-doc \
        doc pygount coverage coverage_html \
        mostlyclean clean conda-build

test: cython
	$(MAKE) mypy-run
	$(PYTEST) $(PYTEST_OPTIONS) $(ignores)

test-all: test test-doc

mypy: cython
	$(MAKE) mypy-run

mypy-run:
# It seems that --no-incremental is not needed anymore
	mypy --explicit-package-bases stubs
	mypy --exclude '$(exclude_re)' logic1

pytest: cython
	$(PYTEST) $(PYTEST_OPTIONS) $(ignores)

test-doc: cython
	cd doc && $(MAKE) test

# Local Cython builds take their .pyx files from CYTHON_BASES.
# cythonize --inplace builds the .so via setuptools, which also reads
# pyproject.toml; cython-debug only generates .c and annotated .html.
# Conda instead builds the extension declared in pyproject.toml, using
# the gmpy2 include path supplied by conda/recipe.yaml.
cython: $(CYTHON_SOS)

# In logic1_dev, Python's LDSHARED has the lib rpath twice; Conda's compiler
# activation adds it again via LDFLAGS, causing two harmless linker warnings.
logic1/%$(EXT_SUFFIX): logic1/%.pyx
	cythonize --inplace $<

cython-debug:
	cythonize --annotate --force $(addsuffix .pyx,$(CYTHON_BASES))

doc: cython
	cd doc && $(MAKE) clean
	cd doc && $(MAKE) html

pygount:
	pygount -f summary logic1

# Cython doctests run, but compiled range.pyx lines are not measured until
# the extension is rebuilt with line tracing enabled.
coverage: cython
	$(PYTEST) $(PYTEST_OPTIONS) --cov=logic1 --cov-report= $(ignores)

coverage_html: coverage
	coverage html
	open htmlcov/index.html

# The retained .so keeps make cython from regenerating C;
# use make cython-debug to regenerate C and annotated HTML.
mostlyclean:
	/bin/rm -rf build doc/build/doctrees
	/bin/rm -f $(CYTHON_CS) $(CYTHON_HTMLS)

# output/ contains Conda artifacts from make conda-build and is removed here.
# dist/ contains PyPI artifacts from python -m build (outside this Makefile)
# and is deliberately preserved.
clean: mostlyclean
	/bin/rm -f $(CYTHON_SO_GLOBS)
	/bin/rm -rf htmlcov doc/build .pytest_cache .mypy_cache output
	/bin/rm -f .coverage .coverage.*
	find . -name .git -prune -o -type d -name __pycache__ -prune -exec /bin/rm -rf {} +

conda-build:
	LOGIC1_GIT_REPO="file:$$(pwd)" \
	LOGIC1_GIT_REV="$$(git rev-parse HEAD)" \
	LOGIC1_VERSION="$$(python -m setuptools_scm)" \
	rattler-build build --recipe conda

# Upload release notes w/o creating a new release
# gh release edit v0.2.0 --notes-file releases/v0.2.0.md

# Create new release
# gh release create v0.3.1 --title "0.3.1"--notes-file releases/v0.3.1.md

# Get the SHA-256 checksum of the release tarball
# curl -Ls https://github.com/logic1-eu/logic1/archive/refs/tags/v0.3.1.tar.gz | shasum -a 256
