ESC   := $(shell printf '\033')
BOLD  := $(ESC)[1m
RESET := $(ESC)[0m

EXT_SUFFIX := $(shell python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

CYTHON_MODULES := range  # substitution
CYTHON_BASES   := $(addprefix logic1/theories/RCF/, $(CYTHON_MODULES))
CYTHON_CS      := $(addsuffix .c, $(CYTHON_BASES))
CYTHON_HTMLS   := $(addsuffix .html, $(CYTHON_BASES))
CYTHON_SOS     := $(addsuffix $(EXT_SUFFIX), $(CYTHON_BASES))

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

ign_cython       := --ignore=logic1/theories/RCF/range.pyx
ign_parallel     := --ignore-glob=*parallel*
ign_redlog       := --ignore=logic1/theories/RCF/test_redlog.txt \
				    --ignore=logic1/theories/RCF/test_simplify_motor_redlog.txt \
                    --ignore=logic1/theories/RCF/redlog.py
ign_slow         := --ignore=logic1/theories/RCF/test_simplify_motor.txt \
                    --ignore=logic1/theories/RCF/test_simplify_motor_redlog.txt \
                    --ignore=logic1/theories/RCF/test_qe.txt
ign_redlog_motor := --ignore=logic1/theories/RCF/test_simplify_motor_redlog.txt

ignores := $(ign_redlog_motor)
PYTEST := pytest -n 8 --durations=0 --doctest-cython --exitfirst --doctest-modules

reduce := $(shell echo "quit;" | redcsl -w &>/dev/null; echo $$?)

ifeq ($(reduce), 0)
  $(info Executing Reduce succeeded, will run tests with Redlog)
else
  $(info Executing Reduce failed with exit code $(reduce), will skip tests with Redlog)
  ignores += $(ign_redlog)
endif

.PHONY: cython cython-clean cython-veryclean \
        pytest mypy mypy-run \
        test test-all test-doc \
        doc pygount coverage coverage_html \
        clean veryclean conda-build

test: cython
	$(MAKE) mypy-run
	$(PYTEST) $(ignores)

test-all: test test-doc

mypy: cython
	$(MAKE) mypy-run

mypy-run:
# It seems that --no-incremental is not needed anymore
	mypy --explicit-package-bases stubs
	mypy --exclude '$(exclude_re)' logic1

pytest: cython
	$(PYTEST) $(ignores)

test-doc:
	cd doc && make test

cython: $(CYTHON_SOS)

logic1/theories/RCF/%$(EXT_SUFFIX): logic1/theories/RCF/%.pyx cython-setup.py
	python cython-setup.py build_ext --inplace

cython-clean:
	/bin/rm -f $(CYTHON_CS) $(CYTHON_HTMLS) $(CYTHON_SOS)

cython-veryclean: cython-clean
	/bin/rm -f $(addsuffix .cpython-*-darwin.so, $(CYTHON_BASES))

doc:
	cd doc && make clean html

pygount:
	pygount -f summary logic1

coverage: cython
	coverage run -m pytest --doctest-modules $(ignores)

coverage_html: coverage
	coverage html
	open htmlcov/index.html

clean:
	/bin/rm -rf build dist logic1.egg-info

veryclean: clean cython-veryclean
	/bin/rm -rf htmlcov .coverage

conda-build:
	LOGIC1_GIT_REPO="file:$$(pwd)" \
	LOGIC1_GIT_REV="$$(git rev-parse HEAD)" \
	LOGIC1_VERSION="$$(python -m setuptools_scm)" \
	rattler-build build --recipe conda

# Upload release notes:
# gh release edit v0.2.0 --notes-file releases/v0.2.0.md
