# cspell:disable

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

POLYLIB_TARGETS := pytest pytest-fast pytest-seq pytest-full pytest-full-seq mypy test-all coverage coverage_html

ifneq ($(filter $(GOALS), $(POLYLIB_TARGETS)),)
  POLYLIB := $(shell PYTHONPATH=. python -c 'from logic1.theories.RCF.term import POLYLIB; print(POLYLIB)')
  $(info Determined POLYLIB == $(BOLD)"$(POLYLIB)"$(RESET) via Python import)

  ifeq ($(POLYLIB), FLINT)
    ign_other_backend := --ignore=logic1/theories/RCF/term/term_sage.py
    exclude_re := logic1/theories/RCF/term/term_sage\.py

  else ifeq ($(POLYLIB), SAGE)
    ign_other_backend := --ignore=logic1/theories/RCF/term/term_flint.py \
                         --ignore=logic1/theories/RCF/test_term_flint.txt
    exclude_re := logic1/theories/RCF/term/term_flint\.py|logic1/theories/RCF/test_term_flint\.txt

  else
    $(error Could not determine POLYLIB via Python import)
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

ignores := $(ign_other_backend) $(ign_redlog_motor)

.PHONY: cython cython-clean cython-html \
        pytest pytest-fast pytest-seq pytest-full pytest-full-seq \
        test-doc mypy mypy_noinc test test-all doc pygount \
        coverage coverage_html clean veryclean conda-build

cython: $(CYTHON_SOS)

logic1/theories/RCF/%$(EXT_SUFFIX): logic1/theories/RCF/%.pyx cython-setup.py
	python cython-setup.py build_ext --inplace

cython-clean:
	/bin/rm -f $(CYTHON_CS) $(CYTHON_HTMLS) $(CYTHON_SOS)

cython-veryclean: cython-clean
	/bin/rm -f $(addsuffix .cpython-*-darwin.so, $(CYTHON_BASES))

cython-html:
	cd logic1/theories/RCF && open range.html

pytest: cython
	pytest -n 8 --durations=0 --doctest-cython --exitfirst --doctest-modules $(ignores)

pytest-fast: cython
	PYTHONOPTIMIZE=TRUE pytest -n 8 --disable-warnings --exitfirst --doctest-modules $(ignores)

pytest-seq: cython
	pytest  --durations=0 --doctest-cython --exitfirst --doctest-modules $(ignores)

pytest-full: cython
	pytest -n 8 --doctest-modules $(ignores)

pytest-full-seq: cython
	pytest --durations=0 --doctest-modules $(ignores)

test-doc:
	cd doc && make test

mypy:
	mypy --explicit-package-bases stubs
	mypy --exclude '$(exclude_re)' logic1

# It seems that --no-incremental is not needed anymore
mypy_noinc:
	mypy --no-incremental --explicit-package-bases stubs
	mypy --no-incremental --exclude '$(exclude_re)' logic1

test: cython
	$(MAKE) mypy
	$(MAKE) pytest

test-all: test test-doc

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
	rm -r build dist logic1.egg-info

veryclean:
	rm -rf htmlcov .coverage

conda-build:
	LOGIC1_GIT_REPO="file:$$(pwd)" \
	LOGIC1_GIT_REV="$$(git rev-parse HEAD)" \
	LOGIC1_VERSION="$$(python -m setuptools_scm)" \
	rattler-build build --recipe conda
