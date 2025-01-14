# ignores = --ignore=logic1/theories/RCF/test_simplify_motor_redlog.txt --ignore-glob=*parallel*
ignores = --ignore=logic1/theories/RCF/test_simplify_motor_redlog.txt

.PHONY: pytest pytest-full test-doc mypy test test-all doc pygount\
		coverage coverage_html clean veryclean conda-build

cython:
	python cython-setup.py build_ext --inplace

cython-clean:
	/bin/rm logic1/theories/RCF/substitution.c logic1/theories/RCF/substitution.cpython-311-darwin.so

pytest:
	pytest -n 8 --doctest-cython --exitfirst --doctest-modules $(ignores)

pytest-fast:
	PYTHONOPTIMIZE=TRUE pytest -n 8 --disable-warnings --exitfirst --doctest-modules $(ignores)

pytest-seq:
	pytest --doctest-cython --exitfirst --doctest-modules $(ignores)

pytest-full:
	pytest -n 8 --doctest-modules $(ignores)

pytest-full-seq:
	pytest --doctest-modules $(ignores)

test-doc:
	cd doc && make test

mypy:
	mypy --explicit-package-bases stubs
	mypy logic1

test: mypy pytest

test-all: test test-doc

doc:
	cd doc && make clean html

pygount:
	pygount -f summary logic1

coverage:
	coverage run -m pytest --doctest-modules

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
