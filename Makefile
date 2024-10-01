.PHONY: pytest pytest-full test-doc mypy test test-all doc pygount\
		coverage coverage_html clean veryclean

pytest:
	pytest -n 8 --exitfirst --doctest-modules

pytest-seq:
	pytest --exitfirst --doctest-modules

pytest-full:
	pytest -n 8 --doctest-modules

test-doc:
	cd doc && make test

mypy:
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
