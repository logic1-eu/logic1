pytest:
	pytest -n 5 --exitfirst --doctest-modules

pytest-full:
	pytest -n 5 --doctest-modules

mypy:
	mypy logic1

test: mypy pytest

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
