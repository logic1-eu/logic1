wheel:
	python setup.py bdist_wheel

pytest:
	pytest --doctest-modules

mypy:
	mypy logic1

test: mypy pytest

coverage:
	coverage run -m pytest --doctest-modules

coverage_html: coverage
	coverage html
	open htmlcov/index.html

clean:
	rm -r build dist logic1.egg-info

veryclean:
	rm -r htmlcov
