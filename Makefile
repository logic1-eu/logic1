wheel:
	python setup.py bdist_wheel

pytest:
	pytest --doctest-modules\
		--ignore="logic1/theories/depricated"\
		--ignore-glob="logic1/theories/Sets/*.txt"

mypy:
	mypy logic1 --exclude logic1/theories/depricated

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
