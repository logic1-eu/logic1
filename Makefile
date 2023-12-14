wheel:
	python setup.py bdist_wheel

pytest:
	pytest --doctest-modules\
		--ignore="logic1/theories/depricated"

mypy:
	mypy logic1 --exclude logic1/theories/depricated

test: mypy pytest

coverage:
	coverage run -m pytest --doctest-modules\
		--ignore="logic1/theories/depricated"

coverage_html: coverage
	coverage html
	open htmlcov/index.html

clean:
	rm -r build dist logic1.egg-info

veryclean:
	rm -rf htmlcov .coverage

pygount:
	pygount logic1 --names-to-skip=*.txt --format=summary
