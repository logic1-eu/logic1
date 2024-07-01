pytest:
	pytest -n 5 --exitfirst --doctest-modules\
		--ignore="logic1/theories/depricated"\
		--ignore="logic1/theories/ZModM"

pytest-full:
	pytest --doctest-modules\
		--ignore="logic1/theories/depricated"\
		--ignore="logic1/theories/ZModM"

mypy:
	mypy logic1 --exclude logic1/theories/depricated\
		--exclude logic1/theories/ZModM

test: mypy pytest

pygount:
	pygount -f summary logic1

coverage:
	coverage run -m pytest --doctest-modules\
		--ignore="logic1/theories/depricated"\
		--ignore="logic1/theories/ZModM"

coverage_html: coverage
	coverage html
	open htmlcov/index.html

clean:
	rm -r build dist logic1.egg-info

veryclean:
	rm -rf htmlcov .coverage
