all_tests: unittest doctest mypy


unittest:
	python -m unittest discover --verbose .

doctest:
	python -m pytest --doctest-modules --verbose src/cone

mypy:
	python -m mypy src/cone tests

doc:
	pdoc3 --html --force src/cone
