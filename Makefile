test:
	python -m unittest discover --verbose .

mypy:
	python -m mypy src tests