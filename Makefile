install:
	poetry install

black:
	black huddles tests

ruff:
	ruff huddles tests

style: black ruff