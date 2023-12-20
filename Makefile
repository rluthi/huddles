install:
	poetry install

black:
	black huddler tests

ruff:
	ruff huddler tests

style: black ruff