.PHONY: clean lint format
	DIR = src

default: lint format

format:
	ruff check --select I --fix $(DIR)
	ruff format $(DIR)

lint:
	ruff check $(DIR)
	mypy $(DIR)
