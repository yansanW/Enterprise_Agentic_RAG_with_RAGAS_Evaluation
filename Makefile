# Project Makefile

.PHONY: lint typecheck test ci

lint:
	ruff check src/

typecheck:
	mypy src/

test:
	pytest -v

ci: lint typecheck test
