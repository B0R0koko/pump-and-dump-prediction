set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

format-all:
    poetry run python -m black src test

pylint:
    poetry run python -m pylint src test

mypy:
    poetry run python -m mypy src test
