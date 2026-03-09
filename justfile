set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

format-all:
    poetry run python -m black core features market_data preprocessing backtest test

pylint:
    poetry run python -m pylint core features market_data preprocessing backtest test

mypy:
    poetry run python -m mypy core features market_data preprocessing backtest test
