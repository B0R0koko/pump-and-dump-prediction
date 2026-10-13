set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# The IEEE Access class and Type 1 fonts live in paper/styles, while figures
# live in paper/src/images. The build runs from paper/src.
tex_env := 'TEXINPUTS="../styles//:./images//:" TEXFONTS="../styles//:" T1FONTS="../styles//:"'

format-all:
    poetry run python -m black src tests

pylint:
    poetry run python -m pylint src/pumps_and_dumps src/installer tests

mypy:
    poetry run python -m mypy src/pumps_and_dumps src/installer tests

test:
    poetry run pytest

prepare-data:
    ./scripts/install.sh

paper:
    cd paper/src && {{tex_env}} pdflatex -interaction=nonstopmode access.tex && {{tex_env}} pdflatex -interaction=nonstopmode access.tex

paper-highlighted:
    cd paper/src && {{tex_env}} pdflatex -interaction=nonstopmode access_highlighted.tex && {{tex_env}} pdflatex -interaction=nonstopmode access_highlighted.tex

presentation:
    cd presentation && lualatex -interaction=nonstopmode main.tex && lualatex -interaction=nonstopmode main.tex
