"""Execute definition-only notebook cells so tests cover the actual analysis source."""

import json
from pathlib import Path
import sys
from types import ModuleType


def load_analysis():
    module = ModuleType("_notebook_analysis")
    sys.modules[module.__name__] = module
    path = Path(__file__).resolve().parents[1] / "notebooks/analysis_helpers.ipynb"
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec"), module.__dict__)
    return module


analysis = load_analysis()
