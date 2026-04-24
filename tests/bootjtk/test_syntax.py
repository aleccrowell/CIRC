"""Verify that every .py file in the project parses as valid Python 3 syntax."""
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_SKIP_DIRS = {".venv", "__pycache__", ".git", "build", "dist"}

_PY_FILES = sorted(
    p for p in ROOT.rglob("*.py")
    if not _SKIP_DIRS.intersection(p.parts)
)


@pytest.mark.parametrize("py_file", _PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_py3_syntax(py_file):
    """Every .py file must parse as valid Python 3 syntax."""
    py_compile.compile(str(py_file), doraise=True)
