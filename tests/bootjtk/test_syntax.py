"""
Guards against Python 2 syntax and built-ins that break silently or
loudly in Python 3.  These tests would have caught the issues found in
the py2→py3 migration.

Two complementary approaches:
  1. py_compile  — catches outright SyntaxErrors in .py files (e.g. bare
                   ``print`` statements).
  2. Regex scan  — catches Python 2 built-ins/methods that are runtime
                   errors in Python 3 (e.g. xrange, .iteritems()).
"""
import py_compile
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_SKIP_DIRS = {".venv", "__pycache__", ".git", "build", "dist"}


def _collect(suffix, extra_skip=frozenset()):
    return sorted(
        p for p in ROOT.rglob(f"*{suffix}")
        if not (_SKIP_DIRS | extra_skip).intersection(p.parts)
    )


_PY_FILES        = _collect(".py")                        # all .py files (syntax check)
_SOURCE_PY_FILES = _collect(".py", extra_skip={"tests"})  # non-test .py (runtime scan)

# Python 2 built-ins / dict methods that raise NameError or AttributeError
# at runtime in Python 3.  Print statement is a SyntaxError caught by
# test_py3_syntax, so it is not repeated here.
_RUNTIME_PATTERNS = [
    (r"\bxrange\s*\(",     "xrange() was removed in Python 3; use range()"),
    (r"\.has_key\s*\(",    ".has_key() was removed; use `in`"),
    (r"\.iteritems\s*\(",  ".iteritems() was removed; use .items()"),
    (r"\.itervalues\s*\(", ".itervalues() was removed; use .values()"),
    (r"\.iterkeys\s*\(",   ".iterkeys() was removed; use .keys()"),
    (r"\braw_input\s*\(",  "raw_input() was removed; use input()"),
    (r"\bbasestring\b",    "basestring does not exist in Python 3"),
    (r"\blong\s*\(",       "long() was removed; use int()"),
]


def _scan(path, patterns):
    """Return a list of violation strings for any matched patterns."""
    violations = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern, reason in patterns:
            if re.search(pattern, line):
                violations.append(
                    f"  line {lineno}: {reason}\n    {line.rstrip()}"
                )
    return violations


@pytest.mark.parametrize("py_file", _PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_py3_syntax(py_file):
    """Every .py file must parse as valid Python 3 syntax.

    Catches bare ``print x`` statements, which are SyntaxErrors in Python 3
    but valid Python 2.  py_compile only checks syntax, so import errors for
    missing dependencies do not cause false failures.
    """
    py_compile.compile(str(py_file), doraise=True)


@pytest.mark.parametrize(
    "src_file",
    _SOURCE_PY_FILES,
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_no_python2_builtins(src_file):
    """Source files must not use Python 2-only built-ins or dict methods.

    These calls are syntactically valid Python 3 but raise NameError or
    AttributeError at runtime, so py_compile would not catch them.
    """
    violations = _scan(src_file, _RUNTIME_PATTERNS)
    if violations:
        pytest.fail(
            f"{src_file.relative_to(ROOT)}:\n" + "\n".join(violations)
        )
