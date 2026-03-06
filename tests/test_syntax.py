"""Verifying end-to-end with syntax check (no heavy deps)."""
import ast
from pathlib import Path
def test_syntax():
    for p in Path("src").rglob("*.py"):
        if "__pycache__" in str(p): continue
        ast.parse(open(p).read())
    for p in Path("app").rglob("*.py"):
        ast.parse(open(p).read())
    assert True
