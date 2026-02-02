"""Cleaning temporary files after inference."""
from pathlib import Path
import tempfile
import os
def cleanup_temp(pattern="tmp*.dcm"):
    tmpdir = Path(tempfile.gettempdir())
    for f in tmpdir.glob(pattern):
        try: f.unlink()
        except: pass
def safe_remove(path: str):
    try: Path(path).unlink(missing_ok=True)
    except: pass
    return True
