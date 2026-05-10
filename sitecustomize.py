from __future__ import annotations

import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path


_TEMP_ROOT = Path.cwd() / "tmp" / "_python_temp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(_TEMP_ROOT, stat.S_IRWXU)
except OSError:
    pass

os.environ["TMP"] = str(_TEMP_ROOT)
os.environ["TEMP"] = str(_TEMP_ROOT)
os.environ["TMPDIR"] = str(_TEMP_ROOT)
tempfile.tempdir = str(_TEMP_ROOT)


def _unique_temp_path(prefix: str = "tmp", suffix: str = "", dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(dir) if dir is not None else _TEMP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(1000):
        candidate = root / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir()
            try:
                os.chmod(candidate, stat.S_IRWXU)
            except OSError:
                pass
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate temporary directory")


def _patched_mkdtemp(suffix: str = "", prefix: str = "tmp", dir: str | os.PathLike[str] | None = None):
    return str(_unique_temp_path(prefix=prefix, suffix=suffix, dir=dir))


class _PatchedTemporaryDirectory:
    def __init__(self, suffix: str = "", prefix: str = "tmp", dir: str | os.PathLike[str] | None = None):
        self.name = str(_unique_temp_path(prefix=prefix, suffix=suffix, dir=dir))

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=True)


tempfile.mkdtemp = _patched_mkdtemp
tempfile.TemporaryDirectory = _PatchedTemporaryDirectory
