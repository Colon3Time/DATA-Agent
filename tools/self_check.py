from __future__ import annotations

import py_compile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compile_sources() -> bool:
    files = [ROOT / "orchestrator_v3.py"]
    files.extend(sorted((ROOT / "anna_core").glob("*.py")))
    files.extend(sorted((ROOT / "tests").glob("test_*.py")))

    ok = True
    print("[check] py_compile")
    for file_path in files:
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  OK  {file_path.relative_to(ROOT)}")
        except py_compile.PyCompileError as exc:
            ok = False
            print(f"  ERR {file_path.relative_to(ROOT)}")
            print(exc.msg)
    return ok


def run_unittests() -> bool:
    print("[check] unittest")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main() -> int:
    compile_ok = compile_sources()
    tests_ok = run_unittests()
    if compile_ok and tests_ok:
        print("[check] PASS")
        return 0
    print("[check] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
