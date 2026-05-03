import tempfile
import unittest
import zipfile
from pathlib import Path
import shutil

from anna_core.agent_runtime import latest_input_file, scout_input_csv


class ScoutArchiveExtractionTests(unittest.TestCase):
    def _make_project(self) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        project_dir = root / "demo_project"
        (project_dir / "input").mkdir(parents=True, exist_ok=True)
        return project_dir

    def _write_zip_with_csv(self, archive_path: Path, csv_name: str = "dataset.csv") -> None:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_name, "id,value\n1,10\n2,20\n")

    def test_latest_input_file_extracts_zip_dataset(self):
        project_dir = self._make_project()
        archive = project_dir / "input" / "sample_dataset.zip"
        self._write_zip_with_csv(archive)

        resolved = latest_input_file(project_dir)

        self.assertTrue(resolved.endswith(".csv"), resolved)
        self.assertTrue(Path(resolved).exists())
        self.assertIn("_extracted", resolved)

    def test_scout_input_csv_is_idempotent_after_extraction(self):
        project_dir = self._make_project()
        archive = project_dir / "input" / "sample_dataset.zip"
        self._write_zip_with_csv(archive, csv_name="nested/data.csv")

        first = scout_input_csv(project_dir)
        second = scout_input_csv(project_dir)

        self.assertEqual(first, second)
        self.assertTrue(Path(first).exists())
        self.assertTrue(first.endswith(".csv"))
        self.assertIn("_extracted", first)


if __name__ == "__main__":
    unittest.main()
