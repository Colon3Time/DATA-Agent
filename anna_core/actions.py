from __future__ import annotations

from pathlib import Path


class WorkspacePaths:
    """Resolve Anna action paths under a known workspace root."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()

    def resolve(self, raw_path: str) -> Path:
        raw = Path(raw_path)
        candidate = raw if raw.is_absolute() else self.base_dir / raw_path
        return candidate.resolve()

    def resolve_for_delete(self, raw_path: str) -> Path:
        path = self.resolve(raw_path)
        if not self.is_within_base(path):
            raise PermissionError(f"Refuse to delete outside workspace: {path}")
        return path

    def is_within_base(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.base_dir)
            return True
        except ValueError:
            return False

