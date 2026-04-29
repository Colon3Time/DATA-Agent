from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def load_app_env(*extra_paths: str | Path) -> None:
    """Load environment variables from fallback paths, then the project .env.

    Fallback values only fill missing/empty variables. The project .env is
    applied last and can override fallback values.
    """
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [*extra_paths, base_dir / ".env"]
    for env_path in candidates:
        values = dotenv_values(env_path)
        for key, value in values.items():
            if value is None:
                continue
            current = os.environ.get(key)
            if env_path == base_dir / ".env" or not current:
                os.environ[key] = value
