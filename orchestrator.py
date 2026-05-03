"""
Legacy wrapper for the v3 orchestrator.

Keep this file as a thin compatibility entrypoint so existing shortcuts can
continue to launch the current pipeline implementation.
"""

from __future__ import annotations

from orchestrator_v3 import main


if __name__ == "__main__":
    main()
