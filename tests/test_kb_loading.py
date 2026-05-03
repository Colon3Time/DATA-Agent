import tempfile
import unittest
from pathlib import Path

from anna_core.kb import KnowledgeBase


class KnowledgeBaseLoadingTests(unittest.TestCase):
    def test_load_includes_shared_global_and_agent_kbs(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            (kb_dir / "shared_methods.md").write_text("shared section", encoding="utf-8")
            (kb_dir / "global_world_class_analytics.md").write_text("global section", encoding="utf-8")
            (kb_dir / "anna_methods.md").write_text("anna section", encoding="utf-8")

            kb = KnowledgeBase(kb_dir)
            text = kb.load("anna")

            self.assertIn("shared section", text)
            self.assertIn("global section", text)
            self.assertIn("anna section", text)
            self.assertLess(text.index("shared section"), text.index("global section"))
            self.assertLess(text.index("global section"), text.index("anna section"))

