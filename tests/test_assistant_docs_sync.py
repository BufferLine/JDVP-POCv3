from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AssistantDocsSyncTests(unittest.TestCase):
    def test_assistant_docs_are_synchronized(self) -> None:
        result = subprocess.run(
            ["python3", "scripts/check_assistant_docs_sync.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("synchronized", result.stdout)
