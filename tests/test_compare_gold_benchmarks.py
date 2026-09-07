from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_gold_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("compare_gold_benchmarks", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_gold_benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_gold_benchmarks)


class CompareGoldBenchmarksTests(unittest.TestCase):
    def test_accuracy_normalizes_legacy_gold_categories_before_matching(self) -> None:
        gold = {
            "interaction-1": [{
                "turn_number": 1,
                "gold_label": {
                    "judgment_holder": "AI",
                    "delegation_awareness": "Implicit",
                    "cognitive_engagement": "Reactive",
                    "information_seeking": "Passive",
                },
                "contested": False,
            }]
        }
        extracts = {
            "interaction-1": [{
                "turn_number": 1,
                "jsv_hint": {
                    "judgment_holder": 9,
                    "delegation_awareness": 5,
                    "cognitive_engagement": 5,
                    "information_seeking": 5,
                },
            }]
        }

        result = compare_gold_benchmarks.compute_accuracy(gold, extracts)

        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertEqual(result["overall_total"], 4)

    def test_accuracy_counts_missing_prediction_as_incorrect(self) -> None:
        gold = {
            "interaction-1": [{
                "turn_number": 1,
                "gold_label": {
                    "judgment_holder": 9,
                    "delegation_awareness": 5,
                    "cognitive_engagement": 5,
                    "information_seeking": 5,
                },
                "contested": False,
            }]
        }
        extracts = {"interaction-1": [{"turn_number": 1, "jsv_hint": {}}]}

        result = compare_gold_benchmarks.compute_accuracy(gold, extracts)

        self.assertEqual(result["overall_accuracy"], 0.0)
        self.assertEqual(result["overall_total"], 4)
