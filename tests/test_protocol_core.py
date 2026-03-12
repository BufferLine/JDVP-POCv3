from __future__ import annotations

import unittest

from src.protocol_core.dv_ordinal import build_dv
from src.protocol_core.trajectory import build_trajectory, validate_continuity


class ProtocolCoreTests(unittest.TestCase):
    def test_ordinal_dv_derivation(self) -> None:
        before = {
            "interaction_id": "session-1",
            "turn_number": 1,
            "judgment_holder": "Human",
            "delegation_awareness": "Explicit",
            "cognitive_engagement": "Active",
            "information_seeking": "Active",
            "context_module": "general",
        }
        after = {
            "interaction_id": "session-1",
            "turn_number": 2,
            "judgment_holder": "Shared",
            "delegation_awareness": "Implicit",
            "cognitive_engagement": "Reactive",
            "information_seeking": "Passive",
            "context_module": "general",
        }
        dv = build_dv(before, after).to_dict()
        self.assertEqual(dv["delta_judgment_holder"], 0.5)
        self.assertEqual(dv["delta_delegation_awareness"], 0.5)
        self.assertEqual(dv["delta_cognitive_engagement"], 0.5)
        self.assertEqual(dv["delta_information_seeking"], 0.5)

    def test_undefined_judgment_holder_maps_to_null(self) -> None:
        before = {
            "interaction_id": "session-1",
            "turn_number": 1,
            "judgment_holder": "Undefined",
            "delegation_awareness": "Explicit",
            "cognitive_engagement": "Active",
            "information_seeking": "Active",
            "context_module": "general",
        }
        after = {
            "interaction_id": "session-1",
            "turn_number": 2,
            "judgment_holder": "AI",
            "delegation_awareness": "Implicit",
            "cognitive_engagement": "Reactive",
            "information_seeking": "Passive",
            "context_module": "general",
        }
        dv = build_dv(before, after).to_dict()
        self.assertIsNone(dv["delta_judgment_holder"])

    def test_trajectory_continuity_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            validate_continuity(
                [
                    {"interaction_id": "session-1", "from_turn": 0, "to_turn": 1},
                    {"interaction_id": "session-1", "from_turn": 2, "to_turn": 3},
                ],
                "session-1",
            )

    def test_trajectory_builds_when_continuous(self) -> None:
        trajectory = build_trajectory(
            "session-1",
            [
                {"interaction_id": "session-1", "from_turn": 0, "to_turn": 1},
                {"interaction_id": "session-1", "from_turn": 1, "to_turn": 2},
            ],
        ).to_dict()
        self.assertEqual(trajectory["interaction_id"], "session-1")
        self.assertEqual(len(trajectory["vectors"]), 2)


if __name__ == "__main__":
    unittest.main()
