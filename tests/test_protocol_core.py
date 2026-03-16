from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.protocol_core.dv_ordinal import build_dv
from src.protocol_core.schema_sync import (
    build_snapshot_manifest,
    compare_schema_roots,
    refresh_schema_snapshot,
    validate_snapshot_manifest,
)
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

    def test_schema_sync_detects_matching_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_root = Path(temp_dir) / "upstream"
            vendored_root = Path(temp_dir) / "vendored"
            upstream_root.mkdir()
            vendored_root.mkdir()

            payload = {"$schema": "https://json-schema.org/draft/2020-12/schema"}
            for filename in ("jsv-schema.json", "dv-schema.json", "trajectory-schema.json"):
                for root in (upstream_root, vendored_root):
                    with (root / filename).open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle)

            self.assertEqual(compare_schema_roots(upstream_root, vendored_root), [])

    def test_schema_sync_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_root = Path(temp_dir) / "upstream"
            vendored_root = Path(temp_dir) / "vendored"
            upstream_root.mkdir()
            vendored_root.mkdir()

            for filename in ("jsv-schema.json", "dv-schema.json", "trajectory-schema.json"):
                with (upstream_root / filename).open("w", encoding="utf-8") as handle:
                    json.dump({"title": f"upstream-{filename}"}, handle)
                with (vendored_root / filename).open("w", encoding="utf-8") as handle:
                    json.dump({"title": f"vendored-{filename}"}, handle)

            diffs = compare_schema_roots(upstream_root, vendored_root)
            self.assertEqual(len(diffs), 3)
            self.assertEqual({diff.filename for diff in diffs}, {
                "jsv-schema.json",
                "dv-schema.json",
                "trajectory-schema.json",
            })

    def test_schema_snapshot_refresh_copies_upstream_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_root = Path(temp_dir) / "upstream"
            vendored_root = Path(temp_dir) / "vendored"
            upstream_root.mkdir()

            for filename in ("jsv-schema.json", "dv-schema.json", "trajectory-schema.json"):
                with (upstream_root / filename).open("w", encoding="utf-8") as handle:
                    json.dump({"title": f"upstream-{filename}"}, handle)

            refresh_schema_snapshot(upstream_root, vendored_root)

            self.assertEqual(compare_schema_roots(upstream_root, vendored_root), [])
            validate_snapshot_manifest(vendored_root)

    def test_schema_snapshot_manifest_uses_vendored_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_root = Path(temp_dir) / "upstream"
            vendored_root = Path(temp_dir) / "vendored"
            upstream_root.mkdir()
            vendored_root.mkdir()

            payloads = {
                "jsv-schema.json": {"title": "jsv"},
                "dv-schema.json": {"title": "dv"},
                "trajectory-schema.json": {"title": "trajectory"},
            }
            for filename, payload in payloads.items():
                with (upstream_root / filename).open("w", encoding="utf-8") as handle:
                    json.dump({"upstream": payload["title"]}, handle)
                with (vendored_root / filename).open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            manifest = build_snapshot_manifest(
                upstream_root=upstream_root,
                vendored_root=vendored_root,
                upstream_revision="abc123",
            )
            self.assertEqual(manifest.upstream_revision, "abc123")
            self.assertEqual(set(manifest.files.keys()), {
                "jsv-schema.json",
                "dv-schema.json",
                "trajectory-schema.json",
            })


if __name__ == "__main__":
    unittest.main()
