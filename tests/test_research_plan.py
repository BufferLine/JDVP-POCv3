from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from src.research.plan import build_frozen_plan, freeze_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan(tmp_path):
    folder = tmp_path / "demo"
    shutil.copytree(ROOT / "config/research/demo", folder)
    return folder / "plan.json"


def change(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_freeze_stable_portable_and_preserves_original_bytes(plan, tmp_path):
    first = build_frozen_plan(plan)
    duplicate = tmp_path / "relocated"
    shutil.copytree(plan.parent, duplicate)
    assert build_frozen_plan(duplicate / "plan.json") == first
    output = tmp_path / "frozen.json"
    freeze_plan(plan, output)
    original = output.read_bytes()
    (plan.parent / "rubric.md").write_text("Changed rubric")
    assert build_frozen_plan(plan)["plan_sha256"] != first["plan_sha256"]
    with pytest.raises(FileExistsError):
        freeze_plan(plan, output)
    assert output.read_bytes() == original
    assert json.loads(original)["rubric"]["text"] == first["rubric"]["text"]
    assert not list(tmp_path.glob(".research-plan-*"))


@pytest.mark.parametrize("edit,match", [
    (lambda c: c['items'][1].update(path='train.json'), 'duplicate'),
    (lambda c: c['items'][1].update(template_family='demo-train'), 'family crosses'),
    (lambda c: c['items'][0].update(domain='demo-test'), 'reserved test domain'),
    (lambda c: c['items'][1].update(split='train'), 'each contain'),
    (lambda c: c.update(reserved_test_domains=['absent']), 'must have test examples'),
    (lambda c: c['sampling'].update(max_items=3), 'evaluation-only'),
    (lambda c: c['external_criteria'].update(source_path='test.json'), 'separate'),
    (lambda c: c['budget'].update(max_spend_usd=float('nan')), 'non-finite'),
])
def test_rejects_invalid_experiment_without_publishing(plan, tmp_path, edit, match):
    change(plan, edit)
    output = tmp_path / 'frozen.json'
    with pytest.raises(ValueError, match=match):
        freeze_plan(plan, output)
    assert not output.exists()


@pytest.mark.parametrize('edit', [
    lambda c: c['reference'].pop('model_version'),
    lambda c: c['sampling'].update(random_audit_fraction=0),
    lambda c: c['thresholds'].update(min_recall=1.1),
    lambda c: c['budget'].update(max_tokens=True),
    lambda c: c.update(measurement_profile='ai_ai'),
    lambda c: c['external_criteria'].update(missing_data_policy='negative'),
])
def test_requires_versioned_bounded_human_ai_contract(plan, edit):
    change(plan, edit)
    with pytest.raises(ValidationError):
        build_frozen_plan(plan)


def test_invalid_raw_record_rejected(plan):
    source = plan.parent / 'train.json'
    data = json.loads(source.read_text())
    del data['turns']
    source.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        build_frozen_plan(plan)


def test_content_and_split_policy_change_fingerprint(plan):
    before = build_frozen_plan(plan)
    source = plan.parent / 'train.json'
    source.write_text(source.read_text() + '\n')
    after = build_frozen_plan(plan)
    assert after['plan_sha256'] != before['plan_sha256']
    assert after['split_sha256'] == before['split_sha256']
    change(plan, lambda c: c['thresholds'].update(min_recall=0.9))
    assert build_frozen_plan(plan)['plan_sha256'] != after['plan_sha256']


@pytest.mark.parametrize("field", ["item", "rubric", "criteria"])
@pytest.mark.parametrize("escape", ["absolute", "parent", "symlink"])
def test_rejects_outside_snapshot_paths(plan, tmp_path, field, escape):
    outside = tmp_path / "outside.json"
    outside.write_text((plan.parent / "train.json").read_text())
    if escape == "absolute":
        name = str(outside)
    elif escape == "parent":
        name = "../outside.json"
    else:
        (plan.parent / "link.json").symlink_to(outside)
        name = "link.json"
    def edit(config):
        if field == "item":
            config["items"][0]["path"] = name
        elif field == "rubric":
            config["rubric_path"] = name
        else:
            config["external_criteria"]["source_path"] = name
    change(plan, edit)
    output = tmp_path / "frozen.json"
    with pytest.raises(ValueError, match="plan directory"):
        freeze_plan(plan, output)
    assert not output.exists()


def test_rejects_rubric_reused_as_external_evidence(plan):
    change(plan, lambda c: c["external_criteria"].update(source_path="rubric.md"))
    with pytest.raises(ValueError, match="separate.*rubric"):
        build_frozen_plan(plan)
