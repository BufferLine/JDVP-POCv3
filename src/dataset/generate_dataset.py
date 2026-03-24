"""Synthetic dataset generation and manifest materialization for M6."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.catalog.sqlite_store import (
    CatalogDatasetGenerationItemRecord,
    CatalogDatasetGenerationRunRecord,
    CatalogDatasetRecord,
    CatalogStore,
)
from src.contracts.raw_interaction_validate import RawInteractionValidator
from src.method.evidence.prompt_loader import load_prompt
from src.method.tracks.llm_observer import LLMProvider, create_env_backed_provider
from src.pipeline.run_storage import write_json
from src.shared_utils import load_json as _load_json
from src.shared_utils import tokenize as _tokenize


DEFAULT_SCENARIO_PACK = Path("config/datasets/general_scenarios_v1.json")
DATASET_GENERATION_PROMPT_VERSION = "dataset-utterance-generator-v1"
DATASET_TURN_SIM_PROMPT_VERSION = "dataset-turn-simulator-v1"
DEFAULT_LLM_JSON_RETRY_COUNT = 3
QUALITY_BANNED_PATTERN = re.compile(r"(as an ai|language model|i cannot|i can't provide)")


def _json_candidates(text: str) -> list[str]:
    out: list[str] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                out.append(text[start : i + 1])
    return out


def _load_json_obj(text: str) -> dict[str, Any]:
    for candidate in [text.strip()] + _json_candidates(text):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in model output")


def _salvage_scalar_response(text: str, *, expected_key: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    pattern = re.compile(rf'"{re.escape(expected_key)}"\s*:\s*"((?:[^"\\]|\\.)*)"')
    match = pattern.search(cleaned)
    if match:
        raw_value = match.group(1)
        return {expected_key: json.loads(f'"{raw_value}"')}
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
            match = pattern.search(cleaned)
            if match:
                raw_value = match.group(1)
                return {expected_key: json.loads(f'"{raw_value}"')}
    return None


def _render_template(template: Any, slots: Mapping[str, str]) -> Any:
    if isinstance(template, str):
        return template.format(**slots)
    if isinstance(template, list):
        return [_render_template(value, slots) for value in template]
    if isinstance(template, dict):
        return {key: _render_template(value, slots) for key, value in template.items()}
    return template


def _build_generation_run_id(
    dataset_root: Path,
    generation_mode: str,
    seed: int | None = None,
    count_per_scenario: int | None = None,
) -> str:
    run_id = f"{dataset_root.resolve(strict=False)}::{generation_mode}"
    if seed is not None:
        run_id += f"::seed={seed}"
    if count_per_scenario is not None:
        run_id += f"::count={count_per_scenario}"
    return run_id


def _build_item_id(*, scenario_id: str, sample_index: int) -> str:
    return f"{scenario_id}:{sample_index:03d}"


def _write_generation_progress(
    *,
    progress_path: Path,
    dataset_id: str,
    generation_run_id: str,
    generation_mode: str,
    target_item_count: int,
    accepted_count: int,
    failed_count: int,
    pending_count: int,
) -> None:
    write_json(
        progress_path,
        {
            "schema_version": "pocv3-dataset-generation-progress-v1",
            "dataset_id": dataset_id,
            "generation_run_id": generation_run_id,
            "generation_mode": generation_mode,
            "target_item_count": target_item_count,
            "accepted_count": accepted_count,
            "failed_count": failed_count,
            "pending_count": pending_count,
            "is_complete": accepted_count == target_item_count,
        },
    )


def _quality_gate_interaction(interaction: dict[str, Any]) -> list[str]:
    turns = interaction.get("turns", [])
    reasons: list[str] = []
    if not isinstance(turns, list) or not turns:
        return ["missing_turns"]

    all_text = []
    seen_turn_texts: set[str] = set()
    duplicate_turns = 0
    for turn in turns:
        human_input = str(turn.get("human_input", "")).strip()
        ai_response = str(turn.get("ai_response", "")).strip()
        combined = f"{human_input} || {ai_response}".lower()
        if combined in seen_turn_texts:
            duplicate_turns += 1
        seen_turn_texts.add(combined)
        if QUALITY_BANNED_PATTERN.search(human_input.lower()) or QUALITY_BANNED_PATTERN.search(ai_response.lower()):
            reasons.append("meta_ai_phrase")
        all_text.extend(_tokenize(human_input))
        all_text.extend(_tokenize(ai_response))
        jsv_hint = turn.get("meta", {}).get("jsv_hint", {})
        holder = str(jsv_hint.get("judgment_holder", ""))
        if holder == "AI" and not re.search(
            r"(recommend|choose|pick|what would you|your recommendation|just tell me|if you had to pick)",
            human_input.lower(),
        ):
            reasons.append("weak_ai_delegation_signal")
        if holder == "Human" and re.search(r"(just decide|you choose|pick for me)", human_input.lower()):
            reasons.append("human_turn_overdelegates")

    if duplicate_turns > 0:
        reasons.append("duplicate_turn_pair")
    if len(all_text) < 30:
        reasons.append("too_short_tokens")
    if all_text:
        lexical_diversity = len(set(all_text)) / len(all_text)
        if lexical_diversity < 0.32:
            reasons.append("low_lexical_diversity")
    return sorted(set(reasons))


def _turn_guidance_from_jsv_hint(jsv_hint: Mapping[str, Any]) -> dict[str, str]:
    holder = str(jsv_hint.get("judgment_holder", ""))
    if holder == "AI":
        return {
            "human_guidance": (
                "This turn should end with the human asking for a recommendation, judgment call, "
                "or direct choice. The human should not introduce major new facts on this turn."
            ),
            "assistant_guidance": (
                "This turn should answer with a recommendation or clear direction, grounded only "
                "in already stated priorities and constraints. Do not ask another follow-up question."
            ),
        }
    if holder == "Shared":
        return {
            "human_guidance": (
                "This turn should sound collaborative: ask for weighing, comparison, or structured help "
                "while staying engaged in the decision."
            ),
            "assistant_guidance": (
                "This turn should compare tradeoffs or structure the decision without taking full ownership."
            ),
        }
    return {
        "human_guidance": (
            "This turn should keep judgment with the human: ask for framing, clarification, or comparison "
            "without handing over the decision."
        ),
        "assistant_guidance": (
            "This turn should help structure the decision without deciding for the human."
        ),
    }


@dataclass(frozen=True)
class DatasetUtteranceGenerator:
    provider: LLMProvider
    model_id: str
    prompt_version: str = DATASET_GENERATION_PROMPT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_prompt", load_prompt("dataset_utterance_generator_system.txt"))

    def materialize_turns(
        self,
        *,
        interaction_id: str,
        scenario_id: str,
        scenario_title: str,
        context_module: str,
        slots: Mapping[str, str],
        blueprint_id: str | None,
        turn_payloads: Sequence[dict[str, Any]],
    ) -> list[dict[str, str]]:
        prompt_payload = {
            "interaction_id": interaction_id,
            "scenario_id": scenario_id,
            "scenario_title": scenario_title,
            "context_module": context_module,
            "blueprint_id": blueprint_id,
            "slots": dict(slots),
            "turns": [
                {
                    "turn_number": int(turn["turn_number"]),
                    "reference_human_input": str(turn["human_input"]),
                    "reference_ai_response": str(turn["ai_response"]),
                    "jsv_hint": turn.get("meta", {}).get("jsv_hint", {}),
                }
                for turn in turn_payloads
            ],
        }
        raw_response = self.provider.generate(
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError("dataset utterance generator returned non-JSON content") from exc
        turns = payload.get("turns")
        if not isinstance(turns, list) or len(turns) != len(turn_payloads):
            raise ValueError("dataset utterance generator returned an invalid turn list")
        rewritten_turns: list[dict[str, str]] = []
        for expected_turn, generated_turn in zip(turn_payloads, turns):
            if not isinstance(generated_turn, dict):
                raise ValueError("dataset utterance generator returned a non-object turn")
            if int(generated_turn.get("turn_number", -1)) != int(expected_turn["turn_number"]):
                raise ValueError("dataset utterance generator returned mismatched turn numbering")
            human_input = generated_turn.get("human_input")
            ai_response = generated_turn.get("ai_response")
            if not isinstance(human_input, str) or not human_input.strip():
                raise ValueError("dataset utterance generator returned empty human_input")
            if not isinstance(ai_response, str) or not ai_response.strip():
                raise ValueError("dataset utterance generator returned empty ai_response")
            rewritten_turns.append(
                {
                    "human_input": human_input.strip(),
                    "ai_response": ai_response.strip(),
                }
            )
        return rewritten_turns


@dataclass(frozen=True)
class TurnSimulatedDatasetGenerator:
    provider: LLMProvider
    model_id: str
    prompt_version: str = DATASET_TURN_SIM_PROMPT_VERSION
    max_attempts: int = DEFAULT_LLM_JSON_RETRY_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "human_system_prompt", load_prompt("dataset_turn_sim_human_system.txt"))
        object.__setattr__(self, "assistant_system_prompt", load_prompt("dataset_turn_sim_assistant_system.txt"))

    def materialize_turns(
        self,
        *,
        interaction_id: str,
        scenario_id: str,
        scenario_title: str,
        context_module: str,
        slots: Mapping[str, str],
        blueprint_id: str | None,
        turn_payloads: Sequence[dict[str, Any]],
    ) -> list[dict[str, str]]:
        history: list[dict[str, Any]] = []
        generated_turns: list[dict[str, str]] = []
        scenario_brief = {
            "interaction_id": interaction_id,
            "scenario_id": scenario_id,
            "scenario_title": scenario_title,
            "context_module": context_module,
            "blueprint_id": blueprint_id,
            "slots": dict(slots),
            "turn_count": len(turn_payloads),
        }
        for turn_payload in turn_payloads:
            turn_guidance = _turn_guidance_from_jsv_hint(turn_payload.get("meta", {}).get("jsv_hint", {}))
            turn_spec = {
                "turn_number": int(turn_payload["turn_number"]),
                "reference_human_input": str(turn_payload["human_input"]),
                "reference_ai_response": str(turn_payload["ai_response"]),
                "jsv_hint": turn_payload.get("meta", {}).get("jsv_hint", {}),
                "human_guidance": turn_guidance["human_guidance"],
                "assistant_guidance": turn_guidance["assistant_guidance"],
            }
            human_input = self._generate_human_turn(
                scenario_brief=scenario_brief,
                turn_spec=turn_spec,
                history=history,
            )
            history.append({"speaker": "human", "content": human_input})
            ai_response = self._generate_assistant_turn(
                scenario_brief=scenario_brief,
                turn_spec=turn_spec,
                history=history,
            )
            history.append({"speaker": "ai", "content": ai_response})
            generated_turns.append(
                {
                    "human_input": human_input,
                    "ai_response": ai_response,
                }
            )
        return generated_turns

    def _generate_human_turn(
        self,
        *,
        scenario_brief: dict[str, Any],
        turn_spec: dict[str, Any],
        history: Sequence[dict[str, Any]],
    ) -> str:
        payload = {
            "scenario": scenario_brief,
            "current_turn": turn_spec,
            "history": list(history),
        }
        obj = self._generate_json_response(
            system_prompt=self.human_system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            expected_key="human_input",
        )
        content = obj.get("human_input")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("turn simulator returned empty human_input")
        return content.strip()

    def _generate_assistant_turn(
        self,
        *,
        scenario_brief: dict[str, Any],
        turn_spec: dict[str, Any],
        history: Sequence[dict[str, Any]],
    ) -> str:
        payload = {
            "scenario": scenario_brief,
            "current_turn": turn_spec,
            "history": list(history),
        }
        obj = self._generate_json_response(
            system_prompt=self.assistant_system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            expected_key="ai_response",
        )
        content = obj.get("ai_response")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("turn simulator returned empty ai_response")
        return content.strip()

    def _generate_json_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_key: str,
    ) -> dict[str, Any]:
        current_prompt = user_prompt
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw_response = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=current_prompt,
            )
            try:
                obj = _load_json_obj(raw_response)
            except ValueError as exc:
                salvaged = _salvage_scalar_response(raw_response, expected_key=expected_key)
                if salvaged is not None:
                    return salvaged
                last_error = exc
                current_prompt = (
                    f"{user_prompt}\n\n"
                    "Your previous response was invalid because it did not contain a JSON object.\n"
                    f"Return JSON only with a single key `{expected_key}`.\n"
                    f"Previous invalid response:\n{raw_response}"
                )
                continue
            value = obj.get(expected_key)
            if isinstance(value, str) and value.strip():
                return obj
            last_error = ValueError(f"turn simulator response missing non-empty {expected_key}")
            current_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid because `{expected_key}` was missing or empty.\n"
                f"Return JSON only with a single key `{expected_key}`.\n"
                f"Previous invalid response:\n{raw_response}"
            )
        if last_error is not None:
            raise last_error
        raise ValueError(f"turn simulator failed to produce {expected_key}")


def _choose_slots(slot_options: Mapping[str, Sequence[str]], rng: random.Random) -> dict[str, str]:
    return {name: str(rng.choice(list(values))) for name, values in slot_options.items()}


def _select_variant(
    template: dict[str, Any],
    *,
    field_name: str,
    rng: random.Random,
) -> tuple[str | None, int | None]:
    options_key = f"{field_name}_options"
    options = template.get(options_key)
    if not isinstance(options, list) or not options:
        value = template.get(field_name)
        return (str(value) if isinstance(value, str) else None), None

    selected_index = rng.randrange(len(options))
    selected_value = options[selected_index]
    if not isinstance(selected_value, str):
        raise ValueError(f"{options_key} entries must be strings")
    return selected_value, selected_index


def _materialize_turn_template(
    turn_template: Mapping[str, Any],
    *,
    slots: Mapping[str, str],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    template = dict(turn_template)
    human_input, human_variant_index = _select_variant(template, field_name="human_input", rng=rng)
    ai_response, ai_variant_index = _select_variant(template, field_name="ai_response", rng=rng)
    if human_input is not None:
        template["human_input"] = human_input
    if ai_response is not None:
        template["ai_response"] = ai_response
    template.pop("human_input_options", None)
    template.pop("ai_response_options", None)

    rendered = _render_template(template, slots)
    variant_metadata = {
        "turn_number": int(rendered["turn_number"]),
        "human_input_variant_index": human_variant_index,
        "ai_response_variant_index": ai_variant_index,
    }
    return rendered, variant_metadata


def _select_turn_templates(
    scenario: Mapping[str, Any],
    *,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], str | None]:
    blueprints = scenario.get("blueprints")
    if isinstance(blueprints, list) and blueprints:
        selected_blueprint = blueprints[rng.randrange(len(blueprints))]
        if not isinstance(selected_blueprint, dict):
            raise ValueError("scenario blueprints must be objects")
        turn_templates = selected_blueprint.get("turn_templates")
        if not isinstance(turn_templates, list) or not turn_templates:
            raise ValueError("scenario blueprint turn_templates must be a non-empty list")
        blueprint_id = selected_blueprint.get("blueprint_id")
        return list(turn_templates), (str(blueprint_id) if blueprint_id is not None else None)

    turn_templates = scenario.get("turn_templates")
    if not isinstance(turn_templates, list) or not turn_templates:
        raise ValueError("scenario must define turn_templates or blueprints")
    return list(turn_templates), None


def _build_interaction(
    *,
    dataset_name: str,
    scenario: dict[str, Any],
    sample_index: int,
    rng: random.Random,
    utterance_generator: DatasetUtteranceGenerator | TurnSimulatedDatasetGenerator | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = _choose_slots(scenario.get("slot_options", {}), rng)
    turn_templates, blueprint_id = _select_turn_templates(scenario, rng=rng)
    interaction_id = f"{dataset_name}-{scenario['scenario_id']}-{sample_index:03d}"
    turns = []
    turn_variant_choices = []
    for turn_template in turn_templates:
        turn_payload, variant_metadata = _materialize_turn_template(
            turn_template,
            slots=slots,
            rng=rng,
        )
        turns.append(turn_payload)
        turn_variant_choices.append(variant_metadata)
    if utterance_generator is not None:
        generated_turns = utterance_generator.materialize_turns(
            interaction_id=interaction_id,
            scenario_id=str(scenario["scenario_id"]),
            scenario_title=str(scenario.get("title", scenario["scenario_id"])),
            context_module=str(scenario["context_module"]),
            slots=slots,
            blueprint_id=blueprint_id,
            turn_payloads=turns,
        )
        for turn_payload, generated_turn in zip(turns, generated_turns):
            turn_payload["human_input"] = generated_turn["human_input"]
            turn_payload["ai_response"] = generated_turn["ai_response"]
    interaction = {
        "interaction_id": interaction_id,
        "context_module": scenario["context_module"],
        "participants": {
            "human_id": str(scenario.get("participants", {}).get("human_id", "human-1")),
            "ai_id": str(scenario.get("participants", {}).get("ai_id", "ai-1")),
        },
        "turns": turns,
    }
    item_metadata = {
        "interaction_id": interaction_id,
        "scenario_id": scenario["scenario_id"],
        "scenario_title": scenario.get("title", scenario["scenario_id"]),
        "slot_values": slots,
        "blueprint_id": blueprint_id,
        "turn_variant_choices": turn_variant_choices,
    }
    return interaction, item_metadata


def _assign_splits(
    interaction_ids: list[str],
    *,
    rng: random.Random,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[str]]:
    shuffled = list(interaction_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_cutoff = int(total * train_ratio)
    validation_cutoff = train_cutoff + int(total * validation_ratio)
    return {
        "train": shuffled[:train_cutoff],
        "validation": shuffled[train_cutoff:validation_cutoff],
        "test": shuffled[validation_cutoff:],
    }


def _manifest_items(
    *,
    items: list[dict[str, Any]],
    split_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    split_by_id = {
        interaction_id: split_name
        for split_name, interaction_ids in split_map.items()
        for interaction_id in interaction_ids
    }
    manifest_items = []
    for item in items:
        manifest_items.append(
            {
                **item,
                "split": split_by_id[item["interaction_id"]],
            }
        )
    return manifest_items


def generate_dataset(
    *,
    dataset_name: str,
    dataset_version: str,
    output_root: Path,
    scenario_pack_path: Path = DEFAULT_SCENARIO_PACK,
    dataset_kind: str = "generated",
    count_per_scenario: int = 1,
    seed: int = 7,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    generation_mode: str = "template",
    enable_quality_gate: bool = True,
) -> Path:
    if count_per_scenario < 1:
        raise ValueError("count_per_scenario must be at least 1")
    if train_ratio < 0 or validation_ratio < 0 or train_ratio + validation_ratio > 1:
        raise ValueError("split ratios must be non-negative and sum to at most 1")
    if generation_mode not in {"template", "llm", "llm_turn_simulated"}:
        raise ValueError("generation_mode must be one of: template, llm, llm_turn_simulated")

    scenario_pack = _load_json(scenario_pack_path)
    rng = random.Random(seed)
    validator = RawInteractionValidator()
    utterance_generator: DatasetUtteranceGenerator | TurnSimulatedDatasetGenerator | None = None
    if generation_mode == "llm":
        provider, model_id = create_env_backed_provider()
        utterance_generator = DatasetUtteranceGenerator(provider=provider, model_id=model_id)
    elif generation_mode == "llm_turn_simulated":
        provider, model_id = create_env_backed_provider()
        utterance_generator = TurnSimulatedDatasetGenerator(provider=provider, model_id=model_id)

    dataset_root = output_root / dataset_name / dataset_version
    dataset_id = f"{dataset_kind}/{dataset_name}/{dataset_version}"
    generation_run_id = _build_generation_run_id(
        dataset_root, generation_mode, seed=seed, count_per_scenario=count_per_scenario,
    )
    progress_path = dataset_root / "generation_progress.json"
    catalog = CatalogStore()
    target_item_count = len(scenario_pack["scenarios"]) * count_per_scenario
    catalog.upsert_dataset_generation_run(
        CatalogDatasetGenerationRunRecord(
            generation_run_id=generation_run_id,
            dataset_id=dataset_id,
            dataset_root=str(dataset_root),
            generation_mode=generation_mode,
            scenario_pack_path=str(scenario_pack_path),
            target_item_count=target_item_count,
            accepted_count=0,
            failed_count=0,
            status="running",
        )
    )
    existing_generation_items = {
        row["item_id"]: row
        for row in catalog.list_dataset_generation_items(generation_run_id=generation_run_id)
    }
    manifest_items_seed: list[dict[str, Any]] = []
    interaction_ids: list[str] = []
    accepted_count = 0
    failed_count = 0
    rejected_count = 0

    for scenario in scenario_pack["scenarios"]:
        for sample_index in range(count_per_scenario):
            item_id = _build_item_id(
                scenario_id=str(scenario["scenario_id"]),
                sample_index=sample_index,
            )
            planned_relative_path = (
                Path("interactions")
                / f"{dataset_name}-{scenario['scenario_id']}-{sample_index:03d}.json"
            )
            existing_row = existing_generation_items.get(item_id)
            if (
                existing_row is not None
                and existing_row["status"] == "accepted"
                and (dataset_root / str(existing_row["relative_path"])).is_file()
                and existing_row.get("item_payload_json")
            ):
                item_payload = json.loads(str(existing_row["item_payload_json"]))
                manifest_items_seed.append(item_payload)
                interaction_ids.append(str(item_payload["interaction_id"]))
                accepted_count += 1
                continue
            attempt_count = int(existing_row["attempt_count"]) if existing_row is not None else 0
            # Use a per-item deterministic RNG so that skipping accepted items
            # during a rerun does not shift the random sequence for later items.
            item_rng = random.Random(f"{seed}:{scenario['scenario_id']}:{sample_index}")
            try:
                interaction, item_metadata = _build_interaction(
                    dataset_name=dataset_name,
                    scenario=scenario,
                    sample_index=sample_index,
                    rng=item_rng,
                    utterance_generator=utterance_generator,
                )
                validator.validate(interaction)
                quality_reasons = (
                    _quality_gate_interaction(interaction)
                    if enable_quality_gate and generation_mode != "template"
                    else []
                )
                if quality_reasons:
                    rejected_count += 1
                    catalog.upsert_dataset_generation_item(
                        CatalogDatasetGenerationItemRecord(
                            generation_run_id=generation_run_id,
                            item_id=item_id,
                            interaction_id=str(interaction["interaction_id"]),
                            scenario_id=str(scenario["scenario_id"]),
                            sample_index=sample_index,
                            relative_path=str(planned_relative_path),
                            status="rejected",
                            attempt_count=attempt_count + 1,
                            item_payload_json=json.dumps(
                                {
                                    **item_metadata,
                                    "relative_path": str(planned_relative_path),
                                    "context_module": interaction["context_module"],
                                    "turn_count": len(interaction["turns"]),
                                    "raw_interaction": interaction,
                                },
                                ensure_ascii=False,
                            ),
                            error_message="; ".join(quality_reasons),
                        )
                    )
                    continue
                relative_path = Path("interactions") / f"{interaction['interaction_id']}.json"
                write_json(dataset_root / relative_path, interaction)
                item_payload = {
                    **item_metadata,
                    "relative_path": str(relative_path),
                    "context_module": interaction["context_module"],
                    "turn_count": len(interaction["turns"]),
                }
                manifest_items_seed.append(item_payload)
                interaction_ids.append(str(interaction["interaction_id"]))
                accepted_count += 1
                catalog.upsert_dataset_generation_item(
                    CatalogDatasetGenerationItemRecord(
                        generation_run_id=generation_run_id,
                        item_id=item_id,
                        interaction_id=str(interaction["interaction_id"]),
                        scenario_id=str(scenario["scenario_id"]),
                        sample_index=sample_index,
                        relative_path=str(relative_path),
                        status="accepted",
                        attempt_count=attempt_count + 1,
                        item_payload_json=json.dumps(item_payload, ensure_ascii=False),
                    )
                )
            except Exception as exc:
                failed_count += 1
                catalog.upsert_dataset_generation_item(
                    CatalogDatasetGenerationItemRecord(
                        generation_run_id=generation_run_id,
                        item_id=item_id,
                        interaction_id=f"{dataset_name}-{scenario['scenario_id']}-{sample_index:03d}",
                        scenario_id=str(scenario["scenario_id"]),
                        sample_index=sample_index,
                        relative_path=str(planned_relative_path),
                        status="failed",
                        attempt_count=attempt_count + 1,
                        error_message=str(exc),
                    )
                )

    split_rng = random.Random(seed + 1)
    split_map = _assign_splits(
        interaction_ids,
        rng=split_rng,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    manifest = {
        "schema_version": "pocv3-dataset-manifest-v1",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_kind": dataset_kind,
        "context_module": scenario_pack["context_module"],
        "scenario_pack_id": scenario_pack["scenario_pack_id"],
        "item_count": len(manifest_items_seed),
        "split_counts": {name: len(values) for name, values in split_map.items()},
        "generation": {
            "mode": generation_mode,
            "seed": seed,
            "count_per_scenario": count_per_scenario,
            "scenario_pack_path": str(scenario_pack_path),
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1 - train_ratio - validation_ratio,
            "llm_model_id": (utterance_generator.model_id if utterance_generator is not None else None),
            "prompt_version": (
                utterance_generator.prompt_version
                if utterance_generator is not None
                else None
            ),
        },
        "items": _manifest_items(items=manifest_items_seed, split_map=split_map),
    }
    pending_count = max(0, target_item_count - accepted_count - failed_count - rejected_count)
    _write_generation_progress(
        progress_path=progress_path,
        dataset_id=dataset_id,
        generation_run_id=generation_run_id,
        generation_mode=generation_mode,
        target_item_count=target_item_count,
        accepted_count=accepted_count,
        failed_count=failed_count + rejected_count,
        pending_count=pending_count,
    )
    if accepted_count != target_item_count:
        catalog.upsert_dataset_generation_run(
            CatalogDatasetGenerationRunRecord(
                generation_run_id=generation_run_id,
                dataset_id=dataset_id,
                dataset_root=str(dataset_root),
                generation_mode=generation_mode,
                scenario_pack_path=str(scenario_pack_path),
                target_item_count=target_item_count,
                accepted_count=accepted_count,
                failed_count=failed_count + rejected_count,
                status="partial",
                error_message="dataset generation incomplete; rerun the same command to retry failed or rejected items",
            )
        )
        raise RuntimeError(
            f"dataset generation incomplete: accepted={accepted_count}, "
            f"failed={failed_count}, rejected={rejected_count}, target={target_item_count}. "
            "rerun the same command to retry failed or rejected items"
        )

    write_json(dataset_root / "manifest.json", manifest)
    write_json(dataset_root / "splits.json", split_map)
    catalog.upsert_dataset(
        CatalogDatasetRecord(
            dataset_id=str(manifest["dataset_id"]),
            dataset_root=str(dataset_root),
            dataset_kind=str(manifest["dataset_kind"]),
            scenario_pack_id=str(manifest["scenario_pack_id"]),
            generation_seed=int(seed),
            count_per_scenario=int(count_per_scenario),
        ),
        items=manifest["items"],
    )
    catalog.upsert_dataset_generation_run(
        CatalogDatasetGenerationRunRecord(
            generation_run_id=generation_run_id,
            dataset_id=dataset_id,
            dataset_root=str(dataset_root),
            generation_mode=generation_mode,
            scenario_pack_path=str(scenario_pack_path),
            target_item_count=target_item_count,
            accepted_count=accepted_count,
            failed_count=failed_count + rejected_count,
            status="completed",
        )
    )
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic JDVP research dataset")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--scenario-pack", type=Path, default=DEFAULT_SCENARIO_PACK)
    parser.add_argument("--dataset-kind", choices=["generated", "fixtures", "raw"], default="generated")
    parser.add_argument("--count-per-scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--generation-mode", choices=["template", "llm", "llm_turn_simulated"], default="template")
    parser.add_argument("--disable-quality-gate", action="store_true")
    args = parser.parse_args()

    dataset_root = generate_dataset(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        output_root=args.output_root,
        scenario_pack_path=args.scenario_pack,
        dataset_kind=args.dataset_kind,
        count_per_scenario=args.count_per_scenario,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        generation_mode=args.generation_mode,
        enable_quality_gate=not args.disable_quality_gate,
    )
    print(f"Dataset written: {dataset_root}")


if __name__ == "__main__":
    main()
