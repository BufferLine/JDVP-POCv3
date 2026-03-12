# POCv1 and POCv2 Migration Matrix

Classification:

- `keep`: reusable with little or no semantic change
- `adapt`: reusable concept or structure, but code must change before adoption
- `drop`: do not carry into POCv3

## POCv1

| Source | Decision | Reason |
|---|---|---|
| `README.md` | adapt | Useful history of user flows and interfaces, but protocol model is obsolete. |
| `src/models/jsv.py` | drop | Encodes pre-v1.2 variables such as `decision_status` and `confidence_source`. |
| `src/models/dv.py` | drop | Encodes obsolete DV fields and non-canonical aggregate helpers. |
| `src/extraction/dv_calculator.py` | drop | Uses lookup constants instead of JDVP v1.4 ordinal derivation. |
| `src/extraction/jsv_extractor.py` | adapt | Marker/extraction heuristics may be mined for baseline patterns, but output model must be rewritten. |
| `src/extraction/llm_jsv_extractor.py` | adapt | Prompting and parsing flow may be reusable after contract rewrite. |
| `src/extraction/ensemble_extractor.py` | adapt | Ensemble idea is still useful, but only after single-track core is stable. |
| `src/agent/citizen_ai.py` | adapt | Observation preprocessing may be useful if decoupled from obsolete JSV semantics. |
| `src/reporting/report_generator.py` | drop | Report layer is too tied to old artifact shapes. |
| `src/simulation/` | adapt | Simulation concepts and test personas are potentially useful for fixture generation, not as a first milestone. |
| `src/web/` | drop | UI-heavy surface is out of scope for initial POCv3. |
| `src/calibration/` | drop | Calibration against old variable set is not transferable directly. |
| `src/benchmark/` | drop | Benchmark utilities assume obsolete fields and reporting contracts. |
| `tests/test_contexts.py` | adapt | Context coverage ideas may become fixture cases after model rewrite. |
| `tests/test_core.py` | adapt | Can inspire new protocol-core tests, but not directly reusable. |
| `tests/test_visualization.py` | drop | Visualization is not in initial POCv3 scope. |
| `tests/test_web.py` | drop | Web surface removed from initial scope. |

## POCv2

| Source | Decision | Reason |
|---|---|---|
| `README.md` | adapt | Clear separation of protocol and method layers is worth preserving. |
| `docs/ARCHITECTURE.md` | adapt | Track-oriented architecture is reusable after simplifying scope. |
| `docs/OPERATIONS.md` | adapt | Runbook structure is useful, but current commands assume heavy research automation. |
| `docs/DATA_STRATEGY.md` | adapt | Governance concepts are useful later, not for M1-M2 core build. |
| `docs/ARTIFACT_MANAGEMENT.md` | adapt | Artifact separation is valuable, but registry-heavy layout is too large for the first pass. |
| `src/tracks/base.py` | keep | Clean extraction-track interface worth retaining conceptually. |
| `src/tracks/heuristic_tracks.py` | adapt | Baseline extractor shell is useful, but DV generation must be removed from track code. |
| `src/tracks/live_llm_tracks.py` | adapt | Provider adapters may be reusable after protocol-core separation. |
| `src/tracks/factory.py` | adapt | Factory pattern is useful once at least two tracks exist. |
| `src/core/dv_transition.py` | drop | Lookup-based DV semantics conflict with JDVP v1.4. |
| `src/validation/annotation_validator.py` | adapt | Validation wrapper approach is useful, but schema target must change. |
| `src/validation/raw_validator.py` | adapt | Raw input validation remains necessary in POCv3. |
| `src/orchestration/run_pipeline.py` | adapt | Overall orchestration shape is reusable after drastic simplification. |
| `src/orchestration/release_gate.py` | drop | Release gate is premature for initial v3 milestones. |
| `src/orchestration/run_learning_cycle.py` | drop | Learning-cycle automation is too heavy before the core path is stable. |
| `src/labeling/` | drop | Silver/gold machinery is deferred until later milestones. |
| `src/eval/` | adapt | Comparative evaluation logic can return after M3, not before. |
| `src/storage/` | drop | Storage abstraction is unnecessary in initial v3. |
| `src/common/simple_yaml.py` | keep | Small local utility may be reusable if config stays YAML-based. |
| `src/common/registry.py` | drop | Registry system is not needed for first build. |
| `src/schemas/raw_conversation_v2.json` | adapt | Raw conversation contract is a good starting point for v3 input schema. |
| `src/schemas/annotation_v2.json` | drop | Mixed canonical/overlay schema is too permissive and not aligned to protocol v1.4 strictness. |
| `config/tracks*.yaml` | adapt | Useful examples for track configuration, but simplify to baseline-first. |
| `config/prompts/` | adapt | Prompt-pack layout is reusable for future LLM tracks. |
| `scripts/validate_raw_dataset.py` | adapt | Raw validation entrypoint is likely reusable after schema swap. |
| `scripts/generate_*dataset.py` | drop | Dataset generation is not in the first implementation milestone. |

## POCv3 Adoption Rule

POCv3 only imports code when both conditions are true:

1. the imported component does not encode obsolete JDVP semantics
2. adapting the component is cheaper than rewriting it cleanly

When either condition fails, rewrite.
