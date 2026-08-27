# Migration Notes

## Why POCv3 Is A New Build

POCv1 and POCv2 both contain useful implementation ideas, but neither is a safe semantic base for the current JDVP protocol.

POCv1 issue:

- core data models still encode removed variables and old report expectations

POCv2 issue:

- protocol and method layers were separated more cleanly, but DV calculation still depends on lookup constants rather than canonical ordinal derivation

## Migration Principle

Move concepts forward, not protocol mistakes.

This means:

- reuse interfaces, not stale schemas
- reuse orchestration shape, not heavy automation scope
- reuse prompt/config organization, not old extraction contracts

## Rewrite-First Areas

- canonical JSV model
- canonical DV derivation
- trajectory builder
- schema validation against protocol repo

## Adapt-Later Areas

- heuristic baseline extractor
- LLM extraction adapters
- raw input validation
- comparative evaluation

## Defer-Until-Needed Areas

- silver/gold promotion
- release gating
- registries
- dashboards
- bulk synthetic dataset generation

## Immediate Build Target

The first target is not a research automation platform.

It is one correct command that takes one interaction and emits schema-valid JDVP artifacts under the current canonical specification.

## Schema Snapshot Rule

POCv3 uses a vendored copy of canonical protocol schemas for repeatable CI.

This is an operational snapshot, not a new source of truth.
The canonical semantics still belong to `JDVP-protocol`, and schema snapshots here must be refreshed from there.

## Current Version-Alignment Gap

POCv3's vendored snapshot and protocol core currently implement the JDVP v1.4 categorical surface. The canonical sibling protocol repository has since adopted v1.5 level-based scoring. The difference is semantic, not cosmetic:

- core JSV fields change from categorical enums to 0–10 integer levels
- core DV fields change from normalized ordinal fractions to direct integer deltas in `[-10, 10]`
- legacy categorical fixtures require explicit midpoint conversion or regeneration; direct multiplication of prior DVs is not valid

Until the migration is complete, POCv3 should describe itself as a v1.4-compatible research implementation, not as fully aligned with the current canonical protocol. The migration is prioritized in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) and is a prerequisite for new product-facing extraction claims.
