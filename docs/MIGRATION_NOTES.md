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
