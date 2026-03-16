"""CLI wrapper for the service-oriented POC run flow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.service.poc_service import run_interaction_file


def run_poc(
    input_path: Path,
    run_id: str,
    output_root: Path,
    protocol_schema_root: Path | None = None,
    track_name: str = "fixture_hint",
    resume: bool = False,
) -> Path:
    return run_interaction_file(
        input_path=input_path,
        run_id=run_id,
        output_root=output_root,
        protocol_schema_root=protocol_schema_root,
        track_name=track_name,
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the POCv3 M1 smoke pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--protocol-schema-root", type=Path, default=None)
    parser.add_argument(
        "--track",
        default="fixture_hint",
        choices=["fixture_hint", "heuristic_baseline", "llm_observer", "fewshot_prompt"],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = run_poc(
        input_path=args.input,
        run_id=args.run_id,
        output_root=args.output_root,
        protocol_schema_root=args.protocol_schema_root,
        track_name=args.track,
        resume=bool(args.resume),
    )
    print(f"Run written: {run_dir}")


if __name__ == "__main__":
    main()
