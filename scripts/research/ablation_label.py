#!/usr/bin/env python3
"""Prior-anchoring ablation labeler (Task 4 of V3 final re-analysis).

Re-labels a fixed conversation subset with gemma4:26b (local ollama) under two
conditioning regimes so their volatility / trend distributions can be compared:

    --conditioning prior_scores  : feed the observer its own prior-turn scores
                                   (reproduces scripts/label_sharegpt_3axis.py)
    --conditioning independent   : label each turn with no prior-score anchor

Only the *conditioning* differs between runs; context window, prompt, model,
temperature and DA derivation are identical to the reference pipeline.

Usage:
    python3 scripts/research/ablation_label.py --conditioning prior_scores --subset pilot15
    python3 scripts/research/ablation_label.py --conditioning independent  --subset pilot15
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

# Data + prompt live in the checkout's data/ + config/. data/ is gitignored, so in
# a git worktree it is absent; default to the repo root and let JDVP_DATA_ROOT /
# --data-root point at the primary checkout that holds the extracted archives.
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("JDVP_DATA_ROOT", HERE.parents[1]))
FIELDS = ["judgment_delegation", "cognitive_passivity", "information_dependency"]


def derive_da(jh, cp, id_):
    return max(0, min(10, round(0.162 * jh + 0.570 * cp + 0.268 * id_, 2)))


def call_ollama(system, user, model):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.95, "top_k": 64},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["message"]["content"]


def parse(raw):
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
                r = {}
                for f in FIELDS:
                    v = obj.get(f)
                    if isinstance(v, (int, float)) and 0 <= v <= 10:
                        r[f] = round(float(v), 1)
                    else:
                        return None
                r["reasoning"] = obj.get("reasoning", "")
                return r
    return None


def build_prompt(iid, tn, human_input, ctx, prior, use_prior):
    """Identical to reference build_prompt, but prior block is gated by use_prior."""
    parts = []
    for ct in ctx[-5:]:
        parts.append(f"[T{ct['turn_number']}] H: {ct['human_input'][:250]}")
        parts.append(f"[T{ct['turn_number']}] A: {ct.get('ai_response', '')[:150]}")
    if use_prior and prior:
        parts.append(
            f"\nPrior: JD={prior.get('judgment_delegation', '?')} "
            f"CP={prior.get('cognitive_passivity', '?')} "
            f"ID={prior.get('information_dependency', '?')}"
        )
    parts.append(f"\n>>> Turn {tn} | {iid}\nHuman: {human_input[:500]}\n\nJSON only.")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditioning", required=True, choices=["prior_scores", "independent"])
    ap.add_argument("--subset", default="pilot15", help="key in ablation_iids.json (pilot15|all30)")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--prompt", type=Path,
                    default=DEFAULT_DATA_ROOT / "config/prompts/level_observer_3axis_cot.txt")
    args = ap.parse_args()

    use_prior = args.conditioning == "prior_scores"
    prompt_sys = args.prompt.read_text()
    iids = json.loads((HERE / "ablation_iids.json").read_text())[args.subset]

    outdir = args.data_root / "data" / "silver" / f"sharegpt-3axis-ablation-{args.conditioning}"
    outdir.mkdir(parents=True, exist_ok=True)

    inter_dir = args.data_root / "data/open-data/sharegpt/interactions"
    total = failed = skipped = 0
    t0 = time.time()

    for idx, iid in enumerate(iids):
        d = json.loads((inter_dir / f"{iid}.json").read_text())
        turns = d["turns"]
        iid_dir = outdir / iid
        iid_dir.mkdir(parents=True, exist_ok=True)

        prior = None
        for turn in turns:
            tn = turn["turn_number"]
            out = iid_dir / f"turn-{tn}.json"
            if out.exists():
                prior = json.loads(out.read_text())
                skipped += 1
                continue
            ctx = [t for t in turns if t["turn_number"] < tn]
            prompt = build_prompt(iid, tn, turn["human_input"], ctx, prior, use_prior)

            result = None
            for attempt in range(3):
                try:
                    raw = call_ollama(prompt_sys, prompt, args.model)
                    result = parse(raw)
                    if result:
                        break
                except Exception as e:
                    if attempt == 2:
                        print(f"  ERR {iid} t{tn}: {e}", flush=True)
                    time.sleep(1)
            if not result:
                failed += 1
                continue
            result["delegation_awareness_derived"] = derive_da(
                result["judgment_delegation"], result["cognitive_passivity"],
                result["information_dependency"],
            )
            # prior carries forward regardless of use_prior so the "independent"
            # run is a pure ablation of the prompt block, not of state tracking.
            prior = result
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            total += 1
        el = time.time() - t0
        print(f"[{idx+1}/{len(iids)}] {iid}: {len(turns)}t | ok={total} skip={skipped} "
              f"fail={failed} | {total/el:.3f}t/s | {el:.0f}s", flush=True)

    print(f"\nDONE {args.conditioning}/{args.subset}: {total} ok, {skipped} skip, "
          f"{failed} fail in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
