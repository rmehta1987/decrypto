#!/usr/bin/env python3
"""Find example trajectories illustrating the two shared qualitative patterns:

  * SUCCESS pattern: model copies *hint* words into its keyword guess
    (rp/fb prediction != true_keywords AND non-default).
  * FAILURE pattern: model parrots the *ground-truth* keywords verbatim
    (rp/fb prediction == true_keywords).

Prints a few example trajectory paths per (condition, task type, pattern).

Usage:
    python find_examples.py <results_dir> [--n 3]
e.g.
    python find_examples.py results/figure_4_tom_gopnik_qwen3_4b_199 --n 3

Notes:
- Operates on combined_history.json (the source of truth used by the runner).
- Also prints the "trajectory file" path under game_transcripts/gopnik_trajectories
  if it exists, for convenience.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from analyze_gopnik import (
    cmp_kw, parse_kw, parse_true, qtype, episode_trials,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # decrypto/
TRAJ_BASE = REPO_ROOT / "game_transcripts" / "gopnik_trajectories"


def trajectory_path(model_seed, cond, env, ep, turn, task):
    """Path under game_transcripts/gopnik_trajectories/ if it exists."""
    p = TRAJ_BASE / model_seed / cond / env / ep / f"turn_{turn}_{task}.json"
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--model-seed", default="model_seed0")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Substring filter on condition folder names "
                         "(default: all qwen3_4b folders)")
    ap.add_argument("--n", type=int, default=3,
                    help="How many examples per (condition, task, pattern)")
    args = ap.parse_args()

    base = args.results_dir / args.model_seed
    if not base.is_dir():
        print(f"Not a directory: {base}", file=sys.stderr)
        sys.exit(2)

    conds = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if args.include is not None:
        conds = [p for p in conds if any(s in p.name for s in args.include)]
    else:
        conds = [p for p in conds if "qwen3_4b" in p.name]

    for cond_dir in conds:
        print("\n" + "=" * 80)
        print(f"CONDITION: {cond_dir.name}")
        print("=" * 80)

        # success_X = copies *hint* words: rp/fb parsed AND != true keywords
        # failure_X = parrots truth:      rp/fb parsed AND == true keywords
        successes = {"rep_change": [], "false_belief": []}
        failures  = {"rep_change": [], "false_belief": []}

        for env_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            for ep_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
                ch = ep_dir / "combined_history.json"
                if not ch.exists():
                    continue
                with open(ch) as f:
                    trials = episode_trials(json.load(f))
                for t in trials:
                    if cmp_kw(t["v"], t["true_kw"]):
                        continue  # vanilla was correct -> not counted
                    for task, kw_key, def_key in [
                        ("rep_change",   "rp", "rp_def"),
                        ("false_belief", "fb", "fb_def"),
                    ]:
                        if t[def_key]:
                            continue  # skip default-fallback responses
                        if cmp_kw(t["true_kw"], t[kw_key]):
                            failures[task].append((env_dir.name, ep_dir.name, t, kw_key))
                        else:
                            successes[task].append((env_dir.name, ep_dir.name, t, kw_key))

        for label, bucket in [("SUCCESS (copy *hint* words)", successes),
                              ("FAILURE (parrot ground-truth keywords)", failures)]:
            print(f"\n  {label}")
            for task in ("rep_change", "false_belief"):
                items = bucket[task][: args.n]
                print(f"    {task}: {len(bucket[task])} total in dataset; showing {len(items)}")
                for env, ep, t, kw_key in items:
                    traj = trajectory_path(
                        args.model_seed, cond_dir.name, env, ep, t["turn"], task
                    )
                    print(f"      true: {t['true_kw']}")
                    print(f"      pred: {t[kw_key]}")
                    print(f"      combined_history: "
                          f"{cond_dir / env / ep / 'combined_history.json'}")
                    if traj is not None:
                        print(f"      trajectory:       {traj}")
                    print()


if __name__ == "__main__":
    main()
