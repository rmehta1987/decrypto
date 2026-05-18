#!/usr/bin/env python3
"""Decompose Piaget ToM (Alice predicts Eve's guess) success rates into
genuine perspective-taking vs ['0-0-0'] default-fallback artifacts.

The Piaget setup tests whether the encoder (Alice) can predict the
interceptor's (Eve's) guess. The runner (src/runner.py:187-209, 385-392)
computes three per-episode rates:

  * avg_predicted_success      = (Alice's prediction == Eve's guess)        / turns
  * avg_predicted_success_on_intercept
                                = (Alice's prediction == Eve's guess)        / # intercepts
                                  (only when there's at least one intercept)
  * avg_prediction_is_code     = (Alice's prediction == actual code)        / turns
                                  This is the egocentric failure mode --
                                  Alice predicting Eve will guess the right
                                  code (i.e. ascribing her own knowledge).

When JSON parsing fails after 10 retries, the runner substitutes
['guess': '0-0-0'] -> code = [0, 0, 0]
(see src/agents/role_client.py:450-453, 470-471). Effects on the metrics:

  * predicts_eve (avg_predicted_success):
      * Alice [0,0,0] vs Eve real guess  -> miss   (DEFLATES rate)
      * Alice [0,0,0] vs Eve [0,0,0]     -> "hit"  (INFLATES rate, artifact)
  * predicts_code:
      * Alice [0,0,0] vs real code (digits 1-4) -> always miss (DEFLATES rate)
  * predicts_eve_on_intercept (numerator only counted when Eve actually
    intercepts -> Eve != [0,0,0]):
      * Alice [0,0,0] vs intercepting Eve guess  -> miss (DEFLATES rate)

So in this setup (Alice = qwen3_4b, frequent defaults; Eve = llama3.1_70B,
near-zero defaults), defaults DEFLATE all three metrics. The "egocentric"
predicts_code rate, in particular, is a LOWER bound on Alice's true
egocentric tendency.

This script:
  1. Reproduces the per-env CSV numbers (sanity check vs runner methodology).
  2. Reports rate of Alice JSON-parse defaults.
  3. Re-computes each metric conditional on Alice having produced a
     parsed (non-default) prediction.

Usage:
    python analyze_piaget.py <results_dir>
e.g.
    python analyze_piaget.py results/figure_4_tom_piaget_qwen3_4b_199
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_GUESS = [0, 0, 0]


# ----------------------------- parsing helpers -----------------------------

def parse_guess(response):
    """Parse a code guess. Returns (guess_list, is_default).

    The runner substitutes [0, 0, 0] on JSON-parse failure.
    """
    if not response or not isinstance(response, str):
        return DEFAULT_GUESS, True
    if "Maximum attempts reached" in response:
        return DEFAULT_GUESS, True
    # Find the LAST ANSWER occurrence (in case of repeated mentions in CoT).
    matches = list(re.finditer(r'ANSWER:\s*(\{.*?\})', response, re.DOTALL))
    if not matches:
        return DEFAULT_GUESS, True
    raw = matches[-1].group(1)
    try:
        d = json.loads(raw)
    except Exception:
        return DEFAULT_GUESS, True
    g = d.get("guess")
    if not isinstance(g, str):
        return DEFAULT_GUESS, True
    parts = g.split("-")
    if len(parts) != 3:
        return DEFAULT_GUESS, True
    try:
        return [int(x) for x in parts], False
    except Exception:
        return DEFAULT_GUESS, True


CODE_RE = re.compile(
    r"The code is\s*(\d-\d-\d)", re.IGNORECASE
)
PREDICT_PROMPT_TAG = "What do you predict will be the guess of the interceptor"


def parse_code_from_prompt(content):
    if not content:
        return None
    m = CODE_RE.search(content)
    if not m:
        return None
    return [int(x) for x in m.group(1).split("-")]


def is_predict_prompt(c):
    return PREDICT_PROMPT_TAG in (c or "")


def is_interceptor_prompt(c):
    return "[INTERCEPTOR]" in (c or "") and "What is your guess for the three-digit code" in (c or "")


def is_encoder_hint_prompt(c):
    """The first encoder prompt of each turn has the actual code."""
    if not c:
        return False
    return c.startswith("[ENCODER]") and CODE_RE.search(c) is not None


# --------------------------- trial extraction ---------------------------

def episode_trials(combined_history):
    """Return list of dicts, one per turn that has all three things:
       actual code, Alice's prediction, Eve's guess."""
    trials = []
    current = {}  # accumulating fields for the current turn

    for i, msg in enumerate(combined_history):
        role = msg.get("role")
        c = msg.get("content", "")
        # Encoder hint prompt -> records actual code, starts a new turn
        if role == "user" and is_encoder_hint_prompt(c):
            if current.get("code") is not None:
                # If we accumulated a previous turn but it's incomplete, drop.
                pass
            code = parse_code_from_prompt(c)
            current = {"code": code}
            continue
        # Encoder prediction prompt -> next assistant is Alice's prediction
        if role == "user" and is_predict_prompt(c):
            current["awaiting"] = "alice"
            continue
        # Interceptor prompt -> next assistant is Eve's guess
        if role == "user" and is_interceptor_prompt(c):
            current["awaiting"] = "eve"
            continue
        if role == "assistant":
            tag = current.pop("awaiting", None)
            if tag == "alice":
                guess, is_def = parse_guess(c.replace("[ENCODER] ", ""))
                current["alice"] = guess
                current["alice_def"] = is_def
            elif tag == "eve":
                guess, is_def = parse_guess(c.replace("[INTERCEPTOR] ", ""))
                current["eve"] = guess
                current["eve_def"] = is_def
                # End of turn for our purposes (turn summary follows).
                if all(k in current for k in ("code", "alice", "eve")):
                    trials.append(current)
                current = {}
            continue
    return trials


def collect_condition(cond_dir):
    """Walk env_seed*/episode_*/ under one (encoder_decoder_interceptor) folder."""
    by_env = {}
    for env_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
        env_trials, ep_metrics = [], []
        for ep_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            ch = ep_dir / "combined_history.json"
            if not ch.exists():
                continue
            with open(ch) as f:
                trials = episode_trials(json.load(f))
            for t in trials:
                t["env"], t["ep"] = env_dir.name, ep_dir.name
            env_trials.extend(trials)

            episode_turns = len(trials)
            num_intercepts = sum(1 for t in trials if t["eve"] == t["code"])
            alice_predicted = sum(1 for t in trials if t["alice"] == t["eve"])
            alice_pred_on_int = sum(
                1 for t in trials if t["alice"] == t["eve"] and t["eve"] == t["code"]
            )
            alice_predicted_code = sum(1 for t in trials if t["alice"] == t["code"])
            if episode_turns > 0:
                ep_metrics.append({
                    "episode_turns":         episode_turns,
                    "num_intercepts":        num_intercepts,
                    "predicts_eve":          alice_predicted / episode_turns,
                    "predicts_code":         alice_predicted_code / episode_turns,
                    "predicts_eve_on_intercept": (
                        alice_pred_on_int / num_intercepts
                        if num_intercepts > 0 else None
                    ),
                })
        by_env[env_dir.name] = {"trials": env_trials, "ep_metrics": ep_metrics}
    return by_env


# --------------------------- reporting ---------------------------

def runner_style_table(label, by_env):
    print(f"\n--- 1. Runner-style per-env (matches CSV) [{label}] ---")
    print(f"{'env':<10} {'n_eps':>5} {'turns':>6} "
          f"{'predicts_eve':>13} {'pred_code':>10} {'pred_eve|int':>13}")
    for env, d in by_env.items():
        eps = d["ep_metrics"]
        if not eps:
            continue
        n_turns = sum(e["episode_turns"] for e in eps)
        avg_pe = float(np.mean([e["predicts_eve"]  for e in eps]))
        avg_pc = float(np.mean([e["predicts_code"] for e in eps]))
        on_int = [e["predicts_eve_on_intercept"]
                  for e in eps if e["predicts_eve_on_intercept"] is not None]
        avg_oi = float(np.mean(on_int)) if on_int else float("nan")
        print(f"{env:<10} {len(eps):>5} {n_turns:>6} "
              f"{avg_pe:>13.4f} {avg_pc:>10.4f} {avg_oi:>13.4f}")


def decompose(label, by_env):
    """Decomposition over all valid trials in the condition."""
    trials = sum((d["trials"] for d in by_env.values()), [])
    n = len(trials)
    if n == 0:
        print(f"\nNo trials for {label}")
        return

    n_a_def = sum(1 for t in trials if t["alice_def"])
    n_e_def = sum(1 for t in trials if t["eve_def"])
    print(f"\n--- 2. Decomposition: artifact vs genuine [{label}] ---")
    print(f"  total trials = {n}")
    print(f"  Alice DEFAULTED (parse fail): {n_a_def}  ({100*n_a_def/n:.1f}%)")
    print(f"  Eve   DEFAULTED (parse fail): {n_e_def}  ({100*n_e_def/n:.1f}%)")

    # predicts_eve: alice == eve
    pe_pass = [t for t in trials if t["alice"] == t["eve"]]
    pe_both_def = [t for t in pe_pass if t["alice_def"] and t["eve_def"]]
    pe_genuine  = [t for t in pe_pass if not t["alice_def"] and not t["eve_def"]]
    pe_other    = [t for t in pe_pass if t["alice_def"] != t["eve_def"]]
    print(f"\n  PREDICTS_EVE passes: {len(pe_pass)}/{n} = {100*len(pe_pass)/n:.1f}%")
    print(f"     ├ both Alice & Eve defaulted (artifact): {len(pe_both_def)} "
          f"({100*len(pe_both_def)/n:.1f}%)")
    print(f"     ├ genuine (both produced parsed guesses):  {len(pe_genuine)} "
          f"({100*len(pe_genuine)/n:.1f}%)")
    print(f"     └ mixed (one defaulted, other matched):    {len(pe_other)} "
          f"({100*len(pe_other)/n:.1f}%)")

    # predicts_code: alice == code  (defaults can never satisfy this since code is 1-4 digits)
    pc_pass = [t for t in trials if t["alice"] == t["code"]]
    pc_def = [t for t in pc_pass if t["alice_def"]]  # should be 0
    pc_real = [t for t in pc_pass if not t["alice_def"]]
    print(f"\n  PREDICTS_CODE passes (egocentric failure mode): "
          f"{len(pc_pass)}/{n} = {100*len(pc_pass)/n:.1f}%")
    print(f"     ├ artifact (Alice default == code): {len(pc_def)} ({100*len(pc_def)/n:.1f}%)")
    print(f"     └ genuine: {len(pc_real)} ({100*len(pc_real)/n:.1f}%)")

    # predicts_eve_on_intercept: among Eve real-intercepts, alice == eve
    intercepts = [t for t in trials if t["eve"] == t["code"]]
    int_pass = [t for t in intercepts if t["alice"] == t["eve"]]
    int_pass_def = [t for t in int_pass if t["alice_def"]]  # should be 0
    int_pass_real = [t for t in int_pass if not t["alice_def"]]
    print(f"\n  PREDICTS_EVE_ON_INTERCEPT (per-trial, not per-episode):")
    print(f"     intercepts: {len(intercepts)}; Alice predicted Eve correctly on "
          f"those: {len(int_pass)} ({100*len(int_pass)/max(len(intercepts),1):.1f}%)")
    print(f"     ├ artifact (Alice default): {len(int_pass_def)}")
    print(f"     └ genuine: {len(int_pass_real)}")


def conditional_on_alice_parsed(label, by_env):
    """Section 3: rate conditional on Alice producing a parsed (non-default) prediction.

    This strips the deflation caused by Alice JSON-parse failures.
    """
    trials = sum((d["trials"] for d in by_env.values()), [])
    parsed = [t for t in trials if not t["alice_def"]]
    n = len(parsed)
    print(f"\n--- 3. Conditional on Alice parsed (no default) [{label}] ---")
    print(f"  ALICE PARSED: {n} ({100*n/max(len(trials),1):.1f}% of trials)")
    if n == 0:
        return
    pe = sum(1 for t in parsed if t["alice"] == t["eve"])
    pc = sum(1 for t in parsed if t["alice"] == t["code"])
    intercepts = [t for t in parsed if t["eve"] == t["code"]]
    pe_int = sum(1 for t in intercepts if t["alice"] == t["eve"])
    print(f"     genuine predicts_eve:               {pe}/{n} = {100*pe/n:.1f}%")
    print(f"     genuine predicts_code (egocentric): {pc}/{n} = {100*pc/n:.1f}%")
    if intercepts:
        print(f"     genuine predicts_eve | intercept:   "
              f"{pe_int}/{len(intercepts)} = {100*pe_int/len(intercepts):.1f}%")


def both_parsed_breakdown(label, by_env):
    """Section 4: breakdown when BOTH Alice and Eve parsed — pure perspective taking.

    Among those, what fraction did Alice match Eve? And how often was she
    egocentric (predicted the actual code)?
    """
    trials = sum((d["trials"] for d in by_env.values()), [])
    both = [t for t in trials if not t["alice_def"] and not t["eve_def"]]
    n = len(both)
    print(f"\n--- 4. Conditional on BOTH Alice & Eve parsed [{label}] ---")
    print(f"  BOTH PARSED: {n} ({100*n/max(len(trials),1):.1f}% of trials)")
    if n == 0:
        return
    pe = sum(1 for t in both if t["alice"] == t["eve"])
    pc = sum(1 for t in both if t["alice"] == t["code"])
    intercepts = [t for t in both if t["eve"] == t["code"]]
    pe_int = sum(1 for t in intercepts if t["alice"] == t["eve"])
    print(f"     genuine predicts_eve:               {pe}/{n} = {100*pe/n:.1f}%")
    print(f"     genuine predicts_code (egocentric): {pc}/{n} = {100*pc/n:.1f}%")
    if intercepts:
        print(f"     genuine predicts_eve | intercept:   "
              f"{pe_int}/{len(intercepts)} = {100*pe_int/len(intercepts):.1f}%")
    # Direction of error: Alice's egocentric predictions vs. correct theory-of-mind
    eve_eq_code = [t for t in both if t["eve"] == t["code"]]
    eve_neq_code = [t for t in both if t["eve"] != t["code"]]
    egocentric_when_eve_neq = sum(
        1 for t in eve_neq_code if t["alice"] == t["code"] and t["alice"] != t["eve"]
    )
    print(f"     among {len(eve_neq_code)} trials where Eve guessed wrong, "
          f"Alice predicted the actual code (egocentric error): "
          f"{egocentric_when_eve_neq} "
          f"({100*egocentric_when_eve_neq/max(len(eve_neq_code),1):.1f}%)")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path,
                    help="e.g. results/figure_4_tom_piaget_qwen3_4b_199")
    ap.add_argument("--model-seed", default="model_seed0")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Substring filter on condition folder names "
                         "(default: all qwen3_4b folders)")
    args = ap.parse_args()

    base = args.results_dir / args.model_seed
    if not base.is_dir():
        print(f"Not a directory: {base}", file=sys.stderr)
        sys.exit(2)

    cond_dirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if args.include is not None:
        cond_dirs = [p for p in cond_dirs if any(s in p.name for s in args.include)]
    else:
        cond_dirs = [p for p in cond_dirs if "qwen3_4b" in p.name]
    if not cond_dirs:
        print("No condition folders found.", file=sys.stderr)
        sys.exit(2)

    for cond_dir in cond_dirs:
        print("\n" + "=" * 80)
        print(f"CONDITION: {cond_dir.name}")
        print("=" * 80)
        by_env = collect_condition(cond_dir)
        runner_style_table(cond_dir.name, by_env)
        decompose(cond_dir.name, by_env)
        conditional_on_alice_parsed(cond_dir.name, by_env)
        both_parsed_breakdown(cond_dir.name, by_env)


if __name__ == "__main__":
    main()
