#!/usr/bin/env python3
"""Decompose Gopnik ToM success rates into genuine perspective-taking
vs ['no_keyword']*4 default-fallback artifacts.

The runner (src/runner.py) substitutes ['no_keyword']*4 for any Gopnik query
where the model fails to produce valid JSON after 10 retries
(see src/agents/role_client.py:444-471). Because ['no_keyword']*4 is never
equal to the true keywords, default-fallback responses inflate weak_success
metrics; because two defaults are equal to each other, they also inflate
strong_success and self_other_consistency. This script:

  1. Re-creates the runner's per-episode-then-mean metric (matches the CSV).
  2. Splits each headline pass count into "default-fallback artifact" vs
     "genuine model output".
  3. Reports the parroting rate of the true keywords among parsed answers.

Usage:
    python analyze_gopnik.py <results_dir>
e.g.
    python analyze_gopnik.py results/figure_4_tom_gopnik_qwen3_4b_199

Expects:
    <results_dir>/model_seed*/
        <encoder>_<decoder>_<interceptor>_*/
            env_seed*/episode_*/combined_history.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# What the runner returns when JSON parsing fails after max retries.
DEFAULT_KW = ["no_keyword"] * 4


# ----------------------------- parsing helpers -----------------------------

def parse_kw(response):
    """Parse a Gopnik keyword response. Returns (kw_list, is_default).

    The runner substitutes ['no_keyword']*4 on parse failure. We mirror that
    here, but track whether the substitution happened so we can decompose.
    """
    if not response or not isinstance(response, str):
        return DEFAULT_KW, True
    if "Maximum attempts reached" in response:
        return DEFAULT_KW, True
    m = re.search(r"ANSWER:\s*(\{.*\})", response, re.DOTALL)
    if not m:
        return DEFAULT_KW, True
    try:
        d = json.loads(m.group(1))
    except Exception:
        return DEFAULT_KW, True
    kws = d.get("keywords")
    if not isinstance(kws, list) or len(kws) != 4:
        return DEFAULT_KW, True
    return [str(x) for x in kws], False


def parse_true(prompt):
    m = re.search(
        r"Keywords:\s*\n\s*1\.\s*(\w+)\s*\n\s*2\.\s*(\w+)\s*\n\s*3\.\s*(\w+)\s*\n\s*4\.\s*(\w+)",
        prompt or "",
    )
    return [m.group(i) for i in range(1, 5)] if m else None


def cmp_kw(l1, l2):
    """Case-insensitive 4-tuple equality. Mirrors src/runner.py:compare_kw_lists."""
    return (
        bool(l1)
        and bool(l2)
        and len(l1) == len(l2)
        and all(str(a).lower() == str(b).lower() for a, b in zip(l1, l2))
    )


def qtype(content):
    if "Given this hint history, what do you think are the four secret keywords" in content:
        return "vanilla"
    if "Before the keywords were revealed to you" in content:
        return "rep_change"
    if "Another player, the Second Interceptor" in content:
        return "false_belief"
    return None


# --------------------------- trial extraction ---------------------------

def episode_trials(combined_history):
    """Return one dict per turn that has all 3 Gopnik queries, with parsed
    + default flags + true keywords."""
    by_turn = {}
    for i, msg in enumerate(combined_history):
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if "[INTERCEPTOR] Turn " not in c:
            continue
        q = qtype(c)
        if q is None:
            continue
        tm = re.search(r"\[INTERCEPTOR\] Turn (\d+):", c)
        if not tm or i + 1 >= len(combined_history):
            continue
        resp = combined_history[i + 1]
        if resp.get("role") != "assistant":
            continue
        kws, is_def = parse_kw(resp.get("content", "").replace("[INTERCEPTOR] ", ""))
        true_kw = parse_true(c)
        by_turn.setdefault(int(tm.group(1)), {})[q] = {
            "kws": kws,
            "is_default": is_def,
            "true_kw": true_kw,
        }
    out = []
    for turn, qs in by_turn.items():
        if not all(k in qs for k in ("vanilla", "rep_change", "false_belief")):
            continue
        true_kw = qs["rep_change"]["true_kw"] or qs["false_belief"]["true_kw"]
        if not true_kw:
            continue
        out.append({
            "turn": turn, "true_kw": true_kw,
            "v":  qs["vanilla"]["kws"],     "v_def":  qs["vanilla"]["is_default"],
            "rp": qs["rep_change"]["kws"],  "rp_def": qs["rep_change"]["is_default"],
            "fb": qs["false_belief"]["kws"],"fb_def": qs["false_belief"]["is_default"],
        })
    return out


def collect_condition(cond_dir):
    """Walk env_seed*/episode_*/ under one (encoder_decoder_interceptor) folder.
    Returns dict: env_name -> {"trials": [...], "ep_metrics": [{...}, ...]}"""
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

            num_failed_kw_pred = 0
            pass_strong_RP = pass_strong_FB = 0
            pass_weak_RP = pass_weak_FB = pass_SO = 0
            for t in trials:
                if cmp_kw(t["v"], t["true_kw"]):
                    continue  # vanilla was correct -> not counted
                num_failed_kw_pred += 1
                if cmp_kw(t["v"], t["rp"]):    pass_strong_RP += 1
                if cmp_kw(t["v"], t["fb"]):    pass_strong_FB += 1
                if not cmp_kw(t["true_kw"], t["rp"]): pass_weak_RP += 1
                if not cmp_kw(t["true_kw"], t["fb"]): pass_weak_FB += 1
                if cmp_kw(t["rp"], t["fb"]):   pass_SO += 1
            if num_failed_kw_pred > 0:
                ep_metrics.append({
                    "n_failed":   num_failed_kw_pred,
                    "weak_rp":    pass_weak_RP / num_failed_kw_pred,
                    "weak_fb":    pass_weak_FB / num_failed_kw_pred,
                    "strong_rp":  pass_strong_RP / num_failed_kw_pred,
                    "strong_fb":  pass_strong_FB / num_failed_kw_pred,
                    "so":         pass_SO / num_failed_kw_pred,
                })
        by_env[env_dir.name] = {"trials": env_trials, "ep_metrics": ep_metrics}
    return by_env


# --------------------------- reporting ---------------------------

def runner_style_table(label, by_env):
    """Section 1: Per-env CSV-matching numbers (mean over per-episode rates)."""
    print(f"\n--- 1. Runner-style per-env (matches CSV) [{label}] ---")
    print(f"{'env':<10} {'n_eps':>5} {'tot_valid':>9} {'weak_rp':>8} "
          f"{'weak_fb':>8} {'strong_rp':>10} {'strong_fb':>10} {'SO':>6}")
    for env, d in by_env.items():
        eps = d["ep_metrics"]
        if not eps:
            print(f"{env:<10} {0:>5} {0:>9} {'—':>8} {'—':>8} {'—':>10} {'—':>10} {'—':>6}")
            continue
        total = sum(e["n_failed"] for e in eps)
        wr = float(np.mean([e["weak_rp"]   for e in eps]))
        wf = float(np.mean([e["weak_fb"]   for e in eps]))
        sr = float(np.mean([e["strong_rp"] for e in eps]))
        sf = float(np.mean([e["strong_fb"] for e in eps]))
        so = float(np.mean([e["so"]        for e in eps]))
        print(f"{env:<10} {len(eps):>5} {total:>9} {wr:>8.4f} {wf:>8.4f} "
              f"{sr:>10.4f} {sf:>10.4f} {so:>6.4f}")


def decompose(label, by_env):
    """Section 2 (artifact decomposition) + Section 3 (all-three-parsed subset)."""
    trials = sum((d["trials"] for d in by_env.values()), [])
    valid = [t for t in trials if not cmp_kw(t["v"], t["true_kw"])]
    n = len(valid)
    if n == 0:
        print(f"\n--- 2/3. No valid trials for {label} ---")
        return

    n_v_def = sum(1 for t in valid if t["v_def"])
    print(f"\n--- 2. Decomposition: artifact vs genuine [{label}] ---")
    print(f"  total trials = {len(trials)}, valid = {n}")
    print(f"  vanilla DEFAULTED (parse fail): {n_v_def}  ({100*n_v_def/n:.1f}%)")
    print(f"  vanilla parsed real answer:     {n - n_v_def}  ({100*(n - n_v_def)/n:.1f}%)")

    def report_strong(name, key, def_key):
        passes = [t for t in valid if cmp_kw(t["v"], t[key])]
        both_def = [t for t in passes if t["v_def"] and t[def_key]]
        genuine  = [t for t in passes if not t["v_def"] and not t[def_key]]
        print(f"\n  STRONG {name}: {len(passes)}/{n} = {100*len(passes)/n:.1f}%")
        print(f"     ├ both v & {key} defaulted (artifact): "
              f"{len(both_def)} ({100*len(both_def)/n:.1f}%)")
        print(f"     └ genuine (both parsed, model held belief): "
              f"{len(genuine)} ({100*len(genuine)/n:.1f}%)")

    def report_weak(name, key, def_key):
        passes = [t for t in valid if not cmp_kw(t["true_kw"], t[key])]
        defaulted = [t for t in passes if t[def_key]]
        genuine   = [t for t in passes if not t[def_key]]
        print(f"\n  WEAK {name}: {len(passes)}/{n} = {100*len(passes)/n:.1f}%")
        print(f"     ├ {key} defaulted (auto-pass: 'no_keyword' != truth): "
              f"{len(defaulted)} ({100*len(defaulted)/n:.1f}%)")
        print(f"     └ genuine (parsed, didn't parrot): "
              f"{len(genuine)} ({100*len(genuine)/n:.1f}%)")

    report_strong("RP", "rp", "rp_def")
    report_strong("FB", "fb", "fb_def")
    report_weak("RP",   "rp", "rp_def")
    report_weak("FB",   "fb", "fb_def")

    all_parsed = [t for t in valid if not (t["v_def"] or t["rp_def"] or t["fb_def"])]
    print(f"\n--- 3. Conditional on all three parsed (no defaults) [{label}] ---")
    print(f"  ALL THREE PARSED: {len(all_parsed)} ({100*len(all_parsed)/n:.1f}% of valid)")
    if not all_parsed:
        return
    n2 = len(all_parsed)
    gs_rp = sum(1 for t in all_parsed if cmp_kw(t["v"], t["rp"]))
    gs_fb = sum(1 for t in all_parsed if cmp_kw(t["v"], t["fb"]))
    gw_rp = sum(1 for t in all_parsed if not cmp_kw(t["true_kw"], t["rp"]))
    gw_fb = sum(1 for t in all_parsed if not cmp_kw(t["true_kw"], t["fb"]))
    print(f"     genuine strong RP: {gs_rp}/{n2} = {100*gs_rp/n2:.1f}%")
    print(f"     genuine strong FB: {gs_fb}/{n2} = {100*gs_fb/n2:.1f}%")
    print(f"     genuine weak RP : {gw_rp}/{n2} = {100*gw_rp/n2:.1f}%")
    print(f"     genuine weak FB : {gw_fb}/{n2} = {100*gw_fb/n2:.1f}%")


def parroting(label, by_env):
    """Section 4: Among parsed (non-default) rep_change/false_belief responses,
    what fraction are verbatim copies of the revealed true keywords?"""
    trials = sum((d["trials"] for d in by_env.values()), [])
    valid = [t for t in trials if not cmp_kw(t["v"], t["true_kw"])]
    rp_parsed = [t for t in valid if not t["rp_def"]]
    fb_parsed = [t for t in valid if not t["fb_def"]]
    rp_parrot = sum(1 for t in rp_parsed if cmp_kw(t["true_kw"], t["rp"]))
    fb_parrot = sum(1 for t in fb_parsed if cmp_kw(t["true_kw"], t["fb"]))
    print(f"\n--- 4. Ground-truth parroting among parsed answers [{label}] ---")
    if rp_parsed:
        print(f"  rep_change   parsed: {len(rp_parsed):4d}, "
              f"parroted truth: {rp_parrot:4d} ({100*rp_parrot/len(rp_parsed):.1f}%)")
    if fb_parsed:
        print(f"  false_belief parsed: {len(fb_parsed):4d}, "
              f"parroted truth: {fb_parrot:4d} ({100*fb_parrot/len(fb_parsed):.1f}%)")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path,
                    help="e.g. results/figure_4_tom_gopnik_qwen3_4b_199")
    ap.add_argument("--model-seed", default="model_seed0",
                    help="Sub-folder under results_dir (default: model_seed0)")
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
        parroting(cond_dir.name, by_env)


if __name__ == "__main__":
    main()
