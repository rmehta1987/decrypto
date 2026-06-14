#!/usr/bin/env python
"""Figures for the Polaris 3-model Decrypto cross-play (results/polaris_3model).

Deliberately NOT a 405-row dump: the outcome is overwhelmingly one-sided
(defense wins 97%), so these figures show (1) the headline outcome split,
(2) the only place model identity plausibly matters — the interceptor — with
95% Wilson CIs, and (3) the encoder/decoder marginals whose CIs overlap, i.e.
the cooperating roles are statistically indistinguishable. N is small
(15 env-seeds per combination), so every rate carries a CI by design.

Run on a login node:
    $BASE/conda-envs/decrypto-serve/bin/python analysis/plot_polaris_3model.py
Outputs PNGs to results/polaris_3model/figures/.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "polaris_3model", "experiment_summary.csv")
OUTDIR = os.path.join(REPO, "results", "polaris_3model", "figures")
MODELS = ["qwen2.5_72B", "qwen3_8b", "qwen3_4b"]
LABEL = {"qwen2.5_72B": "Qwen2.5-72B", "qwen3_8b": "Qwen3-8B", "qwen3_4b": "Qwen3-4B"}
# colorblind-safe-ish palette
C_SURV, C_INT, C_MIS, C_BOTH = "#2c7fb8", "#d95f02", "#e7c019", "#999999"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, returned as (lo, hi) in %."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, center - half), 100 * min(1, center + half)


def load():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def fig_outcomes(rows):
    """Fig 1: overall terminal-outcome split (the headline)."""
    n = len(rows)
    counts = {k: sum(int(r[k]) for r in rows) for k in ["survived", "intercepts", "miscomms", "both"]}
    order = [("survived", "Team survived\n(team win)", C_SURV),
             ("intercepts", "Intercepted\n(hints too clear)", C_INT),
             ("miscomms", "Miscommunicated\n(hints too obscure)", C_MIS),
             ("both", "Both", C_BOTH)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = range(len(order))
    vals = [100 * counts[k] / n for k, _, _ in order]
    bars = ax.bar(xs, vals, color=[c for *_, c in order], width=0.62)
    for b, (k, _, _) in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{counts[k]}/{n}\n{100*counts[k]/n:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in order], fontsize=9)
    ax.set_ylabel("share of games (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Decrypto 3-model cross-play: how 405 games ended\n"
                 "Teams win only 2.7%; defense wins 97.3% — and failure is overwhelmingly by transparency",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig1_outcome_split.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_interceptor(rows):
    """Fig 2: defense-win rate by interceptor model, with 95% Wilson CIs."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, rates, los, his, anns = [], [], [], [], []
    for i, m in enumerate(MODELS):
        g = [r for r in rows if r["interceptor"] == m]
        w = sum(1 for r in g if int(r["survived"]) == 0)
        p, lo, hi = wilson(w, len(g))
        xs.append(i); rates.append(p); los.append(p - lo); his.append(hi - p)
        anns.append(f"{w}/{len(g)}")
    ax.errorbar(xs, rates, yerr=[los, his], fmt="o", color=C_INT, capsize=6,
                markersize=9, lw=2, ecolor="#555555")
    for x, r, a in zip(xs, rates, anns):
        ax.annotate(f"{r:.1f}%\n({a})", (x, r), textcoords="offset points",
                    xytext=(14, -4), fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(88, 101)
    ax.set_ylabel("defense-win rate (%)  [95% Wilson CI]")
    ax.set_xlabel("model in the Interceptor role")
    ax.set_title("The only suggestive model effect: bigger interceptor = stronger defense\n"
                 "Qwen2.5-72B was never beaten (135/135); N=135 each, CIs still overlap",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig2_interceptor_ci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_team_marginals(rows):
    """Fig 3: encoder & decoder team-survival marginals with CIs (the overlap = indistinguishable)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.36
    for j, role in enumerate(["encoder", "decoder"]):
        xs, rates, los, his = [], [], [], []
        for i, m in enumerate(MODELS):
            g = [r for r in rows if r[role] == m]
            s = sum(int(r["survived"]) for r in g)
            p, lo, hi = wilson(s, len(g))
            xs.append(i + (j - 0.5) * width); rates.append(p); los.append(p - lo); his.append(hi - p)
        ax.bar(xs, rates, width=width, yerr=[los, his], capsize=4,
               color=(C_SURV if role == "encoder" else "#7fbf7f"), label=f"as {role}",
               error_kw=dict(ecolor="#555555", lw=1.2))
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylabel("team-survival rate (%)  [95% Wilson CI]")
    ax.set_xlabel("model in the cooperating (Encoder / Decoder) role")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False)
    ax.set_title("Cooperating roles are statistically indistinguishable\n"
                 "All CIs overlap (N=135 each) — do NOT read these as a model ranking",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig3_team_marginals_ci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load()
    assert len(rows) == 405, f"expected 405 rows, got {len(rows)}"
    outs = [fig_outcomes(rows), fig_interceptor(rows), fig_team_marginals(rows)]
    for o in outs:
        print("wrote", os.path.relpath(o, REPO))


if __name__ == "__main__":
    main()
