#!/usr/bin/env python
"""Figures for the Polaris 3-model Decrypto cross-play (results/polaris_3model).

Deliberately NOT a 405-row dump: the outcome is overwhelmingly one-sided (the
lone code-cracker wins ~97% of games), so these figures show the shape of the
result rather than every row. Each figure carries a plain-language caption at
the bottom so it reads on its own, without needing the game's or statistics'
vocabulary. N is small (15 games per match-up), so rates are shown with the
range they could plausibly take.

Roles, in plain words:
  - encoder      = the clue-writer (writes three clues for a secret 3-digit code)
  - decoder      = the teammate    (must read the clues and recover the code)
  - interceptor  = the opponent    (a lone player trying to crack the same code)
The clue-writer and teammate are one team; they "win" only by getting through
the game without either giving the code away to the opponent or confusing each
other. Otherwise the opponent wins.

Run on a login node:
    $BASE/conda-envs/decrypto-serve/bin/python analysis/plot_polaris_3model.py
Outputs PNGs to results/polaris_3model/figures/.
"""
import csv
import math
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "polaris_3model", "experiment_summary.csv")
OUTDIR = os.path.join(REPO, "results", "polaris_3model", "figures")
MODELS = ["qwen2.5_72B", "qwen3_8b", "qwen3_4b"]
LABEL = {"qwen2.5_72B": "Qwen2.5-72B", "qwen3_8b": "Qwen3-8B", "qwen3_4b": "Qwen3-4B"}
C_SURV, C_INT, C_MIS, C_BOTH = "#2c7fb8", "#d95f02", "#e7c019", "#999999"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, returned as (point, lo, hi) in %."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, center - half), 100 * min(1, center + half)


def add_caption(fig, text, width=120, left=None):
    """Render a wrapped, plain-language caption along the bottom of a figure."""
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    n_lines = wrapped.count("\n") + 1
    bottom = 0.08 + 0.035 * n_lines
    if left is not None:
        fig.subplots_adjust(bottom=bottom, left=left)
    else:
        fig.subplots_adjust(bottom=bottom)
    fig.text(0.5, 0.015, wrapped, ha="center", va="bottom", fontsize=8.5,
             color="#333333", style="italic")


def load():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def fig_outcomes(rows):
    n = len(rows)
    counts = {k: sum(int(r[k]) for r in rows) for k in ["survived", "intercepts", "miscomms", "both"]}
    order = [("survived", "Team won", C_SURV),
             ("intercepts", "Code given away\nto the opponent", C_INT),
             ("miscomms", "Teammate\nmisread the clues", C_MIS),
             ("both", "Both at once", C_BOTH)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = range(len(order))
    bars = ax.bar(xs, [100 * counts[k] / n for k, _, _ in order],
                  color=[c for *_, c in order], width=0.62)
    for b, (k, _, _) in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{counts[k]} of {n}\n{100*counts[k]/n:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in order], fontsize=9)
    ax.set_ylabel("share of games (%)")
    ax.set_ylim(0, 100)
    ax.set_title("How 405 Decrypto games ended", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    add_caption(fig,
        "Across all 405 games the two-player team almost never won (11 times, about 1 in 37). "
        "When the team lost, it was usually because the clues gave the secret code away to the opponent "
        "(355 games) far more often than because the teammate misread the clues (33 games). In short, "
        "these models struggle to hint to a partner without also tipping off a listening opponent.")
    out = os.path.join(OUTDIR, "fig1_outcome_split.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_interceptor(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs, rates, los, his, anns = [], [], [], [], []
    for i, m in enumerate(MODELS):
        g = [r for r in rows if r["interceptor"] == m]
        w = sum(1 for r in g if int(r["survived"]) == 0)
        p, lo, hi = wilson(w, len(g))
        xs.append(i); rates.append(p); los.append(p - lo); his.append(hi - p)
        anns.append(f"{w} of {len(g)}")
    ax.errorbar(xs, rates, yerr=[los, his], fmt="o", color=C_INT, capsize=6,
                markersize=10, lw=2, ecolor="#555555")
    for x, r, a in zip(xs, rates, anns):
        ax.annotate(f"{r:.1f}%\n({a})", (x, r), textcoords="offset points", xytext=(16, -6), fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(88, 101)
    ax.set_ylabel("share of games the opponent won (%)")
    ax.set_xlabel("model playing the opponent (the lone code-cracker)")
    ax.set_title("How often the opponent won, by which model played it", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    add_caption(fig,
        "Each model took a turn as the opponent trying to crack the code, 135 games apiece. The largest "
        "model (Qwen2.5-72B) won every single one; the two smaller models let a handful of teams slip "
        "through. The vertical lines show how far each number could reasonably move given only 135 games, "
        "so read the gap as a hint that a stronger model defends better, not as a settled fact.")
    out = os.path.join(OUTDIR, "fig2_interceptor_ci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_team_marginals(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    width = 0.36
    for j, role in enumerate(["encoder", "decoder"]):
        xs, rates, los, his = [], [], [], []
        for i, m in enumerate(MODELS):
            g = [r for r in rows if r[role] == m]
            s = sum(int(r["survived"]) for r in g)
            p, lo, hi = wilson(s, len(g))
            xs.append(i + (j - 0.5) * width); rates.append(p); los.append(p - lo); his.append(hi - p)
        ax.bar(xs, rates, width=width, yerr=[los, his], capsize=4,
               color=(C_SURV if role == "encoder" else "#7fbf7f"),
               label=("as clue-writer" if role == "encoder" else "as teammate"),
               error_kw=dict(ecolor="#555555", lw=1.2))
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylabel("share of games the team won (%)")
    ax.set_xlabel("model writing or reading the clues")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False)
    ax.set_title("How often the team won, by who wrote and who read the clues", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    add_caption(fig,
        "Team wins are broken out by which model wrote the clues and which model read them, 135 games each. "
        "All three models land in roughly the same place, and the vertical ranges overlap heavily, which "
        "means the small differences are within chance. On this data the three models are about equally good "
        "in the cooperating roles, so these bars should not be read as a ranking.")
    out = os.path.join(OUTDIR, "fig3_team_marginals_ci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_heatmap(rows):
    """9 (encoder+decoder team) x 3 (interceptor) grid of team wins out of 15."""
    teams = [(e, d) for e in MODELS for d in MODELS]
    grid = [[sum(int(r["survived"]) for r in rows
                 if r["encoder"] == e and r["decoder"] == d and r["interceptor"] == i)
             for i in MODELS] for (e, d) in teams]
    vmax = max(max(row) for row in grid) or 1
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_yticks(range(9))
    ax.set_yticklabels([f"{LABEL[e]} + {LABEL[d]}" for (e, d) in teams], fontsize=8.5)
    ax.set_xlabel("opponent (the lone code-cracker)")
    ax.set_ylabel("team:  clue-writer  +  teammate")
    ax.set_title("Where the team actually won\n(games won out of 15 in each match-up)", fontsize=12)
    for r in range(9):
        for c in range(3):
            v = grid[r][c]
            ax.text(c, r, str(v), ha="center", va="center", fontsize=10,
                    color=("white" if v > vmax * 0.6 else "#333333"),
                    fontweight=("bold" if v > 0 else "normal"))
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("team wins (out of 15)")
    add_caption(fig,
        "Every square is one match-up: a team (a clue-writer plus a teammate, one per row) against a single "
        "opponent (one per column), played over 15 games. The number is how many of those 15 the team won. "
        "Nearly every square is zero, and the entire right-hand story is the leftmost column: the team never "
        "won a single game when the largest model (Qwen2.5-72B) was the opponent cracking the code.",
        width=118, left=0.26)
    out = os.path.join(OUTDIR, "fig4_team_vs_opponent_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load()
    assert len(rows) == 405, f"expected 405 rows, got {len(rows)}"
    for fn in (fig_outcomes, fig_interceptor, fig_team_marginals, fig_heatmap):
        print("wrote", os.path.relpath(fn(rows), REPO))


if __name__ == "__main__":
    main()
