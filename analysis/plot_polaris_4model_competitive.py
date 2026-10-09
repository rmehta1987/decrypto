#!/usr/bin/env python
"""Figures for the Polaris 4-model run of the paper's figure-3 COMPETITIVE setup
(results/polaris_4model_competitive_s{0,1,2}), and its comparison to the paper.

The setup: each model plays as a team (it writes the clues AND reads them, i.e.
encoder == decoder) against each model as the lone opponent trying to crack the
code (the interceptor, "Eve"). 4 teams x 4 opponents = 16 match-ups, each played
32 times on each of 3 seeds = 1,536 games, temperature 0.6, as in
config/paper/figure_3_competitive.yaml.

Metrics follow the paper's own analysis (analysis/comp_cross_4heatmaps.py):
  - interception share = (intercepts + both) / 32 per seed; mean +- standard
    error of the mean across the 3 seeds;
  - average turns per game, aggregated the same way.

The paper's numbers are transcribed from arXiv:2506.20664v1 (Lupu, Willi,
Foerster 2025), rendered at 300 dpi: Figure 3 bottom row (p. 7), Figure 7
(p. 18), Table 1 (p. 8). The figure-3 values are printed in the paper's
heatmap cells; none of them appears in the paper's text.

Run on a login node:
    $BASE/conda-envs/decrypto-serve/bin/python analysis/plot_polaris_4model_competitive.py
Outputs PNGs to results/polaris_4model_competitive/figures/ and prints the
tables used in polaris_4model_competitive_results.md.
"""
import csv
import math
import os
import statistics as st
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVS = [os.path.join(REPO, "results", f"polaris_4model_competitive_s{s}", "experiment_summary.csv") for s in (0, 1, 2)]
OUTDIR = os.path.join(REPO, "results", "polaris_4model_competitive", "figures")
EPISODES = 32

MODELS = ["qwen2.5_72B", "qwen3_8b", "qwen3_4b", "claude4.6_cli"]
LABEL = {"qwen2.5_72B": "Qwen2.5-72B", "qwen3_8b": "Qwen3-8B", "qwen3_4b": "Qwen3-4B",
         "claude4.6_cli": "Claude Sonnet 4.6"}

# --- Paper values (arXiv:2506.20664v1). Rows = team (encoder = decoder),
# columns = interceptor, both in PAPER_AGENTS order. (mean, standard error).
PAPER_AGENTS = ["Word2Vec", "GloVe", "Llama-3.1-8B", "Llama-3.1-70B", "GPT-4o", "DS-R1-Distill-32B"]
PAPER_INTERCEPT = [  # Figure 3, bottom left: proportion of games ending in an intercept
    [(1.00, .00), (1.00, .00), (0.48, .05), (0.85, .01), (0.99, .01), (0.88, .00)],
    [(1.00, .00), (1.00, .00), (0.38, .05), (0.85, .03), (0.94, .00), (0.92, .03)],
    [(0.88, .04), (0.95, .01), (0.71, .08), (0.89, .01), (0.92, .01), (0.94, .02)],
    [(0.99, .01), (0.98, .02), (0.71, .02), (0.98, .01), (1.00, .00), (0.99, .01)],
    [(1.00, .00), (1.00, .00), (0.77, .01), (0.96, .01), (1.00, .00), (1.00, .00)],
    [(0.97, .00), (0.96, .01), (0.71, .05), (0.95, .04), (0.98, .01), (0.95, .02)],
]
PAPER_TURNS = [  # Figure 3, bottom right: average number of turns per episode
    [(4.34, .05), (4.59, .07), (7.00, .09), (6.03, .11), (4.66, .16), (5.44, .08)],
    [(4.45, .09), (4.25, .05), (7.28, .13), (5.38, .06), (4.92, .04), (5.30, .14)],
    [(4.25, .02), (4.25, .08), (5.17, .26), (3.80, .22), (3.46, .06), (3.88, .03)],
    [(4.33, .06), (4.39, .05), (5.93, .12), (4.01, .12), (3.40, .06), (3.94, .11)],
    [(4.46, .06), (4.50, .11), (5.77, .08), (4.15, .03), (3.59, .05), (3.90, .11)],
    [(4.70, .05), (4.62, .05), (5.77, .22), (4.33, .22), (3.94, .05), (4.27, .12)],
]
# Figure 7, left (p. 18): survived games out of 32. Its caption says standard
# deviation, but values such as 0.33 +- 0.33 (seeds 0, 0, 1) are standard errors.
# The figure omits DS-R1-Distill-32B, so it is 5 x 5 in PAPER_AGENTS[:5] order.
PAPER_SURVIVED = [
    [(0.00, .00), (0.00, .00), (16.67, 1.45), (4.67, .33), (0.33, .33)],
    [(0.00, .00), (0.00, .00), (20.00, 1.53), (4.67, .88), (2.00, .00)],
    [(0.33, .33), (0.00, .00), (5.67, 2.19), (0.33, .33), (0.00, .00)],
    [(0.00, .00), (0.00, .00), (8.33, 1.20), (0.33, .33), (0.00, .00)],
    [(0.00, .00), (0.00, .00), (7.33, .33), (1.33, .33), (0.00, .00)],
]
# Table 1 (p. 8): the 10 human games, played against Llama-3.1-70B-Instruct as Eve.
HUMAN_TEAM = {"win_rate": 0.40, "game_length": 6.90, "games": 10}
PAPER_LLMS = PAPER_AGENTS[2:]  # the like-for-like block: no embedding baselines

# --- Style: the reference palette (dataviz skill references/palette.md), light mode.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"        # categorical slot 1 (validated)
CONTEXT = "#8f8e88"       # de-emphasis gray for the paper: 3.2:1 on surface, dE 18.5 vs accent
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("seq_orange", ["#fbe0d2", "#f4a986", "#eb6834", "#b84a1e", "#7a2e10"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "regular",
})


def sem(xs):
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def wilson(k, n, z=1.96):
    """95% Wilson score interval, returned as (point, lo, hi) in %."""
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def add_caption(fig, text, width=125, axis_band=0.10):
    """Plain-language caption along the bottom of a figure. `axis_band` is the
    figure fraction kept free above the caption for tick labels and axis labels."""
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    n_lines = wrapped.count("\n") + 1
    fig.subplots_adjust(bottom=axis_band + 0.032 * n_lines)
    fig.text(0.5, 0.015, wrapped, ha="center", va="bottom", fontsize=8.5, color=INK2, style="italic")


def style_axes(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load():
    rows = []
    for seed, path in enumerate(CSVS):
        with open(path) as f:
            part = [dict(r, seed=seed) for r in csv.DictReader(f)]
        assert len(part) == 16, f"{path}: expected 16 rows, got {len(part)}"
        for r in part:
            assert r["encoder"] == r["decoder"], f"{path}: encoder != decoder in {r}"
            n = sum(int(r[k]) for k in ("intercepts", "miscomms", "both", "survived"))
            assert n == EPISODES, f"{path}: {n} episodes in {r['encoder']} vs {r['interceptor']}"
        rows += part
    return rows


def cell(rows, team, opp, fn):
    vals = [fn(r) for r in rows if r["encoder"] == team and r["interceptor"] == opp]
    assert len(vals) == 3
    return st.mean(vals), sem(vals)


def intercept_share(r):
    return (int(r["intercepts"]) + int(r["both"])) / EPISODES


def turns(r):
    return float(r["avg_turns_per_episode"])


def marginal(grid_rows, axis):
    """Mean of a set of (mean, se) cells and its propagated SE (cells treated as independent)."""
    means = [m for m, _ in grid_rows]
    return st.mean(means), math.sqrt(sum(s * s for _, s in grid_rows)) / len(grid_rows)


def fig_grids(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    specs = [("Share of games ending in an interception", intercept_share, BLUES, (0.80, 1.00), "{:.2f}",
              [0.80, 0.85, 0.90, 0.95, 1.00]),
             ("Average turns per game", turns, ORANGES, (3.0, 6.0), "{:.2f}", [3.0, 4.0, 5.0, 6.0])]
    for k_ax, (ax, (title, fn, cmap, (vmin, vmax), fmt, cticks)) in enumerate(zip(axes, specs)):
        grid = [[cell(rows, t, o, fn) for o in MODELS] for t in MODELS]
        im = ax.imshow([[m for m, _ in row] for row in grid], cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        for k in range(1, 4):  # 2px surface gap between cells
            ax.axhline(k - 0.5, color=SURFACE, linewidth=2.2)
            ax.axvline(k - 0.5, color=SURFACE, linewidth=2.2)
        for i, row in enumerate(grid):
            for j, (m, s) in enumerate(row):
                rgba = cmap((m - vmin) / (vmax - vmin))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{fmt.format(m)}\n±{s:.2f}", ha="center", va="center", fontsize=9,
                        color=("white" if lum < 0.45 else INK))
        ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=20, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels([LABEL[m] for m in MODELS])
        ax.set_xlabel("opponent (the lone code-cracker)")
        if k_ax == 0:
            ax.set_ylabel("team (same model writes and reads the clues)")
        ax.set_title(title, color=INK, pad=8)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03, ticks=cticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.93, wspace=0.42)
    add_caption(fig, axis_band=0.20, text=
        "Each square is one match-up: a team (one model writing the clues and the same model reading them, one per row) "
        "against one opponent model trying to crack the code (one per column), played 32 times on each of 3 seeds. "
        "Left: the share of those games the opponent won by cracking the code twice. It is 0.81 or more everywhere, and "
        "Claude as the code-cracker won 96-100% against every team. Right: how many turns the games lasted (a team that survives all 8 turns "
        "wins). The Qwen3-4B column is the lightest on the left and darkest on the right: the weakest code-cracker let "
        "teams last longest. Small numbers are the standard error across the 3 seeds.")
    out = os.path.join(OUTDIR, "fig1_competitive_grids.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def comparison_points(rows):
    """Per-model marginals over the like-for-like LLM block of each study."""
    ours_int = {o: marginal([cell(rows, t, o, intercept_share) for t in MODELS], 0) for o in MODELS}
    ours_turn = {t: marginal([cell(rows, t, o, turns) for o in MODELS], 1) for t in MODELS}
    idx = [PAPER_AGENTS.index(a) for a in PAPER_LLMS]
    paper_int = {PAPER_AGENTS[j]: marginal([PAPER_INTERCEPT[i][j] for i in idx], 0) for j in idx}
    paper_turn = {PAPER_AGENTS[i]: marginal([PAPER_TURNS[i][j] for j in idx], 1) for i in idx}
    return ours_int, ours_turn, paper_int, paper_turn


def fig_vs_paper(rows):
    ours_int, ours_turn, paper_int, paper_turn = comparison_points(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    panels = [
        (axes[0], ours_int, paper_int, "As the code-cracker", "share of games won by cracking the code\n(averaged over the 4 LLM teams it faced)", (0.6, 1.02), "{:.3f}"),
        (axes[1], ours_turn, paper_turn, "As the team", "average turns per game\n(averaged over the 4 LLM code-crackers it faced)", (3.0, 7.4), "{:.2f}"),
    ]
    for ax, ours, paper, title, xlabel, xlim, fmt in panels:
        pts = [(LABEL[k], v, True) for k, v in ours.items()] + [(k, v, False) for k, v in paper.items()]
        pts.sort(key=lambda p: p[1][0])
        for y, (name, (m, s), is_ours) in enumerate(pts):
            color = ACCENT if is_ours else CONTEXT
            ax.errorbar(m, y, xerr=s, fmt="o" if is_ours else "s", color=color, ecolor=color,
                        elinewidth=1.2, capsize=0, markersize=8 if is_ours else 7,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            if is_ours:
                ax.annotate(fmt.format(m), (m + s, y), textcoords="offset points", xytext=(7, 0),
                            va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(pts)))
        ax.set_yticklabels([p[0] for p in pts])
        for tick, p in zip(ax.get_yticklabels(), pts):
            tick.set_color(INK if p[2] else MUTED)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(pts) - 0.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, color=INK, loc="left")
        style_axes(ax)
    h = HUMAN_TEAM["game_length"]
    axes[1].axvline(h, color=INK2, linewidth=0.9)
    axes[1].text(h - 0.06, 0.2, f"human teams: {h:.2f}\n(10 games vs Llama-3.1-70B,\npaper Table 1)", ha="right",
                 va="bottom", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ACCENT, markersize=8, markeredgecolor=SURFACE, label="this run (Polaris, 2026)"),
               plt.Line2D([], [], marker="s", linestyle="", color=CONTEXT, markersize=7, markeredgecolor=SURFACE, label="paper, Figure 3 (Lupu et al. 2025)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.86, wspace=0.55, left=0.14, right=0.97)
    add_caption(fig, axis_band=0.17, text=
        "Each dot is one model, averaged over the four language models on the other side in its own study (the paper's "
        "embedding baselines are left out). Blue circles are this run; gray squares are the paper's models. Left: the code-cracker "
        "won 87-99% of games here, against 72-98% in the paper; the weakest code-cracker in each study (Qwen3-4B here, "
        "Llama-3.1-8B there) sits clearly below the rest. Right: teams lasted about 4 to 4.6 turns in both studies, far "
        "short of the 6.9 turns human teams reached. The opponents differ between the studies, so read this as the same "
        "pattern, not a head-to-head ranking. Bars are ±1 standard error.", width=128)
    out = os.path.join(OUTDIR, "fig2_vs_paper.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def team_outcomes(rows):
    out = {}
    for t in MODELS:
        g = [r for r in rows if r["encoder"] == t]
        n = EPISODES * len(g)
        out[t] = {
            "n": n,
            "intercepted": sum(int(r["intercepts"]) + int(r["both"]) for r in g),
            "miscomm": sum(int(r["miscomms"]) for r in g),
            "survived": sum(int(r["survived"]) for r in g),
            "both": sum(int(r["both"]) for r in g),
        }
    return out


def fig_outcomes(rows):
    oc = team_outcomes(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), sharey=True)
    panels = [("intercepted", "Opponent cracked the code", (80, 100)),
              ("miscomm", "Teammate misread the clues", (-0.6, 12)),
              ("survived", "Team survived all 8 turns", (-0.6, 12))]
    order = sorted(MODELS, key=lambda t: oc[t]["intercepted"] / oc[t]["n"])
    for ax, (key, title, xlim) in zip(axes, panels):
        for y, t in enumerate(order):
            p, lo, hi = wilson(oc[t][key], oc[t]["n"])
            ax.plot([lo, hi], [y, y], color=ACCENT, linewidth=1.4, solid_capstyle="round", zorder=2)
            ax.plot(p, y, "o", color=ACCENT, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.annotate(f"{p:.1f}%", (hi, y), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(*xlim)
        ax.set_title(title, color=INK, loc="left", fontsize=10.5)
        ax.set_xlabel("share of the team's 384 games (%)")
        style_axes(ax)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABEL[t] for t in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    fig.subplots_adjust(wspace=0.18, left=0.14, right=0.97, top=0.89)
    add_caption(fig, axis_band=0.13, text=
        "How each team's 384 games ended (every opponent, every seed). Most games ended with the opponent cracking the code; "
        "the panels have different scales so the rarer endings are visible. Qwen3-4B lost more often by confusing its own "
        "teammate, and Claude was the only team that never did and the one that most often lasted all 8 turns. The lines "
        "are 95% ranges given 384 games each. The 8 games where both limits were reached on the same turn count as "
        "cracked, as in the paper.", width=130)
    out = os.path.join(OUTDIR, "fig3_outcomes_by_team.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_tables(rows):
    print("\n## Our grids (mean +- SE over 3 seeds); rows = team, cols = opponent")
    for name, fn in [("interception share", intercept_share), ("avg turns", turns),
                     ("survived of 32", lambda r: int(r["survived"]))]:
        print(f"{name}: | team | " + " | ".join(LABEL[m] for m in MODELS) + " |")
        for t in MODELS:
            print(f"| {LABEL[t]} | " + " | ".join("{:.2f} ± {:.2f}".format(*cell(rows, t, o, fn)) for o in MODELS) + " |")
    ours_int, ours_turn, paper_int, paper_turn = comparison_points(rows)
    print("\n## Marginals (mean, SE)")
    for k, v in ours_int.items(): print(f"ours  interceptor {LABEL[k]:18s} {v[0]:.3f} ± {v[1]:.3f}")
    for k, v in paper_int.items(): print(f"paper interceptor {k:18s} {v[0]:.3f} ± {v[1]:.3f}")
    for k, v in ours_turn.items(): print(f"ours  team turns   {LABEL[k]:18s} {v[0]:.2f} ± {v[1]:.2f}")
    for k, v in paper_turn.items(): print(f"paper team turns   {k:18s} {v[0]:.2f} ± {v[1]:.2f}")
    idx = [PAPER_AGENTS.index(a) for a in PAPER_LLMS]
    pi = [PAPER_INTERCEPT[i][j][0] for i in idx for j in idx]
    pt = [PAPER_TURNS[i][j][0] for i in idx for j in idx]
    print(f"\npaper LLM block: interception {st.mean(pi):.3f}, turns {st.mean(pt):.2f}; "
          f"ours: interception {st.mean(intercept_share(r) for r in rows):.3f}, turns {st.mean(turns(r) for r in rows):.2f}")
    print("\n## By opponent (pooled over teams and seeds; Wilson 95%) and turns of games against it (mean ± propagated SE)")
    for o in MODELS:
        g = [r for r in rows if r["interceptor"] == o]
        n = EPISODES * len(g)
        ki = sum(int(r["intercepts"]) + int(r["both"]) for r in g)
        ks = sum(int(r["survived"]) for r in g)
        tm = marginal([cell(rows, t, o, turns) for t in MODELS], 0)
        print(f"{LABEL[o]:18s} intercepted={ki}/{n} ({wilson(ki, n)[0]:.1f}% [{wilson(ki, n)[1]:.1f}, {wilson(ki, n)[2]:.1f}])"
              f"  survived={ks}/{n} ({wilson(ks, n)[0]:.1f}% [{wilson(ks, n)[1]:.1f}, {wilson(ks, n)[2]:.1f}])"
              f"  turns against={tm[0]:.2f} ± {tm[1]:.2f}")
    print("\n## Outcomes by team (count / 384, Wilson 95%)")
    for t, o in team_outcomes(rows).items():
        print(f"{LABEL[t]:18s} " + "  ".join(f"{k}={o[k]} ({wilson(o[k], o['n'])[0]:.1f}% [{wilson(o[k], o['n'])[1]:.1f}, {wilson(o[k], o['n'])[2]:.1f}])"
                                              for k in ("intercepted", "miscomm", "survived")) + f"  both={o['both']}")


# --- Statistics -----------------------------------------------------------------
OUTCOMES = [  # (key, label, per-(cell, seed) value)
    ("intercept", "interception share", intercept_share),
    ("miscomm", "miscommunication share", lambda r: int(r["miscomms"]) / EPISODES),
    ("survived", "survival share", lambda r: int(r["survived"]) / EPISODES),
    ("turns", "turns per game", turns),
]
COMPONENTS = ["opponent", "team", "team x opponent", "between seeds"]
JOBS = {0: 7668935, 1: 7668936, 2: 7668937}


def cube(rows, fn):
    """Balanced 4 (team) x 4 (opponent) x 3 (seed) array of a per-row value."""
    return np.array([[[fn(r) for r in sorted(rows, key=lambda r: r["seed"])
                       if r["encoder"] == t and r["interceptor"] == o] for o in MODELS] for t in MODELS])


def anova2(rows, fn):
    """Two-way ANOVA with replication (team, opponent; 3 seeds per cell). Returns
    sums of squares, eta^2 (share of the total sum of squares), F and p for the two
    main effects and the interaction; the residual is the between-seed variation."""
    Y = cube(rows, fn)
    a, b, n = Y.shape
    gm = Y.mean()
    tm, om, cm = Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=2)
    ss = {
        "team": b * n * ((tm - gm) ** 2).sum(),
        "opponent": a * n * ((om - gm) ** 2).sum(),
        "team x opponent": n * ((cm - tm[:, None] - om[None, :] + gm) ** 2).sum(),
        "between seeds": ((Y - cm[:, :, None]) ** 2).sum(),
    }
    df = {"team": a - 1, "opponent": b - 1, "team x opponent": (a - 1) * (b - 1), "between seeds": a * b * (n - 1)}
    total = sum(ss.values())
    ms_e = ss["between seeds"] / df["between seeds"]
    out = {}
    for k in COMPONENTS:
        f = (ss[k] / df[k]) / ms_e if k != "between seeds" else float("nan")
        p = stats.f.sf(f, df[k], df["between seeds"]) if k != "between seeds" else float("nan")
        out[k] = {"ss": ss[k], "df": df[k], "eta2": ss[k] / total, "F": f, "p": p}
    return out


def binomial_dispersion(rows, fn):
    """Is the seed-to-seed spread within a cell larger than binomial sampling of 32
    games predicts? Pearson heterogeneity X^2 over cells with 0 < p < 1 (df = 2 each),
    and the ratio of observed to expected within-cell variance."""
    Y = cube(rows, fn)
    x2, dfs, obs_var, exp_var = 0.0, 0, [], []
    for i in range(4):
        for j in range(4):
            ks = Y[i, j] * EPISODES
            p = ks.sum() / (EPISODES * len(ks))
            if 0 < p < 1:
                x2 += (((ks - EPISODES * p) ** 2) / (EPISODES * p * (1 - p))).sum()
                dfs += len(ks) - 1
            obs_var.append(Y[i, j].var(ddof=1))
            exp_var.append(p * (1 - p) / EPISODES)
    return {"X2": x2, "df": dfs, "p": stats.chi2.sf(x2, dfs), "ratio": float(np.mean(obs_var) / np.mean(exp_var))}


def pooled_counts(rows, by, key):
    """Pooled counts (k, n) of an outcome per level of `by` (team or opponent)."""
    col = "encoder" if by == "team" else "interceptor"
    k = {"intercept": lambda r: int(r["intercepts"]) + int(r["both"]),
         "miscomm": lambda r: int(r["miscomms"]), "survived": lambda r: int(r["survived"])}[key]
    return {m: (sum(k(r) for r in rows if r[col] == m), EPISODES * sum(1 for r in rows if r[col] == m)) for m in MODELS}


def chi2_levels(counts):
    table = [[k, n - k] for k, n in counts.values()]
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return chi2, dof, p


def fisher_vs_rest(counts, level):
    k1, n1 = counts[level]
    k2 = sum(k for m, (k, n) in counts.items() if m != level)
    n2 = sum(n for m, (k, n) in counts.items() if m != level)
    odds, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return (k1, n1), (k2, n2), odds, p


def grand_mean_se(cells):
    """Mean of 16 (mean, se) cells and its SE, cells treated as independent."""
    return st.mean(m for m, _ in cells), math.sqrt(sum(s * s for _, s in cells)) / len(cells)


def audit_logs():
    """Counts from the job logs (logs/ is not in git; skipped when absent). Only the
    final attempt of each job is counted: that attempt produced the CSV."""
    out = {}
    for seed, jid in JOBS.items():
        path = os.path.join(REPO, "logs", "paper", f"polaris_fused4_{jid}.log")
        if not os.path.exists(path):
            out[seed] = None
            continue
        with open(path, errors="replace") as f:
            lines = f.read().split("\n")
        starts = [i for i, l in enumerate(lines) if l.startswith("=== fused 4-model job ")]
        fin = lines[starts[-1]:]
        out[seed] = {
            "attempts": len(starts),
            "claude_timeouts": sum("subprocess.TimeoutExpired" in l for l in fin),
            "claude_cli_errors": sum(("claude -p failed" in l) or ("claude -p error" in l) for l in fin),
            "rate_limit_msgs": sum(any(s in l for s in ("RateLimitError", "Rate limited", "usage limit", "rate limit")) for l in fin),
            # one vLLM access-log line per rejected request (each error's message is printed twice)
            "context_400": sum('"POST /v1/chat/completions HTTP/1.1" 400' in l for l in fin),
            "context_msgs": sum("maximum context length" in l and "serving_chat.py" in l for l in fin),
        }
    return out


def statistics_report(rows):
    print("\n## Two-way ANOVA (team, opponent, interaction; residual = between seeds), eta^2 = SS / SS_total")
    parts = {}
    for key, label, fn in OUTCOMES:
        a = anova2(rows, fn)
        parts[key] = a
        print(f"{label:24s} " + "  ".join(
            f"{k}: eta2={a[k]['eta2']:.3f}" + (f" F({a[k]['df']},{a['between seeds']['df']})={a[k]['F']:.2f} p={a[k]['p']:.2g}" if k != "between seeds" else "")
            for k in COMPONENTS))
    print("\n## Between-seed spread vs binomial sampling (Pearson heterogeneity over cells with 0<p<1)")
    for key, label, fn in OUTCOMES[:3]:
        d = binomial_dispersion(rows, fn)
        print(f"{label:24s} X2={d['X2']:.1f} df={d['df']} p={d['p']:.2f} observed/expected variance={d['ratio']:.2f}")
    print("\n## Pooled chi-square across levels (4 x 2 tables, 384 games per level)")
    for by, key in [("opponent", "intercept"), ("team", "intercept"), ("team", "miscomm"), ("team", "survived"), ("opponent", "survived")]:
        c = pooled_counts(rows, by, key)
        chi2, dof, p = chi2_levels(c)
        print(f"{key:9s} by {by:8s}: " + ", ".join(f"{LABEL[m]} {k}/{n}" for m, (k, n) in c.items()) + f" | chi2({dof})={chi2:.1f} p={p:.2g}")
    print("\n## Single-level contrasts (Fisher exact, two-sided)")
    for by, key, level in [("opponent", "intercept", "qwen3_4b"), ("team", "survived", "claude4.6_cli"),
                           ("team", "miscomm", "qwen3_4b"), ("team", "miscomm", "claude4.6_cli")]:
        (k1, n1), (k2, n2), odds, p = fisher_vs_rest(pooled_counts(rows, by, key), level)
        print(f"{key:9s} {by} {LABEL[level]:18s} {k1}/{n1} ({100*k1/n1:.1f}%) vs rest {k2}/{n2} ({100*k2/n2:.1f}%) odds ratio={odds:.2f} p={p:.2g}")
    c = pooled_counts(rows, "opponent", "intercept")
    sub = {m: c[m] for m in ("qwen2.5_72B", "qwen3_8b")}
    (k1, n1), (k2, n2), odds, p = fisher_vs_rest(sub, "qwen2.5_72B")
    print(f"size contrast, as opponent: Qwen2.5-72B {k1}/{n1} vs Qwen3-8B {k2}/{n2}, Fisher p={p:.2g}")
    t72 = [turns(r) for r in sorted(rows, key=lambda r: (r["interceptor"], r["seed"])) if r["encoder"] == "qwen2.5_72B"]
    t8 = [turns(r) for r in sorted(rows, key=lambda r: (r["interceptor"], r["seed"])) if r["encoder"] == "qwen3_8b"]
    tt = stats.ttest_rel(t72, t8)
    print(f"size contrast, as team turns: Qwen2.5-72B {st.mean(t72):.2f} vs Qwen3-8B {st.mean(t8):.2f}, "
          f"paired over 12 (opponent, seed) pairs: diff={st.mean(t72) - st.mean(t8):.2f} t(11)={tt.statistic:.2f} p={tt.pvalue:.2g}")
    tc = [turns(r) for r in sorted(rows, key=lambda r: (r["interceptor"], r["seed"])) if r["encoder"] == "claude4.6_cli"]
    for other in ("qwen2.5_72B", "qwen3_8b", "qwen3_4b"):
        to = [turns(r) for r in sorted(rows, key=lambda r: (r["interceptor"], r["seed"])) if r["encoder"] == other]
        tt = stats.ttest_rel(tc, to)
        print(f"team turns Claude {st.mean(tc):.2f} vs {LABEL[other]} {st.mean(to):.2f}: diff={st.mean(tc) - st.mean(to):.2f} t(11)={tt.statistic:.2f} p={tt.pvalue:.2g}")
    print("\n## This run vs the paper's LLM-only block (grand mean of 16 cells, SE propagated from cell SEs)")
    idx = [PAPER_AGENTS.index(x) for x in PAPER_LLMS]
    for label, fn, paper in [("interception share", intercept_share, PAPER_INTERCEPT), ("turns per game", turns, PAPER_TURNS)]:
        ours = grand_mean_se([cell(rows, t, o, fn) for t in MODELS for o in MODELS])
        theirs = grand_mean_se([paper[i][j] for i in idx for j in idx])
        z = (ours[0] - theirs[0]) / math.sqrt(ours[1] ** 2 + theirs[1] ** 2)
        print(f"{label:20s} ours {ours[0]:.3f} ± {ours[1]:.3f} | paper {theirs[0]:.3f} ± {theirs[1]:.3f} | diff {ours[0]-theirs[0]:+.3f} z={z:.2f} p={2*stats.norm.sf(abs(z)):.2g}")
    # Same comparison without each study's weakest interceptor (Qwen3-4B here,
    # Llama-3.1-8B in the paper): how much of the gap that one column carries.
    ours_wo = grand_mean_se([cell(rows, t, o, intercept_share) for t in MODELS for o in MODELS if o != "qwen3_4b"])
    jw = [j for j in idx if PAPER_AGENTS[j] != "Llama-3.1-8B"]
    theirs_wo = grand_mean_se([PAPER_INTERCEPT[i][j] for i in idx for j in jw])
    z = (ours_wo[0] - theirs_wo[0]) / math.sqrt(ours_wo[1] ** 2 + theirs_wo[1] ** 2)
    print(f"interception share without the weakest interceptor: ours {ours_wo[0]:.3f} ± {ours_wo[1]:.3f} | paper "
          f"{theirs_wo[0]:.3f} ± {theirs_wo[1]:.3f} | diff {ours_wo[0]-theirs_wo[0]:+.3f} z={z:.2f} p={2*stats.norm.sf(abs(z)):.2g}")
    print("\n## Survival: team x opponent residuals (observed - additive prediction), games survived per 32")
    Y = cube(rows, lambda r: int(r["survived"]))
    gm, tm, om, cmn = Y.mean(), Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=2)
    res = [(cmn[i, j] - (tm[i] + om[j] - gm), MODELS[i], MODELS[j], cmn[i, j]) for i in range(4) for j in range(4)]
    for r_, t, o, obs in sorted(res, key=lambda x: -abs(x[0]))[:3]:
        print(f"team {LABEL[t]:18s} vs {LABEL[o]:18s} observed {obs:.2f}, additive prediction {obs - r_:.2f}, residual {r_:+.2f}")
    print("\n## Retries (CSV avg_*_attempts: mean retries per turn)")
    for role in ("encoder", "decoder", "interceptor"):
        a = [float(r[f"avg_{role}_attempts"]) for r in rows]
        print(f"{role:11s} mean={st.mean(a):.4f} max={max(a):.3f}; final fails={sum(int(r[f'total_{role}_fails']) for r in rows)}")
    print("\n## Log audit (final attempt per job)")
    for seed, a in audit_logs().items():
        print(f"seed {seed} (job {JOBS[seed]}): {a if a else 'log not found'}")
    return parts


def fig_variance(parts):
    colors = {"opponent": "#2a78d6", "team": "#eb6834", "team x opponent": "#1baf7a", "between seeds": "#eda100"}
    names = {"intercept": "Games ending in an interception", "miscomm": "Games ending in a miscommunication",
             "survived": "Games the team survived", "turns": "Turns per game"}
    order = ["intercept", "turns", "miscomm", "survived"]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for y, key in enumerate(order):
        left = 0.0
        for comp in COMPONENTS:
            w = parts[key][comp]["eta2"]
            ax.barh(y, w, left=left, height=0.5, color=colors[comp], edgecolor=SURFACE, linewidth=1.5)
            if w >= 0.09:
                ax.text(left + w / 2, y, f"{100*w:.0f}%", ha="center", va="center", fontsize=9,
                        color=("white" if comp in ("opponent",) else INK))
            left += w
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the variation across the 48 (match-up, seed) results")
    style_axes(ax)
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in COMPONENTS]
    fig.legend(handles, ["which opponent", "which team", "this team against this opponent (interaction)", "between seeds (sampling)"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.99), fontsize=9)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86)
    add_caption(fig, axis_band=0.15, text=
        "Where the differences between results come from. For each outcome, the bar splits the variation across the 16 "
        "match-ups x 3 seeds into the part explained by which model was the opponent, which model was the team, the specific "
        "pairing beyond those two, and the leftover differences between seeds. Interceptions, survival and game length are "
        "set mostly by the opponent; miscommunications mostly by the team. Only survival has a sizeable pairing-specific part, "
        "which comes from the Claude team against the Qwen3-4B opponent.", width=125)
    out = os.path.join(OUTDIR, "fig4_variance_partition.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load()
    assert len(rows) == 48
    for fn in (fig_grids, fig_vs_paper, fig_outcomes):
        print("wrote", os.path.relpath(fn(rows), REPO))
    print_tables(rows)
    parts = statistics_report(rows)
    print("wrote", os.path.relpath(fig_variance(parts), REPO))


if __name__ == "__main__":
    main()
