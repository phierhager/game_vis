"""Render the static figure gallery into docs/figures (light and dark variants).

    python scripts/make_figures.py                 # everything
    python scripts/make_figures.py signaling_squares meaning_plane --theme light
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamevis import markets as mk  # noqa: E402
from gamevis import normal_form as nf  # noqa: E402
from gamevis import signaling as sg  # noqa: E402
from gamevis import simplex  # noqa: E402
from gamevis.qlearning import QPricing  # noqa: E402
from gamevis.plotting import (  # noqa: E402
    THEMES,
    Theme,
    arrow_along,
    bary_to_xy,
    categorical_raster,
    figure_title,
    line_with_ring,
    marker,
    marker_legend,
    receiver_corner_glyphs,
    save,
    sender_corner_glyphs,
    signaling_flow,
    simplex_streams,
    streams,
    theme,
    triangle,
    unit_square,
)

OUT = ROOT / "docs" / "figures"
FIGURES = {}


def figure(name):
    def register(fn):
        FIGURES[name] = fn
        return fn

    return register


def outcome_legend(ax, t: Theme, labels, codes_present=None, loc="upper left", bbox=(0.0, -0.08), ncol=4):
    colors = t.outcome_colors()
    handles, names = [], []
    for c, (col, lab) in enumerate(zip(colors, labels)):
        if codes_present is not None and c not in codes_present:
            continue
        handles.append(Line2D([], [], color=col, lw=2.2))
        names.append(lab)
    return ax.legend(handles, names, loc=loc, bbox_to_anchor=bbox, ncol=ncol, handlelength=1.4, columnspacing=1.4)


# ---------------------------------------------------------------------------
# Signaling games
# ---------------------------------------------------------------------------


@figure("signaling_squares")
def fig_signaling_squares(t: Theme):
    game = sg.lewis(2)
    rng = np.random.default_rng(11)
    X0 = rng.random((80, 4))
    times, X = sg.simulate_binary(game, X0, "replicator", t_max=80, dt=0.05, record_every=2)
    codes = sg.classify_binary(X[-40:])
    colors = t.outcome_colors()
    hero = 3
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.3))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.75, bottom=0.2, wspace=0.36)
    figure_title(
        fig,
        "A learning run is two paths at once",
        "Lewis game, equal priors, replicator dynamics. Each line is one run; its sender half and receiver half move together.",
        t,
    )
    specs = [
        (axes[0], 0, 1, "Sender: which message for which state", "P(m1 | t1)", "P(m1 | t2)", sender_corner_glyphs),
        (axes[1], 2, 3, "Receiver: which action for which message", "P(a1 | m1)", "P(a1 | m2)", receiver_corner_glyphs),
    ]
    for ax, i, j, name, xl, yl, glyphs in specs:
        unit_square(ax, t, xl, yl)
        ax.set_title(name, pad=30, fontsize=10.5)
        for r in range(X.shape[1]):
            if r == hero:
                continue
            ax.plot(X[:, r, i], X[:, r, j], color=t.mix(colors[codes[r]], 0.5), lw=0.85, zorder=2)
        # the highlighted run, with a dot every 5 time units
        line_with_ring(ax, X[:, hero, i], X[:, hero, j], t.ink, t, lw=2.0, zorder=5)
        every = int(round(5 / (times[1] - times[0])))
        ax.plot(X[::every, hero, i], X[::every, hero, j], "o", ms=3.2, color=t.ink, zorder=6)
        ax.plot(X[0, hero, i], X[0, hero, j], "o", ms=7, mfc=t.surface, mec=t.ink, mew=1.5, zorder=7)
        glyphs(ax, t, offset=0.1, size=0.1)
    labels = sg.outcome_labels(game)
    outcome_legend(axes[0], t, labels, codes_present={0, 1}, bbox=(0.0, -0.17), ncol=2)
    axes[0].text(0.0, -0.27, "Ink: one run, open circle at the start, a dot every 5 time units.", transform=axes[0].transAxes, fontsize=8, color=t.muted, va="top")
    axes[1].text(0.0, -0.17, "Corner diagrams show the pure strategy at each corner:", transform=axes[1].transAxes, fontsize=8, color=t.muted, va="top")
    axes[1].text(0.0, -0.23, "coloured dots are the states t1, t2 (left) and the actions that fit them (right).", transform=axes[1].transAxes, fontsize=8, color=t.muted, va="top")
    return fig


@figure("meaning_plane")
def fig_meaning_plane(t: Theme):
    priors = [0.5, 0.8]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.9))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.17, wspace=0.28)
    figure_title(
        fig,
        "Convention formation is an escape from a saddle",
        "Sender separation vs receiver separation for 150 random starts (replicator). Skewed priors open a trap where nobody means anything.",
        t,
    )
    colors = t.outcome_colors()
    for ax, p in zip(axes, priors):
        game = sg.lewis(2, p)
        X0 = np.random.default_rng(5).random((150, 4))
        times, X = sg.simulate_binary(game, X0, "replicator", t_max=200, dt=0.05, record_every=4)
        codes = sg.classify_binary(X[-60:])
        dx = X[..., 0] - X[..., 1]
        dy = X[..., 2] - X[..., 3]
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0, color=t.grid, lw=0.8, zorder=0)
        ax.axvline(0, color=t.grid, lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec=t.axis, lw=0.8, zorder=0.5))
        if p == 0.5:
            # common payoff (1 + dx dy) / 2: hyperbolic contours of a saddle
            g = np.linspace(-1, 1, 201)
            GX, GY = np.meshgrid(g, g)
            ax.contour(GX, GY, (1 + GX * GY) / 2, levels=[0.1, 0.25, 0.4, 0.6, 0.75, 0.9], colors=[t.axis], linewidths=0.7, zorder=0.6)
            ax.text(0.02, 0.02, "grey curves: equal common payoff (1 + dx\u00b7dy)/2", transform=ax.transAxes, fontsize=7.5, color=t.muted)
        order = np.argsort(codes == 2)
        for r in order:
            ax.plot(dx[:, r], dy[:, r], color=t.mix(colors[codes[r]], 0.6 if codes[r] != 2 else 0.8), lw=0.8, zorder=2 + (codes[r] == 2))
            ax.plot(dx[-1, r], dy[-1, r], "o", ms=3, color=colors[codes[r]], zorder=4)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(length=0, pad=4)
        ax.set_xlabel("sender separation  P(m1|t1) − P(m1|t2)")
        ax.set_ylabel("receiver separation  P(a1|m1) − P(a1|m2)")
        share = np.mean(codes == 2)
        ax.set_title(f"Prior P(t1) = {p:.1f}", pad=22)
        ax.text(0.0, 1.035, f"no information in {share:.0%} of runs", transform=ax.transAxes, color=t.ink2, fontsize=8.5, ha="left", va="bottom")
        ax.text(1.0, 1.1, "m1 means t1", ha="right", va="center", fontsize=8, color=t.ink2)
        ax.text(-1.0, -1.1, "m1 means t2", ha="left", va="center", fontsize=8, color=t.ink2)
    outcome_legend(axes[1], t, sg.outcome_labels(sg.lewis(2)), codes_present={0, 1, 2}, bbox=(0.0, -0.13), ncol=3)
    return fig


@figure("flow_filmstrip")
def fig_flow_filmstrip(t: Theme):
    game = sg.lewis(3)
    rng = np.random.default_rng(21)
    # pick one run that finds a signaling system and one that gets stuck in partial pooling
    S0, R0 = sg.random_start(game, rng, 400)
    tr = sg.simulate(game, S0, R0, "replicator", t_max=300, dt=0.1, record_every=1)
    k = sg.distinguished_states(game, tr.S[-1], tr.R[-1])
    full = int(np.flatnonzero(k > 2.95)[0])
    partial = int(np.flatnonzero(np.abs(k - 2) < 0.05)[0])
    snaps = [0.0, 4.0, 12.0, 30.0, 300.0]
    fig, axes = plt.subplots(2, len(snaps), figsize=(12.8, 6.9))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.05, wspace=0.08, hspace=0.42)
    figure_title(
        fig,
        "Meaning, forming: states \u2192 messages \u2192 actions",
        "Three-state Lewis game, replicator dynamics. Band width is probability; colour follows the state it carries.",
        t,
    )
    for row, run in enumerate([full, partial]):
        for col, s in enumerate(snaps):
            ax = axes[row, col]
            idx = min(int(round(s / (tr.t[1] - tr.t[0]))), len(tr.t) - 1)
            S, R = tr.S[idx, run], tr.R[idx, run]
            first, last = col == 0, col == len(snaps) - 1
            signaling_flow(ax, game, S, R, t, label_states=first, label_messages=first or last, label_actions=last, fontsize=8)
            u = game.payoffs(S, R)[1]
            ax.text(1.0, -0.075, f"t = {s:g}", ha="right", va="top", fontsize=9, color=t.ink2)
            ax.text(1.06, -0.075, f"{u:.0%} success", ha="left", va="top", fontsize=9, color=t.muted)
        axes[row, 0].text(-0.3, 1.16, ["This run finds a signaling system", "This run gets stuck in partial pooling: t2 and t3 share m2, m1 goes unused"][row], fontsize=10, color=t.ink, fontweight="bold", ha="left")
    return fig


@figure("pooling_trap")
def fig_pooling_trap(t: Theme):
    priors = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    rules = [("replicator", 400, 0.1), ("projection", 400, 0.05), ("softmax_pg", 1500, 0.1)]
    ratios = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.14, wspace=0.28)
    figure_title(
        fig,
        "When does language fail to form?",
        "Share of 2,000 random starts that end with no information transmitted (2×2×2 Lewis game).",
        t,
    )
    ax = axes[0]
    for c, (rule, tmax, dt) in enumerate(rules):
        ys = [sg.basin_fractions(sg.lewis(2, p), rule, 2000, tmax, dt)[2] for p in priors]
        ax.plot(priors, ys, color=t.series[c], lw=2, zorder=3, label=simplex.RULE_LABELS[rule])
        ax.plot(priors[-1], ys[-1], "o", ms=6, color=t.series[c], mec=t.surface, mew=1.5, zorder=4)
    ax.legend(loc="upper left", handlelength=1.4)
    ax.set_xlim(0.5, 0.965)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("prior probability of the likelier state")
    ax.set_title("Skewed priors open a pooling trap", pad=8)
    _percent_axis(ax, t, 0.6)
    ax = axes[1]
    game = sg.lewis(2, 0.8)
    ys = [sg.basin_fractions(game, "replicator", 2000, 400, 0.1, rates=(r, 1.0))[2] for r in ratios]
    ax.plot(ratios, ys, color=t.series[0], lw=2, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios[::2])
    ax.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
    ax.minorticks_off()
    ax.set_xlabel("sender learning rate ÷ receiver learning rate")
    ax.set_title("Impatient listeners kill meaning (prior 0.8)", pad=8)
    _percent_axis(ax, t, 0.6)
    ax.plot([ratios[0], ratios[-1]], [ys[0], ys[-1]], "o", ms=6, color=t.series[0], mec=t.surface, mew=1.5, zorder=4)
    ax.text(ratios[0] * 1.25, ys[0], f"{ys[0]:.0%} when the receiver\nlearns 16\u00d7 faster", fontsize=8, color=t.ink2, va="center")
    ax.text(ratios[-1] / 1.2, ys[-1] + 0.045, f"{ys[-1]:.1%} when the sender\nlearns 16\u00d7 faster", fontsize=8, color=t.ink2, ha="right", va="bottom")
    return fig


def _percent_axis(ax, t: Theme, top: float):
    ax.set_ylim(0, top)
    ticks = np.arange(0, top + 1e-9, 0.1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.0%}" for v in ticks])
    ax.grid(axis="y", color=t.grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


@figure("basin_slices")
def fig_basin_slices(t: Theme):
    priors = [0.5, 0.7, 0.8, 0.9]
    n = 161
    g = (np.arange(n) + 0.5) / n
    GX, GY = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, len(priors), figsize=(12.8, 4.5))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.2, wspace=0.12)
    figure_title(
        fig,
        "Basins of attraction, one slice at a time",
        "Every pixel is a sender starting point; the receiver always starts undecided at (0.5, 0.5). Colour = where replicator learning ends up.",
        t,
    )
    colors = t.outcome_colors()
    for ax, p in zip(axes, priors):
        game = sg.lewis(2, p)
        X0 = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, 0.5), np.full(GX.size, 0.5)], -1)
        _, X = sg.simulate_binary(game, X0, "replicator", t_max=800, dt=0.1, record_every=10)
        codes = sg.classify_binary(X[-100:]).reshape(n, n)
        categorical_raster(ax, codes, (0, 1, 0, 1), colors, t)
        unit_square(ax, t, "P(m1 | t1) at start", "P(m1 | t2) at start" if p == priors[0] else "")
        ax.set_title(f"Prior P(t1) = {p:.1f}", pad=20)
        share = np.mean(codes == 2)
        note = "no information only on the diagonal" if p == 0.5 else f"no information from {share:.0%} of the slice"
        ax.text(0.0, 1.03, note, transform=ax.transAxes, fontsize=8.5, color=t.ink2, va="bottom")
    outcome_legend(axes[0], t, sg.outcome_labels(sg.lewis(2)), codes_present={0, 1, 2}, bbox=(0.0, -0.2), ncol=3)
    return fig


@figure("costly_cycles")
def fig_costly_cycles(t: Theme):
    cheap = sg.costly_signaling(prior_high=0.4, cost_high=0.3, cost_low=0.6)
    dear = sg.costly_signaling(prior_high=0.4, cost_high=0.3, cost_low=1.3)
    fig = plt.figure(figsize=(12.8, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1], left=0.06, right=0.98, top=0.78, bottom=0.2, wspace=0.3)
    figure_title(
        fig,
        "When faking is cheap, honesty and trust chase each other",
        "Handicap game: high and low types, a costly display, a receiver who wants to reward only the high type (replicator dynamics).",
        t,
    )
    Sh, Rh = sg.hybrid_equilibrium(cheap)
    hx, hy = Sh[1, 0], Rh[0, 0]
    ax = fig.add_subplot(gs[0])
    unit_square(ax, t, "mimicry: P(display | low type)", "trust: P(reward | display)")
    ax.set_title("Cheap faking: closed orbits", pad=8)
    starts = [0.08, 0.16, 0.24, 0.3]
    for c, d in enumerate(starts):
        X0 = np.array([1.0 - 1e-4, hx - d, hy + d * 0.9, 1e-4])
        times, X = sg.simulate_binary(cheap, X0, "replicator", t_max=80, dt=0.02, record_every=2)
        ax.plot(X[:, 1], X[:, 2], color=t.ink if c == 2 else t.muted, lw=2 if c == 2 else 1.0, zorder=3 if c == 2 else 2)
        arrow_along(ax, X[:, 1], X[:, 2], t.ink if c == 2 else t.muted, frac=0.12, size=10)
    marker(ax, hx, hy, "center", t)
    ax.annotate("hybrid equilibrium", xy=(hx, hy), xytext=(0.03, 0.08), fontsize=8, color=t.ink2, arrowprops=dict(arrowstyle="-", color=t.muted, lw=0.8, shrinkB=6))
    # time series of the highlighted orbit
    ax2 = fig.add_subplot(gs[1])
    X0 = np.array([1.0 - 1e-4, hx - 0.24, hy + 0.216, 1e-4])
    times, X = sg.simulate_binary(cheap, X0, "replicator", t_max=160, dt=0.02, record_every=5)
    ax2.plot(times, X[:, 1], color=t.series[1], lw=2, label="mimicry: P(display | low)")
    ax2.plot(times, X[:, 2], color=t.series[0], lw=2, label="trust: P(reward | display)")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, times[-1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.grid(axis="y", color=t.grid, lw=0.8)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("time")
    ax2.set_title("Mimicry rises, trust falls, mimicry stops paying", pad=8)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, handlelength=1.4)
    # expensive faking: honest signaling
    ax3 = fig.add_subplot(gs[2])
    unit_square(ax3, t, "mimicry: P(display | low type)", "trust: P(reward | display)")
    ax3.set_title("Expensive faking: honesty wins", pad=8)
    rng = np.random.default_rng(3)
    X0 = rng.random((60, 4))
    times, X = sg.simulate_binary(dear, X0, "replicator", t_max=200, dt=0.05, record_every=4)
    codes = sg.classify_binary(X[-40:])
    colors = t.outcome_colors()
    for r in range(X.shape[1]):
        ax3.plot(X[:, r, 1], X[:, r, 2], color=t.mix(colors[codes[r]], 0.55), lw=0.85, zorder=2)
    marker(ax3, 0.0, 1.0, "sink", t, clip=False)
    ax3.text(0.07, 0.985, "honest signaling", fontsize=8, color=t.ink2, va="top")
    share = np.mean(codes == 0)
    ax3.text(0.0, -0.19, f"{share:.0%} of runs end honest; the rest stop displaying", transform=ax3.transAxes, fontsize=8, color=t.ink2, va="top")
    ax.text(0.0, -0.19, "the low type's display cost 0.6 is below the reward 1", transform=ax.transAxes, fontsize=8, color=t.muted, va="top")
    ax3.text(0.0, -0.25, "the low type's display cost 1.3 is above the reward 1", transform=ax3.transAxes, fontsize=8, color=t.muted, va="top")
    return fig


@figure("lewis3_triangles")
def fig_lewis3_triangles(t: Theme):
    game = sg.lewis(3)
    rng = np.random.default_rng(21)
    S0, R0 = sg.random_start(game, rng, 400)
    tr = sg.simulate(game, S0, R0, "replicator", t_max=300, dt=0.1, record_every=1)
    k = sg.distinguished_states(game, tr.S[-1], tr.R[-1])
    runs = [int(np.flatnonzero(k > 2.95)[0]), int(np.flatnonzero(np.abs(k - 2) < 0.05)[0])]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.03, wspace=0.08, hspace=0.18)
    figure_title(
        fig,
        "Three states: each agent is a dot in its own triangle",
        "Left: each state's message mix (corners are messages). Right: each message's action mix (corner a_i fits state t_i).",
        t,
        top=0.985,
    )
    msg_ink = [t.ink, t.ink2, t.muted]
    for row, run in enumerate(runs):
        ax = axes[row, 0]
        triangle(ax, game.message_labels, t)
        for s in range(3):
            path = bary_to_xy(tr.S[:, run, s, :])
            line_with_ring(ax, path[:, 0], path[:, 1], t.series[s], t, lw=1.8, zorder=3)
            ax.plot(*path[0], "o", ms=5, mfc=t.surface, mec=t.series[s], mew=1.4, zorder=5)
            ax.plot(*path[-1], "o", ms=7, color=t.series[s], mec=t.surface, mew=1.4, zorder=6)
            ax.text(path[0, 0], path[0, 1] + 0.045, game.state_labels[s], ha="center", fontsize=8, color=t.ink2, zorder=7)
        ax2 = axes[row, 1]
        triangle(ax2, game.action_labels, t)
        for m in range(3):
            path = bary_to_xy(tr.R[:, run, m, :])
            line_with_ring(ax2, path[:, 0], path[:, 1], msg_ink[m], t, lw=1.6, zorder=3)
            ax2.plot(*path[0], "o", ms=5, mfc=t.surface, mec=msg_ink[m], mew=1.4, zorder=5)
            ax2.plot(*path[-1], "s", ms=6.5, color=msg_ink[m], mec=t.surface, mew=1.2, zorder=6)
            ax2.text(path[-1, 0] + 0.045, path[-1, 1] + 0.02, game.message_labels[m], fontsize=8, color=t.ink2, zorder=7)
        # colour ticks: which state each action fits
        for a, (x, y) in enumerate(bary_to_xy(np.eye(3))):
            ax2.plot(x, y, "o", ms=6, color=t.series[a], zorder=4)
        u = game.payoffs(tr.S[-1, run], tr.R[-1, run])[1]
        axes[row, 0].text(-0.14, 0.93, ["Signaling system", "Partial pooling"][row], fontsize=10.5, fontweight="bold", color=t.ink, transform=axes[row, 0].transAxes)
        axes[row, 0].text(-0.14, 0.86, f"success {u:.0%}", fontsize=8.5, color=t.ink2, transform=axes[row, 0].transAxes)
    axes[1, 1].text(0.5, -0.02, "An unused message gets no signal and freezes where it was.", ha="center", fontsize=8, color=t.muted, transform=axes[1, 1].transAxes)
    return fig


# ---------------------------------------------------------------------------
# Normal-form games
# ---------------------------------------------------------------------------

CORNER_NAMES = {(1, 1): (0, 0), (1, 0): (0, 1), (0, 1): (1, 0), (0, 0): (1, 1)}


def _abbrevs(labels) -> tuple[str, str]:
    """Shortest distinct prefixes of two action names (Swerve/Straight -> Sw/St)."""
    for n in range(1, 6):
        a, b = labels[0][:n], labels[1][:n]
        if a != b:
            return a, b
    return labels[0], labels[1]


def bimatrix_portrait(ax, game: nf.Bimatrix, t: Theme, rule: str = "replicator", basins: bool = True, density: float = 0.9, labels: bool = True, lw: float = 0.7, small: bool = False):
    """Streamlines, basin tint, Nash equilibria and corner action labels for a 2x2 game."""
    eqs = game.nash_equilibria()
    kinds = [game.equilibrium_stability(e) for e in eqs]
    sinks = [e for e, k in zip(eqs, kinds) if k == "sink"]
    if basins and len(sinks) > 1:
        n = 90
        g = (np.arange(n) + 0.5) / n
        GX, GY = np.meshgrid(g, g)
        end = game.simulate(GX, GY, rule, t_max=60, dt=0.05)[-1]
        d = np.stack([np.hypot(end[..., 0] - sp, end[..., 1] - sq) for sp, sq in sinks])
        codes = np.argmin(d, axis=0)
        cols = [t.series[i] for i in range(len(sinks))]
        categorical_raster(ax, codes, (0, 1, 0, 1), cols, t, amount=0.16)
    streams(ax, lambda X, Y: game.velocity(X, Y, rule), (0, 1, 0, 1), t, density=density, lw=lw, n=80, arrowsize=0.55 if small else 0.7)
    for e, k in zip(eqs, kinds):
        marker(ax, e[0], e[1], k, t, size=5.5 if small else 7.5, clip=False)
    if labels:
        ra, ca = _abbrevs(game.row_actions), _abbrevs(game.col_actions)
        for (x, y), (i, j) in CORNER_NAMES.items():
            lab = f"{ra[i]},{ca[j]}"
            ax.text(x + (0.04 if x else -0.04), y + (0.035 if y else -0.035), lab, ha="left" if x else "right", va="bottom" if y else "top", fontsize=7.5, color=t.muted)


@figure("normal_form_gallery")
def fig_normal_form_gallery(t: Theme):
    keys = ["prisoners_dilemma", "stag_hunt", "chicken", "matching_pennies", "battle_of_sexes", "coordination"]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 8.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.1, wspace=0.42, hspace=0.62)
    figure_title(
        fig,
        "Six 2×2 games as flows on the square",
        "Replicator dynamics. x = row player's probability of its first action, y = column player's. Tints mark basins of stable outcomes.",
        t,
    )
    for ax, key in zip(axes.ravel(), keys):
        g = nf.BIMATRIX_PRESETS[key]
        unit_square(ax, t, f"P(row: {g.row_actions[0]})", f"P(column: {g.col_actions[0]})", pad=0.0)
        bimatrix_portrait(ax, g, t)
        ax.set_title(g.name, pad=30, fontsize=10.5)
        ax.text(0, 1.1, _payoff_string(g), transform=ax.transAxes, fontsize=7.5, color=t.ink2, va="bottom")
    marker_legend(axes[1, 1], t, kinds=("sink", "saddle", "center"), bbox=(0.5, -0.32))
    return fig


def _payoff_string(g: nf.Bimatrix) -> str:
    A, B = g.A, g.B
    cells = [f"({A[i, j]:g}, {B[i, j]:g})" for i in range(2) for j in range(2)]
    return f"payoffs {cells[0]} {cells[1]} / {cells[2]} {cells[3]}"


@figure("discrete_vs_continuous")
def fig_discrete_vs_continuous(t: Theme):
    g = nf.BIMATRIX_PRESETS["matching_pennies"]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 4.3))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.72, bottom=0.14, wspace=0.25)
    figure_title(
        fig,
        "Same signal, different clocks: Matching Pennies",
        "Both players follow their own payoff gradient. Whether play cycles, spirals out or settles depends on how the signal is used.",
        t,
    )
    titles = ["Continuous replicator", "Multiplicative weights, η = 0.15", "Optimistic weights, η = 0.15", "Logit response, τ = 0.25"]
    subs = ["closed orbits: a conserved quantity", "each step overshoots: spirals out", "extrapolating the signal: spirals in", "smoothed best reply: settles"]
    for ax, title, sub in zip(axes, titles, subs):
        unit_square(ax, t, "P(row: Heads)", "P(column: Heads)" if ax is axes[0] else "", pad=0.0)
        ax.set_title(title, pad=18, fontsize=10)
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8, color=t.ink2, va="bottom")
    ax = axes[0]
    streams(ax, lambda X, Y: g.velocity(X, Y, "replicator"), (0, 1, 0, 1), t, density=0.7, n=80)
    tr = g.simulate(0.8, 0.5, "replicator", t_max=7.5, dt=0.005)
    line_with_ring(ax, tr[:, 0], tr[:, 1], t.ink, t)
    arrow_along(ax, tr[:, 0], tr[:, 1], t.ink, 0.3, size=11, zorder=6)
    marker(ax, 0.5, 0.5, "center", t)
    for ax, opt in [(axes[1], False), (axes[2], True)]:
        path = g.discrete_path(0.65, 0.5, "replicator", eta=0.15, steps=260 if not opt else 400, optimistic=opt)
        ax.plot(path[:, 0], path[:, 1], color=t.ink, lw=1.0, zorder=3)
        ax.plot(path[:, 0], path[:, 1], "o", ms=2.2, color=t.ink, zorder=4)
        ax.plot(path[0, 0], path[0, 1], "o", ms=7, mfc=t.surface, mec=t.ink, mew=1.4, zorder=5)
        marker(ax, 0.5, 0.5, "source" if not opt else "sink", t, size=6.5)
    ax = axes[3]
    streams(ax, lambda X, Y: g.velocity(X, Y, "logit", temperature=0.25), (0, 1, 0, 1), t, density=0.7, n=80)
    tr = g.simulate(0.9, 0.1, "logit", t_max=25, dt=0.01, temperature=0.25)
    line_with_ring(ax, tr[:, 0], tr[:, 1], t.ink, t)
    arrow_along(ax, tr[:, 0], tr[:, 1], t.ink, 0.15, size=11, zorder=6)
    marker(ax, 0.5, 0.5, "sink", t)
    return fig


@figure("game_atlas")
def fig_game_atlas(t: Theme):
    Ts = [0.25, 0.75, 1.25, 1.75]
    Ss = [0.75, 0.25, -0.25, -0.75]
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 10.8))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.85, bottom=0.1, wspace=0.12, hspace=0.3)
    figure_title(
        fig,
        "An atlas of symmetric 2\u00d72 games",
        "Reward R = 1, punishment P = 0. Temptation T grows to the right, sucker's payoff S grows upward.\nTwo-population replicator flow; the diagonal is what one population playing itself would do.",
        t,
    )
    for r, S in enumerate(Ss):
        for c, T in enumerate(Ts):
            ax = axes[r, c]
            g = nf.symmetric_2x2(1.0, S, T, 0.0)
            unit_square(ax, t, "", "", ticks=(), pad=0.0)
            ax.plot([0, 1], [0, 1], color=t.axis, lw=0.8, zorder=0.8)
            bimatrix_portrait(ax, g, t, density=0.55, labels=False, lw=0.6, small=True)
            if r == 3:
                ax.text(0.5, -0.1, f"T = {T:g}", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=t.ink2)
            if c == 0:
                ax.text(-0.1, 0.5, f"S = {S:g}", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=t.ink2)
    quad = [("Harmony", 0, 0), ("Snowdrift / Chicken", 0, 2), ("Stag Hunt", 2, 0), ("Prisoner's Dilemma", 2, 2)]
    for name, r, c in quad:
        axes[r, c].text(0.0, 1.06, name, transform=axes[r, c].transAxes, fontsize=10, fontweight="bold", color=t.ink, va="bottom")
    fig.text(0.555, 0.045, "temptation T  \u2192", ha="center", fontsize=9.5, color=t.ink2)
    fig.text(0.03, 0.475, "sucker's payoff S  \u2192", rotation=90, va="center", ha="center", fontsize=9.5, color=t.ink2)
    fig.text(0.555, 0.015, "In every panel: x = P(row cooperates), y = P(column cooperates). Tints mark the basins of the two stable outcomes.", ha="center", fontsize=8, color=t.muted)
    return fig


@figure("simplex_gallery")
def fig_simplex_gallery(t: Theme):
    keys = ["rps_zero_sum", "rps_stable", "rps_unstable", "coordination3", "hawk_dove_bourgeois", "repeated_pd"]
    subs = [
        "zero-sum: neutral cycles around the mixed equilibrium",
        "wins pay more than losses cost: spirals in",
        "losses cost more: spirals out to the boundary",
        "payoffs 1, 2, 3 for matching: three basins",
        "owners fight, intruders yield: Bourgeois wins",
        "tit for tat resists defectors, drifts with cooperators",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.08, wspace=0.14, hspace=0.36)
    figure_title(fig, "Three strategies, one population: flows on the triangle", "Replicator dynamics on the 2-simplex. Corners are pure populations; markers are rest points.", t)
    for ax, key, sub in zip(axes.ravel(), keys, subs):
        g = nf.SYMMETRIC_PRESETS[key]
        triangle(ax, g.labels, t, fontsize=8.5)
        simplex_streams(ax, lambda x: g.velocity(x), t, density=0.9)
        for rp in g.rest_points():
            xy = bary_to_xy(rp["x"])
            marker(ax, xy[0], xy[1], rp["stability"], t, size=7, clip=False)
        name = g.name.replace("Rock-Paper-Scissors", "Rock-paper-scissors")
        ax.set_title(name, fontsize=10.5, pad=24)
        ax.text(0.0, 1.035, sub, transform=ax.transAxes, fontsize=8, color=t.ink2, ha="left", va="bottom")
    marker_legend(axes[1, 1], t, kinds=("sink", "saddle", "source", "center", "degenerate"), bbox=(0.5, -0.12), ncol=5)
    return fig


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


@figure("bertrand_plane")
def fig_bertrand_plane(t: Theme):
    m = mk.LogitBertrand()
    lo, hi = 1.2, 2.2
    pn = m.nash()[0]
    pc = m.collusive()[0]
    fig = plt.figure(figsize=(12.8, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], left=0.05, right=0.98, top=0.78, bottom=0.17, wspace=0.28)
    figure_title(
        fig,
        "Bertrand duopoly: following your own marginal profit leads to Nash, not to the cartel",
        "Logit demand (Calvano et al. 2020 baseline). Background: joint profit. Lens: prices where both firms earn more than at Nash.",
        t,
    )
    g = np.linspace(lo, hi, 240)
    P1, P2 = np.meshgrid(g, g)
    pr1, pr2 = m.profit(P1, P2)
    nash_profit = m.profit(pn, pn)[0]
    br = np.linspace(lo, hi, 120)
    br1 = m.best_response(0, br)
    br2 = m.best_response(1, br)
    starts = np.array([[1.25, 2.15], [2.15, 1.3], [2.15, 2.15], [1.3, 1.3], [1.7, 2.18]])
    for k, (lam, title, sub) in enumerate([
        (0.0, "Selfish gradients", "each firm climbs its own profit"),
        (1.0, "Cooperative gradients", "each firm climbs joint profit"),
    ]):
        ax = fig.add_subplot(gs[k])
        ax.imshow(pr1 + pr2, origin="lower", extent=(lo, hi, lo, hi), cmap=t.gray_ramp(), aspect="equal", zorder=0)
        lens = np.minimum(pr1, pr2) - nash_profit
        ax.contourf(P1, P2, lens, levels=[0, 10], colors=[t.mix(t.series[2], 0.28)], zorder=0.5)
        ax.contour(P1, P2, lens, levels=[0], colors=[t.series[2]], linewidths=1.2, zorder=1)
        streams(ax, lambda X, Y: m.velocity(X, Y, sympathy=lam), (lo, hi, lo, hi), t, density=0.7, n=70, lw=0.7)
        ax.plot(br1, br, color=t.series[0], lw=2, zorder=3, label="firm 1's best reply")
        ax.plot(br, br2, color=t.series[1], lw=2, zorder=3, label="firm 2's best reply")
        traj = m.simulate(starts, t_max=120, dt=0.05, sympathy=lam)
        for i in range(len(starts)):
            line_with_ring(ax, traj[:, i, 0], traj[:, i, 1], t.ink, t, lw=1.5, zorder=4)
            ax.plot(traj[0, i, 0], traj[0, i, 1], "o", ms=5, mfc=t.surface, mec=t.ink, mew=1.2, zorder=5)
        marker(ax, pn, pn, "sink" if lam == 0 else "source", t)
        ax.plot([pc], [pc], "D", ms=7, mfc=t.surface if lam == 0 else t.ink, mec=t.ink, mew=1.4, zorder=6)
        box = dict(boxstyle="round,pad=0.15", fc=t.surface, ec="none", alpha=0.85)
        ax.text(pn + 0.035, pn - 0.05, "Nash", fontsize=8, color=t.ink, va="top", bbox=box, zorder=7)
        ax.text(pc + 0.04, pc - 0.02, "cartel", fontsize=8, color=t.ink, va="top", bbox=box, zorder=7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("price of firm 1")
        ax.set_ylabel("price of firm 2" if k == 0 else "")
        for s_ in ax.spines.values():
            s_.set_color(t.axis)
        ax.set_title(title, pad=18, fontsize=10.5)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=t.ink2, va="bottom")
        if k == 0:
            from matplotlib.patches import Patch

            handles = [
                Line2D([], [], color=t.series[0], lw=2),
                Line2D([], [], color=t.series[1], lw=2),
                Patch(fc=t.mix(t.series[2], 0.28), ec=t.series[2]),
            ]
            ax.legend(handles, ["firm 1's best reply", "firm 2's best reply", "both beat Nash"], loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, handlelength=1.4, columnspacing=1.0)
    ax = fig.add_subplot(gs[2])
    lams = np.linspace(0, 1, 21)
    prices = np.array([m.equilibrium(lam)[0] for lam in lams])
    profits = np.array([m.profit(p_, p_)[0] for p_ in prices])
    ax.plot(lams, prices, color=t.ink, lw=2)
    ax.plot([0, 1], [prices[0], prices[-1]], "o", ms=6, color=t.ink, mec=t.surface, mew=1.5)
    ax.text(0.04, prices[0], f"Nash {prices[0]:.3f}", fontsize=8, color=t.ink2, va="center")
    ax.text(0.96, prices[-1], f"cartel {prices[-1]:.3f}", fontsize=8, color=t.ink2, va="center", ha="right")
    ax.set_xlabel("sympathy λ: weight on the rival's profit")
    ax.set_ylabel("equilibrium price")
    ax.set_title("Caring about the rival lifts prices", pad=18, fontsize=10.5)
    ax.text(0, 1.03, "rest point of gradient play on profit_i + λ·profit_j", transform=ax.transAxes, fontsize=8.5, color=t.ink2, va="bottom")
    ax.grid(axis="y", color=t.grid, lw=0.8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(1.4, 2.0)
    ax.set_xlim(-0.03, 1.03)
    ax.text(0.0, -0.17, f"profit per firm rises from {profits[0]:.3f} to {profits[-1]:.3f}", transform=ax.transAxes, fontsize=8, color=t.muted, va="top")
    return fig


@figure("edgeworth_cycles")
def fig_edgeworth_cycles(t: Theme):
    e = mk.Edgeworth(capacity=0.5, cost=0.1, tick=0.01)
    path = e.simulate((0.9, 0.9), 70)
    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], left=0.06, right=0.98, top=0.76, bottom=0.14, wspace=0.2)
    figure_title(
        fig,
        "Edgeworth cycles: a price war that never ends",
        "Identical goods, demand 1 − p, cost 0.1, each firm can serve at most half the market. Firms take turns choosing a best reply.",
        t,
    )
    ax = fig.add_subplot(gs[0])
    periods = np.arange(len(path))
    ax.step(periods, path[:, 0], where="post", color=t.series[0], lw=2, label="firm 1")
    ax.step(periods, path[:, 1], where="post", color=t.series[1], lw=2, label="firm 2")
    for y, lab in [(e.monopoly_price(), "monopoly price"), (e.residual_price(), "relenting price"), (e.cost, "cost")]:
        ax.axhline(y, color=t.axis, lw=0.8, zorder=0)
        ax.text(len(path) - 1, y + 0.01, lab, ha="right", va="bottom", fontsize=8, color=t.muted)
    ax.set_xlim(0, len(path) - 1)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("period (firms alternate)")
    ax.set_ylabel("price")
    ax.set_title("Undercut, undercut, undercut, relent", pad=8, fontsize=10.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), handlelength=1.4)
    ax = fig.add_subplot(gs[1])
    grid = np.round(np.arange(e.cost, 0.7 + 1e-9, e.tick), 10)
    br = np.array([e.best_response(p_) for p_ in grid])
    ax.plot(grid, br, color=t.series[1], lw=1.8, drawstyle="steps-mid", label="firm 2's reply to firm 1")
    ax.plot(br, grid, color=t.series[0], lw=1.8, drawstyle="steps-mid", label="firm 1's reply to firm 2")
    cyc = path[20:]
    ax.plot(cyc[:, 0], cyc[:, 1], color=t.ink, lw=1.0, zorder=4)
    ax.plot(cyc[:, 0], cyc[:, 1], "o", ms=2.5, color=t.ink, zorder=5)
    ax.plot([0.1, 0.7], [0.1, 0.7], color=t.axis, lw=0.8, zorder=0)
    ax.set_xlim(0.12, 0.6)
    ax.set_ylim(0.12, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("price of firm 1")
    ax.set_ylabel("price of firm 2")
    ax.set_title("Best replies jump: no crossing, no rest", pad=8, fontsize=10.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper left", handlelength=1.4)
    ax.text(0.02, 0.8, "ink: the price path\n(alternating moves)", transform=ax.transAxes, fontsize=8, color=t.ink2, va="top")
    return fig


_QCACHE = {}


def _q_sessions():
    """Eight baseline sessions (beta = 4e-6, 1.2 million periods); computed once per run."""
    if "res" not in _QCACHE:
        qp = QPricing(beta=4e-6)
        _QCACHE["qp"] = qp
        _QCACHE["res"] = qp.run(sessions=8, periods=1_200_000, seed=1, record_every=10_000)
    return _QCACHE["qp"], _QCACHE["res"]


@figure("qlearning_pricing")
def fig_qlearning_pricing(t: Theme):
    qp, res = _q_sessions()
    K = res["Q"].shape[0]
    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 0.95], left=0.055, right=0.98, top=0.76, bottom=0.15, wspace=0.3)
    figure_title(
        fig,
        "Q-learning firms remember, punish, and keep prices high",
        "Two tabular Q-learners with one period of memory in the logit Bertrand market (Calvano et al. 2020 baseline, 8 independent runs).",
        t,
    )
    ref = lambda ax: [ax.axhline(v, color=t.axis, lw=0.9, ls=(0, (4, 3)), zorder=0) for v in (qp.p_nash, qp.p_monopoly)]  # noqa: E731
    ax = fig.add_subplot(gs[0])
    ref(ax)
    tt = res["t"] / 1e6
    for k in range(K):
        ax.plot(tt, res["prices"][:, k].mean(axis=1), color=t.mix(t.ink, 0.35), lw=1.0)
    ax.plot(tt, res["prices"].mean(axis=(1, 2)), color=t.ink, lw=2)
    ax.text(tt[-1], qp.p_monopoly + 0.012, "monopoly price", ha="right", va="bottom", fontsize=8, color=t.muted)
    ax.text(tt[-1], qp.p_nash - 0.012, "Nash price", ha="right", va="top", fontsize=8, color=t.muted)
    gains = [qp.cycle_profit_gain(res["Q"][k], int(res["state"][k])) for k in range(K)]
    ax.set_xlabel("periods (millions)")
    ax.set_ylabel("average price")
    ax.set_title("Training: prices drift up, not down", pad=18, fontsize=10.5)
    ax.text(0, 1.03, f"ink: mean of 8 runs; average profit gain {np.mean(gains):.0%} of the way from Nash to monopoly", transform=ax.transAxes, fontsize=8, color=t.ink2, va="bottom")
    ax.set_ylim(qp.prices[0] - 0.02, qp.prices[-1] + 0.02)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # impulse response
    ax = fig.add_subplot(gs[1])
    ref(ax)
    before, after, dev = 3, 14, 3
    paths = []
    for k in range(K):
        s0 = qp.limit_cycle(res["Q"][k], int(res["state"][k]))[0]
        paths.append(qp.prices[qp.play(res["Q"][k], s0, periods=before + after, deviate_at=dev)])
    paths = np.array(paths)  # (K, T, 2)
    periods = np.arange(paths.shape[1]) - dev
    ax.axvspan(-0.5, 0.5, color=t.mix(t.ink, 0.07), lw=0, zorder=0)
    ax.plot(periods, paths[:, :, 0].mean(0), color=t.series[0], lw=2, marker="o", ms=4, label="firm 1 (deviates at 0)")
    ax.plot(periods, paths[:, :, 1].mean(0), color=t.series[1], lw=2, marker="o", ms=4, label="firm 2")
    ax.set_xlabel("periods after firm 1's one-off price cut")
    ax.set_title("A price cut is punished, then forgiven", pad=18, fontsize=10.5)
    ax.text(0, 1.03, "average over the 8 runs; firm 1 plays its static best reply once", transform=ax.transAxes, fontsize=8, color=t.ink2, va="bottom")
    ax.legend(loc="lower left", handlelength=1.4)
    ax.set_ylim(qp.prices[0] - 0.02, qp.prices[-1] + 0.02)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # where the learners end up
    ax = fig.add_subplot(gs[2])
    q_prices = []
    for k in range(K):
        cyc = qp.limit_cycle(res["Q"][k], int(res["state"][k]))
        q_prices.append(np.mean([(qp.prices[c // qp.m] + qp.prices[c % qp.m]) / 2 for c in cyc]))
    m = mk.LogitBertrand()
    starts = np.random.default_rng(2).uniform(1.2, 2.2, size=(K, 2))
    g_prices = m.simulate(starts, t_max=300, dt=0.05)[-1].mean(axis=1)
    rows = [("gradient play", g_prices), ("Q-learning", np.array(q_prices))]
    for v, lab in [(qp.p_nash, "Nash"), (qp.p_monopoly, "monopoly")]:
        ax.axvline(v, color=t.axis, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.text(v, 1.62, lab, ha="center", va="bottom", fontsize=8, color=t.muted)
    jit = np.linspace(-0.12, 0.12, K)
    for r, (lab, vals) in enumerate(rows):
        y = 1 - r
        ax.plot(vals, y + jit, "o", ms=6.5, color=t.ink if r == 0 else t.series[0], mec=t.surface, mew=1.2, zorder=3)
        ax.text(1.43, y + 0.25, lab, fontsize=9, color=t.ink, fontweight="bold", va="bottom")
    ax.set_ylim(-0.45, 1.75)
    ax.set_yticks([])
    ax.set_xlim(1.42, 2.0)
    ax.set_xlabel("long-run price (average of both firms)")
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.set_title("Where the learners end up", pad=18, fontsize=10.5)
    ax.text(0, 1.03, "8 runs each; gradient play always finds Nash", transform=ax.transAxes, fontsize=8, color=t.ink2, va="bottom")
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def swarm_gif(t: Theme, path: Path, n: int = 260, prior: float = 0.8, frames: int = 90):
    from matplotlib.animation import FuncAnimation, PillowWriter

    game = sg.lewis(2, prior)
    X0 = np.random.default_rng(8).random((n, 4))
    times, X = sg.simulate_binary(game, X0, "replicator", t_max=150, dt=0.05, record_every=1)
    codes = sg.classify_binary(X[-200:])
    colors = np.array([t.outcome_colors()[c] for c in codes])
    # ease time: more frames early on, when things move fast
    idx = np.unique((np.linspace(0, 1, frames) ** 2.2 * (len(times) - 1)).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.25, wspace=0.34)
    fig.text(0.012, 0.975, f"260 learning runs at once (Lewis game, prior {prior})", fontsize=12, fontweight="bold", color=t.ink, va="top")
    unit_square(axes[0], t, "P(m1 | t1)", "P(m1 | t2)")
    unit_square(axes[1], t, "P(a1 | m1)", "P(a1 | m2)")
    axes[0].set_title("Sender", fontsize=10, pad=6)
    axes[1].set_title("Receiver", fontsize=10, pad=6)
    ax = axes[2]
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec=t.axis, lw=0.8))
    ax.axhline(0, color=t.grid, lw=0.8)
    ax.axvline(0, color=t.grid, lw=0.8)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(length=0)
    ax.set_title("Meaning plane", fontsize=10, pad=6)
    ax.set_xlabel("sender separation")
    ax.set_ylabel("receiver separation")
    sc = [
        axes[0].scatter(X[0, :, 0], X[0, :, 1], s=9, c=colors, linewidths=0),
        axes[1].scatter(X[0, :, 2], X[0, :, 3], s=9, c=colors, linewidths=0),
        ax.scatter(X[0, :, 0] - X[0, :, 1], X[0, :, 2] - X[0, :, 3], s=9, c=colors, linewidths=0),
    ]
    clock = fig.text(0.98, 0.975, "", ha="right", va="top", fontsize=10, color=t.ink2)
    legend = [Line2D([], [], ls="", marker="o", ms=6, color=c) for c in t.outcome_colors()[:3]]
    fig.legend(legend, ["m1 means t1", "m1 means t2", "no information"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))

    def update(k):
        i = idx[k]
        sc[0].set_offsets(X[i, :, :2])
        sc[1].set_offsets(X[i, :, 2:])
        sc[2].set_offsets(np.stack([X[i, :, 0] - X[i, :, 1], X[i, :, 2] - X[i, :, 3]], -1))
        clock.set_text(f"t = {times[i]:.0f}")
        return sc

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=15), dpi=72, savefig_kwargs={"facecolor": t.surface})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help=f"figures to render (default: all). Choices: {', '.join(FIGURES)}")
    ap.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    ap.add_argument("--gif", action="store_true", help="also render the animated swarm (slow)")
    args = ap.parse_args(argv)
    names = args.names or list(FIGURES)
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        for tn in themes:
            t0 = time.time()
            t = THEMES[tn]
            with theme(t):
                fig = FIGURES[name](t)
                path = save(fig, str(OUT / name), t)
            print(f"{path}  ({time.time() - t0:.1f}s)")
    if args.gif:
        for tn in themes:
            t = THEMES[tn]
            with theme(t):
                path = OUT / f"signaling_swarm_{tn}.gif"
                swarm_gif(t, path)
            print(path)


if __name__ == "__main__":
    main()
