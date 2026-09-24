"""Matplotlib helpers shared by the figure gallery.

Colours follow one fixed system (the same tokens as the web pages):
categorical slots in a fixed order, hairline recessive axes, ink for the
thing that matters, a muted grey for context. Every helper takes the active
:class:`Theme` so each figure renders in a light and a dark variant.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    neutral: str
    blue_ramp: tuple[str, ...]

    def mix(self, color: str, amount: float) -> tuple[float, float, float]:
        """``color`` blended toward the surface: ``amount = 1`` is the pure colour."""
        c = np.array(to_rgb(color))
        s = np.array(to_rgb(self.surface))
        return tuple(s + amount * (c - s))

    def sequential(self, color: str | None = None, name: str = "seq") -> LinearSegmentedColormap:
        """One-hue ramp from the surface to a strong step of ``color`` (blue by default)."""
        top = color or (self.blue_ramp[-2] if self.name == "light" else self.blue_ramp[1])
        return LinearSegmentedColormap.from_list(name, [self.surface, top])

    def gray_ramp(self, name: str = "gray") -> LinearSegmentedColormap:
        """Neutral ramp for context backgrounds (magnitude without a hue)."""
        top = "#cfcdc5" if self.name == "light" else "#55544f"
        return LinearSegmentedColormap.from_list(name, [self.surface, top])

    def outcome_colors(self) -> list[str]:
        """Signaling outcomes: m1 means t1, m1 means t2, no information, cycling or partial."""
        return [self.series[0], self.series[1], self.muted, self.series[2]]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    neutral="#f0efec",
    blue_ramp=("#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    neutral="#383835",
    blue_ramp=("#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"),
)

THEMES = {"light": LIGHT, "dark": DARK}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


@contextmanager
def theme(t: Theme):
    """Apply ``t`` to matplotlib's rcParams for the duration of the block."""
    rc = {
        "figure.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "axes.edgecolor": t.axis,
        "axes.linewidth": 0.8,
        "axes.labelcolor": t.ink2,
        "axes.titlecolor": t.ink,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.grid": False,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "xtick.color": t.axis,
        "ytick.color": t.axis,
        "xtick.labelcolor": t.muted,
        "ytick.labelcolor": t.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "text.color": t.ink,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": t.ink2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
    }
    with mpl.rc_context(rc):
        yield t


def subtitle(fig_or_ax, text: str, t: Theme, y: float = 1.0, x: float = 0.0, size: float = 9.0) -> None:
    """Secondary line under a title, in secondary ink."""
    if hasattr(fig_or_ax, "transAxes"):
        fig_or_ax.text(x, y, text, transform=fig_or_ax.transAxes, color=t.ink2, fontsize=size, va="bottom", ha="left")
    else:
        fig_or_ax.text(x, y, text, color=t.ink2, fontsize=size, va="top", ha="left")


def figure_title(fig, title: str, sub: str | None, t: Theme, top: float = 0.985) -> None:
    height = fig.get_size_inches()[1]
    fig.text(0.012, top, title, color=t.ink, fontsize=13, fontweight="bold", va="top", ha="left")
    if sub:
        fig.text(0.012, top - 0.3 / height, sub, color=t.ink2, fontsize=9.5, va="top", ha="left")


def unit_square(ax, t: Theme, xlabel: str = "", ylabel: str = "", ticks=(0, 0.5, 1), pad: float = 0.03) -> None:
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=t.axis, lw=0.8, zorder=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0, pad=7)


def streams(ax, f, extent, t: Theme, density: float = 1.1, color=None, lw: float = 0.7, n: int = 120, mask=None, arrowsize: float = 0.7, zorder: float = 1):
    """Streamlines of a planar field ``f(X, Y) -> (U, V)`` over ``extent = (x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    U, V = f(X, Y)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if mask is not None:
        m = ~mask(X, Y)
        U = np.ma.array(U, mask=m)
        V = np.ma.array(V, mask=m)
    speed = np.hypot(U, V)
    top = np.nanpercentile(np.ma.filled(speed, np.nan), 95) or 1.0
    width = lw * (0.45 + 0.9 * np.clip(np.ma.filled(speed, 0) / top, 0, 1))
    ax.streamplot(
        xs, ys, U, V, density=density, color=color or t.muted, linewidth=width, arrowsize=arrowsize,
        arrowstyle="-|>", minlength=0.08, zorder=zorder, broken_streamlines=True,
    )


def marker(ax, x: float, y: float, kind: str, t: Theme, size: float = 7.5, color=None, zorder: float = 6, clip: bool = False):
    """Rest-point glyph: sink filled, source open, saddle half, center ringed dot."""
    c = color or t.ink
    common = dict(ms=size, mec=c, mew=1.4, zorder=zorder, clip_on=clip)
    # Surface-colour ring so the glyph stays legible on top of lines.
    ax.plot([x], [y], "o", ms=size + 3.2, color=t.surface, mec=t.surface, zorder=zorder - 0.1, clip_on=clip)
    if kind == "sink":
        ax.plot([x], [y], "o", mfc=c, **common)
    elif kind == "source":
        ax.plot([x], [y], "o", mfc=t.surface, **common)
    elif kind == "saddle":
        ax.plot([x], [y], "o", mfc=c, fillstyle="left", markerfacecoloralt=t.surface, **common)
    elif kind == "center":
        ax.plot([x], [y], "o", mfc=t.surface, **common)
        ax.plot([x], [y], "o", ms=size * 0.3, color=c, zorder=zorder + 0.1, clip_on=clip)
    else:
        ax.plot([x], [y], "D", mfc=t.surface, ms=size * 0.8, mec=c, mew=1.3, zorder=zorder, clip_on=clip)


def marker_legend(ax, t: Theme, kinds=("sink", "saddle", "source", "center"), loc="lower center", bbox=(0.5, -0.02), ncol=4):
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    names = {"sink": "stable", "source": "unstable", "saddle": "saddle", "center": "neutral (cycles)", "degenerate": "non-hyperbolic"}
    handles = []
    for k in kinds:
        if k == "sink":
            h = Line2D([], [], ls="", marker="o", ms=7, mfc=t.ink, mec=t.ink)
        elif k == "source":
            h = Line2D([], [], ls="", marker="o", ms=7, mfc=t.surface, mec=t.ink, mew=1.3)
        elif k == "saddle":
            h = Line2D([], [], ls="", marker="o", ms=7, mfc=t.ink, mec=t.ink, fillstyle="left", markerfacecoloralt=t.surface)
        elif k == "center":
            h = (Line2D([], [], ls="", marker="o", ms=7, mfc=t.surface, mec=t.ink, mew=1.3), Line2D([], [], ls="", marker="o", ms=2.2, color=t.ink))
        else:
            h = Line2D([], [], ls="", marker="D", ms=6, mfc=t.surface, mec=t.ink, mew=1.3)
        handles.append(h)
    return ax.legend(
        handles, [names[k] for k in kinds], loc=loc, bbox_to_anchor=bbox, ncol=ncol, handletextpad=0.3, columnspacing=1.2,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )


def arrow_along(ax, xs, ys, color, frac: float = 0.5, lw: float = 1.2, size: float = 9, zorder: float = 3):
    """Arrowhead on a polyline at fraction ``frac`` of its arclength."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    if seg.sum() == 0:
        return
    cum = np.concatenate([[0], np.cumsum(seg)])
    i = int(np.searchsorted(cum, frac * cum[-1]))
    i = min(max(i, 1), len(xs) - 1)
    j = max(i - 1, 0)
    ax.add_patch(
        FancyArrowPatch((xs[j], ys[j]), (xs[i], ys[i]), arrowstyle="-|>", mutation_scale=size, color=color, lw=0, zorder=zorder)
    )


def line_with_ring(ax, xs, ys, color, t: Theme, lw: float = 2.0, zorder: float = 4, **kw):
    """A 2px line with a surface-coloured halo so it reads over busy backgrounds."""
    ax.plot(xs, ys, color=t.surface, lw=lw + 2.2, zorder=zorder - 0.1, solid_capstyle="round")
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, solid_capstyle="round", **kw)


# ---------------------------------------------------------------------------
# Signaling-game glyphs
# ---------------------------------------------------------------------------


def mapping_glyph(ax, cx: float, cy: float, targets: tuple[int, int], t: Theme, left_colors, size: float = 0.13, right_colors=None):
    """Tiny wiring diagram of a pure 2x2 strategy, drawn in data coordinates.

    Two dots on the left (states or messages) point at the right-hand dots
    they are mapped to: ``targets = (j0, j1)``.
    """
    w, h = size, size * 0.8
    lx, rx = cx - w / 2, cx + w / 2
    ys = (cy + h / 2, cy - h / 2)
    right_colors = right_colors or (t.ink2, t.ink2)
    for i, j in enumerate(targets):
        ax.add_patch(
            FancyArrowPatch(
                (lx, ys[i]), (rx, ys[j]), arrowstyle="-|>", mutation_scale=6, color=t.ink2, lw=0.9,
                shrinkA=3.2, shrinkB=3.2, clip_on=False, zorder=7,
            )
        )
    for i in range(2):
        ax.add_patch(Circle((lx, ys[i]), size * 0.085, color=left_colors[i], clip_on=False, zorder=8, lw=0))
        ax.add_patch(Circle((rx, ys[i]), size * 0.085, fc=t.surface, ec=right_colors[i], lw=1.0, clip_on=False, zorder=8))


def sender_corner_glyphs(ax, t: Theme, offset: float = 0.13, size: float = 0.12):
    """Label the sender square's corners with the pure strategies they are."""
    c = t.series
    corners = {(1, 0): (0, 1), (0, 1): (1, 0), (1, 1): (0, 0), (0, 0): (1, 1)}
    for (x, y), tg in corners.items():
        gx = x + (offset if x else -offset)
        gy = y + (offset if y else -offset)
        mapping_glyph(ax, gx, gy, tg, t, (c[0], c[1]), size)


def receiver_corner_glyphs(ax, t: Theme, offset: float = 0.13, size: float = 0.12):
    corners = {(1, 0): (0, 1), (0, 1): (1, 0), (1, 1): (0, 0), (0, 0): (1, 1)}
    for (x, y), tg in corners.items():
        gx = x + (offset if x else -offset)
        gy = y + (offset if y else -offset)
        mapping_glyph(ax, gx, gy, tg, t, (t.ink2, t.ink2), size, right_colors=(t.series[0], t.series[1]))


def signaling_flow(
    ax, game, S, R, t: Theme, gap: float = 0.06, node_w: float = 0.035, alpha: float = 0.6, fontsize: float = 8.5,
    label_states: bool = True, label_messages: bool = True, label_actions: bool = True,
):
    """States -> messages -> actions, with band widths equal to probability flow.

    Bands keep the colour of the state they carry, so a working language
    shows up as unmixed colours arriving at matching actions.
    """
    n, k, l = game.n_states, game.n_messages, game.n_actions
    pi = game.prior
    flow_tm = pi[:, None] * S  # (n, k)
    flow_tma = flow_tm[:, :, None] * R[None, :, :]  # (n, k, l)
    pm = flow_tm.sum(0)
    pa = flow_tma.sum((0, 1))
    cols = [0.0, 1.0, 2.0]
    # One scale for all three columns (each carries total mass 1), so band
    # widths are comparable end to end.
    sc = 1.0 - gap * (max(n, k, l) - 1)
    t_tops = _even_tops(pi, sc, gap)
    m_tops = _even_tops(pm, sc, gap)
    a_tops = _even_tops(pa, sc, gap)

    state_col = [t.series[i % len(t.series)] for i in range(n)]

    def band(x0, y0a, y0b, x1, y1a, y1b, color):
        xm0, xm1 = x0 + (x1 - x0) * 0.45, x0 + (x1 - x0) * 0.55
        verts = [
            (x0, y0a), (xm0, y0a), (xm1, y1a), (x1, y1a),
            (x1, y1b), (xm1, y1b), (xm0, y0b), (x0, y0b), (x0, y0a),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=t.mix(color, alpha), ec="none", lw=0, zorder=1))

    # states -> messages
    t_cursor = t_tops.copy()
    m_cursor = m_tops.copy()
    m_in = {}  # (t, m) -> (top, bottom) inside message node, used to route onward
    for m in range(k):
        for ti in range(n):
            w = flow_tm[ti, m] * sc
            if w <= 1e-9:
                continue
            y0a = t_cursor[ti]
            y1a = m_cursor[m]
            band(cols[0] + node_w, y0a, y0a - w, cols[1] - node_w, y1a, y1a - w, state_col[ti])
            m_in[(ti, m)] = (y1a, y1a - w)
            t_cursor[ti] -= w
            m_cursor[m] -= w
    # messages -> actions, split by state so colours carry through
    a_cursor = a_tops.copy()
    for a in range(l):
        for m in range(k):
            for ti in range(n):
                w = flow_tma[ti, m, a] * sc
                if w <= 1e-9:
                    continue
                y0a = m_in[(ti, m)][0]
                m_in[(ti, m)] = (y0a - w, m_in[(ti, m)][1])
                y1a = a_cursor[a]
                band(cols[1] + node_w, y0a, y0a - w, cols[2] - node_w, y1a, y1a - w, state_col[ti])
                a_cursor[a] -= w
    # nodes
    for ti in range(n):
        h = pi[ti] * sc
        ax.add_patch(plt.Rectangle((cols[0] - node_w, t_tops[ti] - h), 2 * node_w, h, fc=state_col[ti], ec="none", zorder=3))
        if label_states:
            ax.text(cols[0] - node_w - 0.05, t_tops[ti] - h / 2, game.state_labels[ti], ha="right", va="center", fontsize=fontsize, color=t.ink2)
    for m in range(k):
        h = pm[m] * sc
        ax.add_patch(plt.Rectangle((cols[1] - node_w, m_tops[m] - h), 2 * node_w, max(h, 0.004), fc=t.ink2, ec="none", zorder=3))
        if label_messages:
            ax.text(
                cols[1], m_tops[m] + gap / 2, game.message_labels[m], ha="center", va="center", fontsize=fontsize - 1, color=t.ink2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc=t.surface, ec="none", alpha=0.9),
            )
    for a in range(l):
        h = pa[a] * sc
        ax.add_patch(plt.Rectangle((cols[2] - node_w, a_tops[a] - h), 2 * node_w, max(h, 0.004), fc=t.ink2, ec="none", zorder=3))
        if label_actions:
            ax.text(cols[2] + node_w + 0.05, a_tops[a] - h / 2, game.action_labels[a], ha="left", va="center", fontsize=fontsize, color=t.ink2)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.02, 1.08)
    ax.axis("off")


def _even_tops(masses: np.ndarray, scale: float, gap: float) -> np.ndarray:
    tops, y = [], 1.0
    for mval in masses:
        tops.append(y)
        y -= mval * scale + gap
    return np.array(tops)


# ---------------------------------------------------------------------------
# Simplex (triangle) helpers
# ---------------------------------------------------------------------------

TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def bary_to_xy(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) @ TRI


def xy_to_bary(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x3 = Y / TRI[2, 1]
    x2 = X - 0.5 * x3
    x1 = 1 - x2 - x3
    return np.stack([x1, x2, x3], -1)


def triangle(ax, labels, t: Theme, fontsize: float = 9, label_colors=None, pad: float = 0.06):
    ax.add_patch(Polygon(TRI, closed=True, fill=False, ec=t.axis, lw=0.8, zorder=0.5))
    offs = [(-0.02, -0.05, "right", "top"), (1.02, -0.05, "left", "top"), (0.5, TRI[2, 1] + 0.035, "center", "bottom")]
    for (x, y, ha, va), lab in zip(offs, labels):
        ax.text(x, y, lab, ha=ha, va=va, fontsize=fontsize, color=t.ink2)
    ax.set_xlim(-pad - 0.08, 1 + pad + 0.08)
    ax.set_ylim(-pad - 0.04, TRI[2, 1] + pad + 0.02)
    ax.set_aspect("equal")
    ax.axis("off")


def simplex_streams(ax, velocity, t: Theme, density: float = 1.0, n: int = 160, lw: float = 0.75, arrowsize: float = 0.7):
    """Streamlines of ``velocity(x) -> dx`` (barycentric, shape (..., 3)) on the triangle."""

    def f(X, Y):
        B = xy_to_bary(X, Y)
        inside = np.all(B >= -1e-9, axis=-1)
        Bc = np.clip(B, 0, None)
        Bc = Bc / Bc.sum(-1, keepdims=True)
        V = velocity(Bc)
        D = V @ TRI
        U = np.where(inside, D[..., 0], np.nan)
        W = np.where(inside, D[..., 1], np.nan)
        return U, W

    def mask(X, Y):
        B = xy_to_bary(X, Y)
        return np.all(B >= 0.004, axis=-1)

    streams(ax, f, (0, 1, 0, TRI[2, 1]), t, density=density, n=n, lw=lw, mask=mask, arrowsize=arrowsize)


def categorical_raster(ax, codes: np.ndarray, extent, colors, t: Theme, amount: float = 0.55, zorder: float = 0):
    """Show an integer code grid with softened categorical colours (large areas stay calm)."""
    cmap = ListedColormap([t.mix(c, amount) for c in colors])
    ax.imshow(codes, origin="lower", extent=extent, cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest", zorder=zorder, aspect="auto")


def save(fig, path_stem: str, t: Theme, dpi: int = 150) -> str:
    path = f"{path_stem}_{t.name}.png"
    fig.savefig(path, dpi=dpi, facecolor=t.surface)
    plt.close(fig)
    return path
