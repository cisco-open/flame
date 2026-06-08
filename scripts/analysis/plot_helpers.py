# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reusable matplotlib helpers for per-run telemetry analysis.

Outputs vector PDF with paper-style fonts. Every figure carries a two-line title:
line 1 = what it shows; line 2 (smaller, gray) = a compact experiment-config stamp
so a plot is self-describing. Selectors get fixed colors for visual consistency.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Quiet the very verbose PDF font-subsetting / matplotlib INFO logs.
for _n in ("matplotlib", "matplotlib.font_manager", "fontTools",
           "fontTools.subset", "fontTools.ttLib"):
    logging.getLogger(_n).setLevel(logging.WARNING)

# Paper-style rcParams (fonttype 42 => editable text in PDF/PS). Title/axis-label
# sizes kept crisp so they fit the box; ticks/legend readable.
plt.rcParams.update({
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": [6.5, 3.6],
    "legend.fontsize": 11,
    "legend.columnspacing": 1.5,
    "legend.handletextpad": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Fixed per-selector colors (control arms reuse the hue, dashed) so the same
# selector always reads the same across plots/runs.
SELECTOR_COLORS = {
    "felix": "#1f77b4", "async_oort": "#1f77b4",
    "oort": "#ff7f0e",
    "refl": "#2ca02c", "refl_oort": "#2ca02c",
    "feddance": "#d62728",
    "fedbuff": "#9467bd", "fedavg": "#8c564b", "random": "#7f7f7f",
}


def color_for(label: str) -> Optional[str]:
    key = str(label).lower()
    for name, c in SELECTOR_COLORS.items():
        if name in key:
            return c
    return None


# --- config stamp -----------------------------------------------------------


def config_stamp(run_dir: str) -> str:
    """Compact one-line experiment-config string from snapshot/exec config."""
    import glob
    import json
    try:
        import yaml
    except Exception:
        yaml = None
    info = {}
    # prefer execution_config.yaml (has selector/agg_goal), fall back to snapshot
    for fn in ("execution_config.yaml", "snapshot.yaml"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p) and yaml is not None:
            try:
                info = yaml.safe_load(open(p)) or {}
                break
            except Exception:
                continue
    exp = (info.get("experiment") or {})
    tr = exp.get("trainer", {})
    ds = tr.get("dataset", {})
    agg = info.get("aggregator", {}) or exp.get("aggregator", {})
    ds_cfg = ((tr.get("config_overrides") or {}).get("hyperparameters") or {}).get(
        "data_streaming", {}
    )
    name = exp.get("name", os.path.basename(run_dir))
    sel = agg.get("selector", "?")
    alpha = ds.get("alpha", ds.get("dirichlet_alpha", "?"))
    n = tr.get("num_trainers", "?")
    avail = (tr.get("availability", {}) or {}).get("mode", "?")
    tm = tr.get("time_mode", "?")
    ag = agg.get("agg_goal", "?")
    stream = "off"
    if str(ds_cfg.get("enabled", "False")) == "True":
        stream = f"T={ds_cfg.get('full_data_available_after_s','?')}s"
    else:
        # snapshots don't serialize config_overrides; infer streaming from
        # telemetry (a trainer round with visible < total samples).
        stream = _infer_streaming(os.path.join(run_dir, "telemetry")) or "off"
    return (f"{name} | {sel} | n={n} α={alpha} | {avail} | "
            f"stream:{stream} | aggGoal={ag} | {tm}")


def _infer_streaming(telemetry_dir: str):
    import glob
    import json
    for p in glob.glob(os.path.join(telemetry_dir, "trainer_*.jsonl"))[:5]:
        try:
            for line in open(p):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("event") == "trainer_round":
                    vs, ts = d.get("visible_samples"), d.get("total_samples")
                    if vs is not None and ts and vs < ts:
                        return "on"
        except Exception:
            continue
    return None


def _save(fig, out_dir: str, file_name: str, stamp: Optional[str]) -> str:
    if not file_name.endswith(".pdf"):
        file_name = os.path.splitext(file_name)[0] + ".pdf"
    os.makedirs(out_dir, exist_ok=True)
    if stamp:
        # config line at the TOP of the figure, above the plot title
        fig.text(0.5, 1.04, stamp, ha="center", va="bottom", fontsize=7,
                 color="0.4", wrap=True)
    fig.tight_layout()
    path = os.path.join(out_dir, file_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend(ax, n_series):
    if n_series > 1:
        ax.legend(fontsize=12, frameon=True)


def no_data_plot(title, out_dir, file_name, stamp=None,
                 note="not applicable / no events for this baseline"):
    """Produce a placeholder PDF so every expected plot exists for every run.

    A missing plot is ambiguous (did the metric not fire, or did the code
    crash?). A placeholder makes the absence explicit and visible.
    "NO DATA" is drawn in large red text so it is impossible to miss when
    scanning a plot directory — prompting investigation of whether the
    metric genuinely didn't fire or the value is truly zero.
    """
    fig, ax = plt.subplots()
    ax.text(0.5, 0.62, "NO DATA", ha="center", va="center",
            fontsize=36, color="red", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.38, note, ha="center", va="center",
            fontsize=11, color="0.4", transform=ax.transAxes,
            style="italic")
    ax.set_title(title)
    ax.set_axis_off()
    return _save(fig, out_dir, file_name, stamp)


# --- core figures (names preserved for existing callers) --------------------


def line_plot(series, x_label, y_label, title, out_dir, file_name,
              stamp=None, logy=False, target=None):
    if not series:
        return no_data_plot(title, out_dir, file_name, stamp)
    fig, ax = plt.subplots()
    plotted = False
    for label, (xs, ys) in series.items():
        if xs is None or ys is None or len(xs) == 0:
            continue
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.5, label=label,
                color=color_for(label))
        plotted = True
    if not plotted:
        plt.close(fig)
        return no_data_plot(title, out_dir, file_name, stamp)
    if target is not None:
        ax.axhline(target, ls="--", color="0.5", lw=1)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend(ax, len(series))
    return _save(fig, out_dir, file_name, stamp)


def banded_line(x, mean, lo, hi, x_label, y_label, title, out_dir, file_name,
                stamp=None, marker_x=None, color="#1f77b4"):
    """Mean line with min/max (or P-band) shaded region."""
    if not len(x):
        return None
    fig, ax = plt.subplots()
    ax.plot(x, mean, color=color, lw=2, label="mean")
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, label="min-max")
    if marker_x is not None:
        ax.axvline(marker_x, ls="--", color="0.5", lw=1, label="config horizon")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    return _save(fig, out_dir, file_name, stamp)


def scatter_diag(x, y, x_label, y_label, title, out_dir, file_name, stamp=None,
                 groups=None, group_labels=None):
    """Scatter with y=x reference diagonal (expected-vs-actual style)."""
    if not len(x):
        return None
    x = np.asarray(x, float); y = np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(5, 5))
    if groups is not None:
        g = np.asarray(groups)
        for gv, gl, c in group_labels:
            m = g == gv
            if m.any():
                ax.scatter(x[m], y[m], s=14, alpha=0.5, color=c, label=gl)
        ax.legend(fontsize=12)
    else:
        ax.scatter(x, y, s=14, alpha=0.5, color="#1f77b4")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, ls="--", color="0.4", lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def hist_plot(values, x_label, title, out_dir, file_name, stamp=None, vline=0.0):
    vals = [v for v in values if v is not None]
    if not vals:
        return no_data_plot(title, out_dir, file_name, stamp)
    fig, ax = plt.subplots()
    arr = np.asarray(vals, float)
    counts, bins, patches = ax.hist(arr, bins=min(40, max(8, len(arr) // 5)),
                                    color="#1f77b4", alpha=0.8)
    # percentage of total on top of each non-empty bar
    tot = counts.sum()
    if tot:
        for c, p in zip(counts, patches):
            if c > 0:
                ax.annotate(f"{100 * c / tot:.0f}%",
                            (p.get_x() + p.get_width() / 2, c),
                            ha="center", va="bottom", fontsize=7, color="0.3")
    if vline is not None:
        ax.axvline(vline, color="red", ls="--", lw=1.5)
    for q, lab in [(0.5, "P50"), (0.9, "P90"), (0.99, "P99")]:
        xv = float(np.quantile(arr, q))
        ax.axvline(xv, color="0.3", ls=":", lw=1)
        ax.annotate(f"{lab}={xv:.3g}", (xv, 0), textcoords="offset points",
                    xytext=(3, 12), fontsize=10, rotation=90, color="0.3")
    ax.set_xlabel(x_label)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def signed_bar(categories, values, x_label, y_label, title, out_dir, file_name,
               stamp=None):
    if not len(values):
        return None
    fig, ax = plt.subplots()
    v = np.asarray(values, float)
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in v]
    ax.bar(range(len(v)), v, color=colors, alpha=0.85)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def signed_bar_line(x, bar_values, line_values, x_label, bar_label, line_label,
                    title, out_dir, file_name, stamp=None):
    """Signed bars (green/red) on the primary axis + a line on a secondary axis
    — e.g. per-eval accuracy delta (bars) with cumulative accuracy (line)."""
    if not len(bar_values):
        return None
    fig, ax1 = plt.subplots()
    v = np.asarray(bar_values, float)
    ax1.bar(x, v, color=["#2ca02c" if b >= 0 else "#d62728" for b in v], alpha=0.7)
    ax1.axhline(0, color="0.3", lw=1)
    ax1.set_xlabel(x_label); ax1.set_ylabel(bar_label)
    ax2 = ax1.twinx()
    ax2.plot(x, line_values, color="#1f77b4", lw=2, marker="o", ms=3, label=line_label)
    ax2.set_ylabel(line_label, color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title(title); ax1.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def cdf_plot(values, x_label, title, out_dir, file_name, stamp=None):
    vals = [v for v in values if v is not None]
    if not vals:
        return no_data_plot(title, out_dir, file_name, stamp)
    arr = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(arr) + 1) / len(arr)
    fig, ax = plt.subplots()
    ax.plot(arr, y, color="tab:red", linewidth=2)
    for q, label in [(0.5, "P50"), (0.9, "P90"), (0.99, "P99")]:
        xv = float(np.quantile(arr, q))
        ax.axvline(xv, color="0.4", ls=":", lw=1)
        ax.annotate(f"{label}={xv:.3g}", (xv, q), textcoords="offset points",
                    xytext=(5, -12), fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def cdf_multi(series, x_label, title, out_dir, file_name, stamp=None):
    """Overlay multiple CDFs (one per label) for apples-to-apples comparison.
    ``series`` is {label: [values...]}. Each curve gets its own color + legend."""
    series = {lab: [v for v in vals if v is not None]
              for lab, vals in series.items()}
    series = {lab: vals for lab, vals in series.items() if vals}
    if not series:
        return no_data_plot(title, out_dir, file_name, stamp)
    fig, ax = plt.subplots()
    for lab, vals in series.items():
        arr = np.sort(np.asarray(vals, dtype=float))
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, y, lw=2, label=lab)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def scatter_plot(x, y, groups, x_label, y_label, title, out_dir, file_name, stamp=None):
    if len(x) == 0:
        return None
    fig, ax = plt.subplots()
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if groups is not None:
        g = np.asarray(groups, dtype=bool)
        ax.scatter(x[~g], y[~g], s=16, alpha=0.5, color="tab:gray", label="eligible")
        ax.scatter(x[g], y[g], s=24, alpha=0.8, color="tab:red", label="selected")
        ax.legend(fontsize=12)
    else:
        ax.scatter(x, y, s=16, alpha=0.6, color="tab:blue")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def stacked_area(x, series, x_label, y_label, title, out_dir, file_name, stamp=None):
    if not series or len(x) == 0:
        return None
    labels = list(series.keys())
    ys = [np.asarray(series[k], dtype=float) for k in labels]
    fig, ax = plt.subplots()
    ax.stackplot(x, *ys, labels=labels, alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def stacked_bar(categories, segments, y_label, title, out_dir, file_name, stamp=None,
                horizontal=False):
    if not segments or len(categories) == 0:
        return None
    n = len(categories)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), 4) if not horizontal
                           else (7, max(3, n * 0.25)))
    bottom = np.zeros(n, dtype=float)
    for label, vals in segments.items():
        v = np.asarray(vals, dtype=float)
        if horizontal:
            ax.barh(range(n), v, left=bottom, label=label)
        else:
            ax.bar(range(n), v, bottom=bottom, label=label)
        bottom += v
    if horizontal:
        ax.set_yticks(range(n)); ax.set_yticklabels([str(c) for c in categories], fontsize=7)
        ax.set_xlabel(y_label)
    else:
        ax.set_xticks(range(n)); ax.set_xticklabels([str(c) for c in categories],
                                                    rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=11)
    ax.grid(True, axis=("x" if horizontal else "y"), alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def bar_plot(categories, values, y_label, title, out_dir, file_name, stamp=None):
    if len(categories) == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.3), 3.2))
    ax.bar(range(len(categories)), values, color="tab:blue", alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(c) for c in categories], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def heatmap(matrix, x_label, y_label, title, out_dir, file_name, stamp=None,
            cmap="viridis", cbar_label=None, yticklabels=None, discrete=None):
    """Trainer x round style heatmap (matrix: rows=y, cols=x).

    discrete: optional [(value, label, color), ...] for categorical states —
    renders a discrete colormap + a legend instead of a continuous colorbar.
    """
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, m.shape[0] * 0.12)))
    if discrete:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
        vals = [v for v, _, _ in discrete]
        cmap_d = ListedColormap([c for _, _, c in discrete])
        bounds = [vals[0] - 0.5] + [v + 0.5 for v in vals]
        ax.imshow(m, aspect="auto", interpolation="nearest", cmap=cmap_d,
                  norm=BoundaryNorm(bounds, cmap_d.N))
        ax.legend(handles=[Patch(facecolor=c, edgecolor="0.4", label=lab)
                           for _, lab, c in discrete],
                  fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        im = ax.imshow(m, aspect="auto", interpolation="nearest", cmap=cmap)
        cb = fig.colorbar(im, ax=ax)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=12)
    if yticklabels is not None and len(yticklabels) <= 40:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return _save(fig, out_dir, file_name, stamp)


def dual_axis_line(x, y1, y2, x_label, y1_label, y2_label, title, out_dir,
                   file_name, stamp=None):
    if not len(x):
        return None
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color="#1f77b4", lw=2, label=y1_label)
    ax1.set_xlabel(x_label); ax1.set_ylabel(y1_label, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="#d62728", lw=2, ls="--", label=y2_label)
    ax2.set_ylabel(y2_label, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def lorenz_plot(series, title, out_dir, file_name, stamp=None):
    """Lorenz curve(s) of selection-count inequality. ``series`` is
    {label: [counts...]}; each label gets a curve x=cumulative fraction of
    trainers (sorted ascending), y=cumulative fraction of selections, with its
    Gini in the legend. The diagonal is perfect equality (Gini 0); the more the
    curve bows below it, the more concentrated the selections. Compact regardless
    of trainer count — the readable replacement for a 300-bar frequency chart."""
    def gini(a):
        a = np.sort(np.asarray(a, float))
        if a.size == 0 or a.sum() == 0:
            return 0.0
        n = len(a)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))

    curves = {lab: c for lab, c in series.items() if len(c) > 0 and sum(c) > 0}
    if not curves:
        return None
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="equality")
    for lab, counts in curves.items():
        a = np.sort(np.asarray(counts, float))
        cum = np.cumsum(a) / a.sum()
        x = np.arange(1, len(a) + 1) / len(a)
        ax.plot(np.concatenate([[0], x]), np.concatenate([[0], cum]), lw=2,
                label=f"{lab} (Gini={gini(counts):.2f}, n={len(counts)})")
    ax.set_xlabel("cumulative fraction of trainers (least- to most-selected)")
    ax.set_ylabel("cumulative fraction of selections")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)
