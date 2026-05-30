# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reusable, dependency-light matplotlib helpers for telemetry analysis.

These are the plotting primitives the run analyzer composes. They are kept
generic (lists / arrays in, PNG out) so future comparative plots reuse them
rather than duplicating matplotlib boilerplate. The CDF renderer mirrors the
percentile-annotated style of the legacy ``scripts/plotters/cdf_plot.py``.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; never needs a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, out_dir: str, file_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def cdf_plot(
    values: Sequence[float],
    x_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Percentile-annotated CDF (P50/P90/P99). Returns saved path or None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    count, bins = np.histogram(arr, bins=min(5000, max(10, len(arr))))
    cdf = np.cumsum(count / count.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bins[1:], cdf, color="tab:red", linewidth=2)
    for q, marker, label in [(0.5, "^", "P50"), (0.9, "o", "P90"), (0.99, "x", "P99")]:
        xv = float(np.quantile(arr, q))
        ax.plot(xv, q, marker=marker, color="black", markersize=8)
        ax.annotate(
            f"{label}={xv:.3g}",
            (xv, q),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98,
        0.02,
        f"n={len(arr)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    return _save(fig, out_dir, file_name)


def line_plot(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    x_label: str,
    y_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Overlay multiple (x, y) series. ``series`` maps label -> (xs, ys)."""
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label, (xs, ys) in series.items():
        if xs is None or ys is None or len(xs) == 0:
            continue
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.2, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    return _save(fig, out_dir, file_name)


def stacked_area(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    x_label: str,
    y_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Stacked area chart, e.g. availability composition over rounds."""
    if not series or len(x) == 0:
        return None
    labels = list(series.keys())
    ys = [np.asarray(series[k], dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(x, *ys, labels=labels, alpha=0.85)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name)


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    groups: Optional[Sequence[bool]],
    x_label: str,
    y_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Scatter; if ``groups`` (bool per point) given, color selected vs not."""
    if len(x) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if groups is not None:
        g = np.asarray(groups, dtype=bool)
        ax.scatter(x[~g], y[~g], s=18, alpha=0.5, color="tab:gray", label="eligible")
        ax.scatter(x[g], y[g], s=28, alpha=0.8, color="tab:red", label="selected")
        ax.legend(fontsize=8)
    else:
        ax.scatter(x, y, s=18, alpha=0.6, color="tab:blue")
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name)


def stacked_bar(
    categories: Sequence[str],
    segments: dict[str, Sequence[float]],
    y_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Stacked bar per category, e.g. trainer real-GPU vs sim vs wait time."""
    if not segments or len(categories) == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 0.5), 5))
    bottom = np.zeros(len(categories), dtype=float)
    for label, vals in segments.items():
        v = np.asarray(vals, dtype=float)
        ax.bar(range(len(categories)), v, bottom=bottom, label=label)
        bottom += v
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(c) for c in categories], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name)


def bar_plot(
    categories: Sequence[str],
    values: Sequence[float],
    y_label: str,
    title: str,
    out_dir: str,
    file_name: str,
) -> Optional[str]:
    """Simple bar chart, e.g. selection frequency per trainer."""
    if len(categories) == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 0.4), 5))
    ax.bar(range(len(categories)), values, color="tab:blue", alpha=0.8)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(c) for c in categories], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name)
