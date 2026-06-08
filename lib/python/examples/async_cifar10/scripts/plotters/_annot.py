"""Shared percentile annotation helper for all CDF and line plots.

Usage:
    from plotters._annot import annotate_percentiles

    xs, ys = cdf_xy(my_values)
    ax.plot(xs, ys, color=color, label=label)
    annotate_percentiles(ax, my_values, color=color, label=label)

annotate_percentiles places P50/P90/P99 vertical tick-marks + text in the
line's own color.  When `below=True` the text is placed in a compact table
below the axes instead of inline (use this when multiple series crowd the plot).
"""

from __future__ import annotations

from typing import Sequence


_DEFAULT_PS = (50, 90, 99)


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100)))
    return sorted_vals[idx]


def annotate_percentiles(
    ax,
    vals: Sequence[float],
    *,
    color: str,
    label: str = "",
    ps: tuple[int, ...] = _DEFAULT_PS,
    below: bool = False,
    linestyle: str = "--",
    alpha: float = 0.55,
    fontsize: int = 7,
) -> None:
    """Annotate a CDF/line axes with P50/P90/P99 markers in the series color.

    ax     : matplotlib Axes the series is drawn on (must be a CDF: x=value, y=fraction)
    vals   : raw (unsorted) values used to compute percentiles
    color  : color of the parent series
    label  : short name used in annotations
    ps     : percentile list, default (50, 90, 99)
    below  : if True, put text in a legend-style block below the plot, not inline
    """
    if not vals:
        return
    sv = sorted(float(v) for v in vals if v is not None)
    if not sv:
        return

    pcts = {p: _pct(sv, p) for p in ps}
    fracs = {p: (p / 100) for p in ps}  # y position on CDF

    if not below:
        for p, xv in pcts.items():
            frac = fracs[p]
            # vertical dashed line at the percentile value
            ax.axvline(xv, ymin=0, ymax=frac, color=color, ls=linestyle,
                       lw=0.8, alpha=alpha)
            ax.annotate(
                f"P{p}={xv:.2g}",
                xy=(xv, frac),
                xytext=(3, 0),
                textcoords="offset points",
                color=color,
                fontsize=fontsize,
                va="center",
            )
    else:
        # Collect into the axes' user_data for the caller to flush later via
        # flush_percentile_table(ax).  Each entry: (label, p, value, color).
        _store = getattr(ax, "_pct_table", [])
        for p, xv in pcts.items():
            _store.append((label, p, xv, color))
        ax._pct_table = _store


def flush_percentile_table(ax, fontsize: int = 7) -> None:
    """Render the accumulated percentile table below the axes.

    Call this once after all series have been annotated with below=True.
    """
    rows = getattr(ax, "_pct_table", [])
    if not rows:
        return

    # Build column groups: label | P50 | P90 | P99
    ps_all = sorted({r[1] for r in rows})
    labels_all = list(dict.fromkeys(r[0] for r in rows))
    lookup = {(r[0], r[1]): (r[2], r[3]) for r in rows}

    header = f"{'':20s}" + "".join(f"  P{p:>3}" for p in ps_all)
    lines = [header]
    for lbl in labels_all:
        row_vals = []
        first_color = None
        for p in ps_all:
            entry = lookup.get((lbl, p))
            if entry:
                row_vals.append(f"{entry[0]:>7.3g}")
                if first_color is None:
                    first_color = entry[1]
            else:
                row_vals.append(f"{'N/A':>7}")
        lines.append(f"{lbl[:20]:20s}" + "  ".join(row_vals))

    text = "\n".join(lines)
    ax.figure.text(
        0.5, -0.05, text, ha="center", va="top",
        fontsize=fontsize, family="monospace",
        transform=ax.transAxes,
    )
