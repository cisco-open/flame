"""Parity checker report formatting: stdout table + JSON + PNG.

Grouped by §3 section (A–H), each line tier-tagged.  Used by cli.py.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .checks import overall_verdict

# ── status decorators ─────────────────────────────────────────────────────────

_ICONS = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]", "SKIP": "[--]"}
_TIER_TAG = {"EXACT": "EXACT", "INV": "INV  ", "DIST": "DIST ",
             "DIAG": "DIAG ", "": "     "}


def _icon(ok: bool, skip: bool = False, warn: bool = False) -> str:
    if skip:
        return _ICONS["SKIP"]
    if warn:
        return _ICONS["WARN"]
    return _ICONS["PASS"] if ok else _ICONS["FAIL"]


def _status_str(res: dict, warn_names: set, fail_names: set) -> tuple:
    name = res.get("_name", "")
    ok = res.get("ok", True)
    skip = res.get("status") == "SKIP" or res.get("note", "").startswith("K10:")
    is_warn = name in warn_names or (not ok and name not in fail_names)
    icon = _icon(ok, skip=skip, warn=(not ok and is_warn))
    tier = _TIER_TAG.get(res.get("tier", ""), "     ")
    return icon, tier


# ── section definitions ───────────────────────────────────────────────────────

_SECTIONS = [
    ("H", "Clock & Throughput  ← new enforced core", [
        ("K10 vclock telemetry present",       "vclock_telemetry"),
        ("K1  vclock monotone (sim)",           "sim_commit_monotone"),
        ("K7  sim_rate in [0.01, 100]",         "sim_rate"),
        ("K2  rounds-per-virtual-second parity","throughput"),
        ("K3  per-round advance distribution",  "per_round_advance"),
        ("K4  overlap factor (diagnostic)",     "overlap_factor"),
        ("K8  terminal-state parity",           "terminal_state"),
        ("U2  total commits at matched budget", "total_commits"),
        ("K5  failsafe ceiling",                "failsafe"),
        ("K9  stopped by budget, not cap",      "budget_not_cap"),
    ]),
    ("E", "Update Processing", [
        ("P1  aggregation sequence",            "aggregation_sequence"),
        ("P3  trainer_speed_s parity (control)","trainer_speed"),
    ]),
    ("D", "Updates Received & Ordering", [
        ("U1  first divergence",                "first_divergence_summary"),
        ("U3  staleness distribution",          "staleness"),
        ("U4  agg_goal_count cycles (real)",    "agg_goal_cycles_real"),
        ("U4  agg_goal_count cycles (sim)",     "agg_goal_cycles_sim"),
        ("U5  inter-arrival order (Spearman ρ)","inter_arrival_order"),
    ]),
    ("B", "Selection", [
        ("S1  per-round Jaccard (selection)",   "selection"),
        ("S2  participation frequency",         "participation"),
    ]),
    ("C", "Training", [
        ("T3  GPU budget respected (real)",     "gpu_budget_real"),
        ("T3  GPU budget respected (sim)",      "gpu_budget_sim"),
        ("K6  sim_send_ts correctness",         "sim_send_ts"),
    ]),
    ("F", "Statistical Utility", [
        ("F1-3 utility distributions",          "utility"),
    ]),
    ("G", "Convergence", [
        ("C1-2 accuracy/loss by FL round",      "convergence"),
    ]),
]


def _fmt_metric(name: str, res: dict) -> list:
    """Return per-check detail lines (after the header line)."""
    lines = []
    note = res.get("note") or res.get("status")
    if note and note not in ("PASS", "FAIL", "WARN"):
        lines.append(f"         note: {note}")

    if name == "throughput":
        lines += [
            f"         sim:  {res.get('sim_rounds')} rounds / {res.get('final_vclock_s')}s vclock "
            f"= {res.get('sim_s_per_round')}s/round",
            f"         real: {res.get('real_rounds')} rounds / {res.get('real_wall_elapsed_s')}s wall "
            f"= {res.get('real_s_per_round')}s/round",
            f"         rel_diff={res.get('rel_diff')}  tol={res.get('tol')}",
        ]
    elif name == "per_round_advance":
        lines += [
            f"         sim mean advance:  {res.get('sim_mean_advance_s')}s/round",
            f"         real mean advance: {res.get('real_mean_advance_s')}s/round",
            f"         KS={res.get('ks_stat')} (≤{res.get('ks_tol')})  "
            f"mean_rel_diff={res.get('mean_rel_diff')} (≤{res.get('mean_tol_rel')})",
        ]
    elif name == "overlap_factor":
        interp = res.get("interpretation")
        if interp:
            lines.append(f"         {interp}")
        abs_diff = res.get("abs_diff")
        tol = res.get("tol")
        if abs_diff is not None:
            lines.append(f"         abs_diff={abs_diff}  tol={tol}")
    elif name == "terminal_state":
        lines += [
            f"         matched budget V={res.get('matched_virtual_budget_s')}s",
            f"         rounds: sim={res.get('sim_rounds_at_V')} real={res.get('real_rounds_at_V')} "
            f"rel_diff={res.get('rounds_rel_diff')} (≤{res.get('rounds_tol')})",
            f"         trainers: sim={res.get('sim_trainers_at_V')} real={res.get('real_trainers_at_V')} "
            f"rel_diff={res.get('trainers_rel_diff')} (≤{res.get('trainers_tol')})",
        ]
    elif name == "total_commits":
        lines += [
            f"         V={res.get('matched_virtual_budget_s')}s  "
            f"sim={res.get('n_sim_commits')} real={res.get('n_real_commits')}  "
            f"rel_diff={res.get('rel_diff')} (≤{res.get('tol')})",
        ]
    elif name == "vclock_telemetry":
        lines += [
            f"         {res.get('n_with_vclock')}/{res.get('n_total_events')} agg_round events "
            f"carry vclock_now",
        ]
    elif name == "sim_commit_monotone":
        lines += [
            f"         {res.get('n_stamped')} vclock-stamped events, "
            f"monotone={res.get('monotone')}",
        ]
    elif name == "sim_rate":
        lines += [
            f"         sim_rate={res.get('sim_rate')} virtual-s/wall-s  "
            f"vclock={res.get('final_vclock_s')}s wall={res.get('wall_elapsed_s')}s",
        ]
    elif name == "staleness":
        lines += [
            f"         real_mean={res.get('real_mean')}  sim_mean={res.get('sim_mean')}  "
            f"KS={res.get('ks_stat')}  all_nonneg={res.get('all_nonnegative')}",
        ]
    elif name == "selection":
        sel = res.get("selector")
        gated = res.get("gated")
        lines += [
            f"         rounds_compared={res.get('rounds_compared')}  "
            f"mean_jaccard={res.get('mean_jaccard')}  "
            f"exact_match_frac={res.get('exact_match_frac')}",
            f"         selector={sel}  gated(WARN)={gated}",
        ]
    elif name == "participation":
        lines += [
            f"         avg_diff={res.get('avg_diff')}  max_diff={res.get('max_diff')}",
        ]
    elif name == "trainer_speed":
        if res.get("n_real"):
            lines += [
                f"         KS={res.get('ks_stat')} (≤{res.get('ks_tol')})  "
                f"real_mean={res.get('real_mean_speed_s')}s "
                f"sim_mean={res.get('sim_mean_speed_s')}s  "
                f"real_max={res.get('real_max_speed_s')}s "
                f"sim_max={res.get('sim_max_speed_s')}s",
            ]
    elif name in ("gpu_budget_real", "gpu_budget_sim"):
        mof = res.get("mean_overrun_frac")
        if mof is not None:
            lines += [
                f"         mean_overrun_frac={mof}  "
                f"trainers_with_overrun={res.get('trainers_with_any_overrun')}",
            ]
    elif name == "sim_send_ts":
        issues = res.get("issues", [])
        if issues:
            for iss in issues[:3]:
                lines.append(f"         [!] {iss}")
    elif name == "utility":
        lines += [
            f"         max_KS={res.get('max_ks_stat')} (≤{res.get('max_ks_tol')})  "
            f"avg_mean_diff={res.get('avg_mean_utility_diff')}  "
            f"n_trainers={res.get('n_trainers')}",
        ]
    elif name == "convergence":
        lines += [
            f"         eval_rounds={res.get('eval_rounds_compared')}  "
            f"avg_acc_diff={res.get('avg_accuracy_diff')} (≤{res.get('acc_tol')})  "
            f"avg_loss_diff={res.get('avg_loss_diff')}",
        ]
    elif name == "inter_arrival_order":
        lines += [
            f"         mean_spearman_rho={res.get('mean_spearman_rho')} "
            f"(≥{res.get('min_rho')})  n_rounds={res.get('n_rounds')}",
        ]
    elif name == "failsafe":
        ov = res.get("overshoot_frac")
        if ov is not None:
            lines += [
                f"         wall={res.get('wall_elapsed_s')}s budget={res.get('budget_s')}s "
                f"overshoot={ov:.1%}  failsafe_fired={res.get('failsafe_fired')}",
            ]
    elif name == "budget_not_cap":
        for w in res.get("warnings", []):
            lines.append(f"         [!] {w}")
        if not res.get("warnings"):
            lines.append(
                f"         real_max_round={res.get('real_max_round')}  "
                f"sim_max_round={res.get('sim_max_round')}"
            )
    elif name in ("agg_goal_cycles_real", "agg_goal_cycles_sim"):
        bad = res.get("rounds_over_goal", [])
        if bad:
            lines.append(f"         {len(bad)} rounds exceeded agg_goal: {bad[:5]}")
    elif name == "first_divergence_summary":
        idx = res.get("index")
        lines.append(
            f"         first_divergence_index={idx}  "
            f"real_len={res.get('real_len')}  sim_len={res.get('sim_len')}"
        )
        if idx is not None:
            lines.append(f"         real context: {res.get('real')}")
            lines.append(f"         sim  context: {res.get('sim')}")

    return [l for l in lines if l.strip()]


def print_report(results: dict, strict: bool = False, lenient: bool = False,
                 real_label: str = "", sim_label: str = "") -> bool:
    """Print section-grouped report to stdout.  Returns True if overall passed."""
    passed, failures, warnings = overall_verdict(results, strict=strict, lenient=lenient)
    fail_set, warn_set = set(failures), set(warnings)

    width = 72
    print(f"\n{'='*width}")
    if real_label or sim_label:
        print(f"  PARITY REPORT  real={real_label}  sim={sim_label}")
    else:
        print("  PARITY REPORT: real vs simulated")
    print(f"{'='*width}\n")

    for sec_id, sec_title, checks in _SECTIONS:
        # Skip empty sections
        if not any(key in results for _, key in checks):
            continue
        print(f"  ── §3.{sec_id} {sec_title}")
        for label, key in checks:
            res = results.get(key)
            if res is None:
                continue
            res["_name"] = key
            icon, tier = _status_str(res, warn_set, fail_set)
            ok = res.get("ok", True)
            status_word = "PASS" if ok else ("WARN" if key in warn_set else "FAIL")
            if res.get("status") == "SKIP" or (
                    res.get("note", "").startswith("K10:") and key not in ("vclock_telemetry",)):
                status_word = "SKIP"
                icon = _ICONS["SKIP"]
            print(f"  {icon} [{tier}] [{status_word:4s}] {label}")
            for detail in _fmt_metric(key, res):
                print(detail)
        print()

    verdict_word = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    if warnings and passed:
        verdict_word += f"  ({len(warnings)} warning(s))"
    icon = _ICONS["PASS"] if passed else _ICONS["FAIL"]
    print(f"{'='*width}")
    print(f"  {icon} {verdict_word}")
    if failures:
        print(f"  Failed: {', '.join(failures)}")
    if warnings:
        print(f"  Warned: {', '.join(warnings)}")
    print(f"{'='*width}\n")
    return passed


def write_json(results: dict, path: str, extra: Optional[dict] = None) -> None:
    """Write full results dict as JSON."""
    out = dict(results)
    if extra:
        out.update(extra)
    # Remove internal _name keys
    for v in out.values():
        if isinstance(v, dict):
            v.pop("_name", None)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[parity] JSON results → {path}")


def write_plot(results: dict, path: str,
               real_label: str = "real", sim_label: str = "sim") -> None:
    """Write multi-panel PNG: virtual-time trajectory + per-round advance + overlap factor."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[parity] matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Parity: {real_label} vs {sim_label}", fontsize=12)

    c_real, c_sim = "#4e79a7", "#e15759"

    # ── panel 1: throughput summary bar ──
    ax = axes[0]
    th = results.get("throughput", {})
    sim_spr = th.get("sim_s_per_round")
    real_spr = th.get("real_s_per_round")
    if sim_spr is not None and real_spr is not None:
        ax.bar(["real", "sim"], [real_spr, sim_spr], color=[c_real, c_sim], alpha=0.8)
        ax.set_ylabel("seconds / FL round")
        ax.set_title("K2: s/round (lower is faster)\n"
                     f"real={real_spr:.1f}s  sim={sim_spr:.1f}s  "
                     f"rel_diff={th.get('rel_diff', '?')}")
        ax.axhline(real_spr, color=c_real, ls="--", lw=1)
    else:
        ax.text(0.5, 0.5, "K2 data not available\n(K10: missing vclock)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("K2: throughput (s/round)", fontsize=10)

    # ── panel 2: overlap factor ──
    ax2 = axes[1]
    ov = results.get("overlap_factor", {})
    sim_ov = ov.get("sim_overlap_factor")
    real_ov = ov.get("real_overlap_factor")
    if sim_ov is not None and real_ov is not None:
        ax2.bar(["real", "sim"], [real_ov, sim_ov], color=[c_real, c_sim], alpha=0.8)
        ax2.axhline(1.0, color="gray", ls="--", lw=1, label="1.0 = no overlap")
        ax2.set_ylabel("overlap factor")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "K4 data not available",
                 ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("K4: overlap factor\n(real > sim = sim missing async pipelining)", fontsize=10)

    # ── panel 3: terminal state ──
    ax3 = axes[2]
    ts = results.get("terminal_state", {})
    sim_r = ts.get("sim_rounds_at_V")
    real_r = ts.get("real_rounds_at_V")
    V = ts.get("matched_virtual_budget_s")
    if sim_r is not None and real_r is not None:
        ax3.bar(["real", "sim"], [real_r, sim_r], color=[c_real, c_sim], alpha=0.8)
        ax3.set_ylabel("FL rounds")
        ax3.set_title(
            f"K8: FL rounds at matched V={V}s\n"
            f"real={real_r}  sim={sim_r}  rel_diff={ts.get('rounds_rel_diff')}",
            fontsize=10,
        )
    else:
        ax3.text(0.5, 0.5, "K8 data not available",
                 ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("K8: rounds at matched virtual budget", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[parity] Plot → {path}")


def roll_up_table(batch_results: dict) -> None:
    """Print a roll-up table for --batch mode: one row per baseline."""
    print(f"\n{'='*90}")
    print(f"  {'Baseline':<20} {'rounds(real/sim)':<20} {'s/round(real/sim)':<22} "
          f"{'K2':>5} {'K3':>5} {'K10':>5} {'P3':>5} {'overall':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*22} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
    for baseline, res in sorted(batch_results.items()):
        results = res.get("results", {})
        th = results.get("throughput", {})
        passed, failures, _ = overall_verdict(results)

        rounds_str = (
            f"{th.get('real_rounds','?')}/{th.get('sim_rounds','?')}"
            if results else "n/a"
        )
        spr_str = (
            f"{th.get('real_s_per_round','?')}/{th.get('sim_s_per_round','?')}"
            if results else "n/a"
        )

        def _s(key):
            r = results.get(key, {})
            return "OK" if r.get("ok") else ("SK" if r.get("status") == "SKIP" else "XX")

        overall = "PASS" if passed else "FAIL"
        print(f"  {baseline:<20} {rounds_str:<20} {spr_str:<22} "
              f"  {_s('throughput'):>3}   {_s('per_round_advance'):>3}   "
              f"{_s('vclock_telemetry'):>3}   {_s('trainer_speed'):>3}   {overall:>8}")
    print(f"{'='*90}\n")
