"""Parity checker report formatting: stdout table + JSON + PNG.

Grouped by S3 section (A–H), each line tier-tagged.  Used by cli.py.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .checks import overall_verdict, check_role, check_stage

# ── status decorators ─────────────────────────────────────────────────────────

_ICONS = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]", "SKIP": "[--]",
          "DOWN": "[~~]", "LOWC": "[??]"}
_TIER_TAG = {"EXACT": "EXACT", "INV": "INV  ", "DIST": "DIST ",
             "DIAG": "DIAG ", "": "     "}
_ROLE_TAG = {"CONTROL": "CTRL", "MECHANISM": "MECH", "EMERGENT": "EMRG",
             "DIAG": "DIAG", "": "    "}


# ── section definitions ───────────────────────────────────────────────────────

# Stage-grouped ladder: foundational (clock) → emergent (convergence).
# The first broken rung whose upstreams pass is the root cause.
_SECTIONS = [
    ("0", "Telemetry Coverage (gate)", [
        ("TC1 field coverage matrix",           "field_coverage"),
        ("K10 vclock telemetry present",        "vclock_telemetry"),
    ]),
    ("1", "Clock / Time-base", [
        ("K1   vclock monotone (sim)",          "sim_commit_monotone"),
        ("K7   sim_rate in [0.01, 100]",        "sim_rate"),
        ("P3   trainer_speed_s (control)",      "trainer_speed"),
        ("K3a  modeled-compute advance",        "modeled_compute_advance"),
        ("K3b  overhead residual",              "overhead_residual"),
        ("K4   overlap factor (diagnostic)",    "overlap_factor"),
        ("K3   per-round advance distribution", "per_round_advance"),
        ("K2   rounds-per-virtual-second",      "throughput"),
    ]),
    ("2", "Availability", [
        ("A1  avail_composition parity",        "avail_composition"),
        ("A2  num_eligible / num_candidates",   "eligibility"),
        ("A2b eligible-pool speed composition", "eligible_speed"),
        ("A3  trace time-base consistency",     "avail_timebase"),
        ("A4  per-trainer duty-cycle",          "duty_cycle"),
    ]),
    ("3", "Selection", [
        ("S3/4 num_chosen / in_flight / eff_c", "selection_detail"),
        ("Sr   in-flight residence / carry-over","residence"),
        ("A2c  selected-vs-pool speed bias",    "selection_bias"),
        ("Sx   selector score-term localize",   "selector_score"),
        ("Sd   preferred-duration penalty bind","preferred_duration"),
        ("S2   participation frequency",        "participation"),
        ("Sdet decision determinism (seed)",    "decision_determinism"),
        ("S1   per-round Jaccard",              "selection"),
    ]),
    ("4", "Dispatch & Training", [
        ("T2  training_budget_s (control)",     "training_budget"),
        ("T_  pre_train_s phase",               "phase_pre_train"),
        ("T_  weights_to_gpu_s phase",          "phase_weights_to_gpu"),
        ("T_  gpu_compute_s phase",             "phase_gpu_compute"),
        ("T_  mqtt_fetch_s phase",              "phase_mqtt_fetch"),
        ("T_  weights_to_ram_s phase",          "phase_weights_to_ram"),
        ("T_  post_train_s phase",              "phase_post_train"),
        ("T_  per-phase combined (diagnostic)", "trainer_phase"),
        ("T3  GPU budget respected (real)",     "gpu_budget_real"),
        ("T3  GPU budget respected (sim)",      "gpu_budget_sim"),
        ("K6  sim_send_ts correctness",         "sim_send_ts"),
    ]),
    ("5", "Update Return & Ordering", [
        ("U5  inter-arrival order (Spearman)",  "inter_arrival_order"),
        ("U4  agg_goal_count cycles (real)",    "agg_goal_cycles_real"),
        ("U4  agg_goal_count cycles (sim)",     "agg_goal_cycles_sim"),
    ]),
    ("6", "Aggregation", [
        ("U3  staleness distribution",          "staleness"),
        ("P1  aggregation sequence",            "aggregation_sequence"),
        ("U1  first divergence",                "first_divergence_summary"),
    ]),
    ("7", "Statistical Utility", [
        ("F1-3 utility distributions",          "utility"),
    ]),
    ("8", "Emergent Outcomes", [
        ("K8  terminal-state parity",           "terminal_state"),
        ("U2  total commits at matched budget", "total_commits"),
        ("C1  accuracy by FL round",            "convergence"),
        ("C2  loss by FL round",                "convergence_loss"),
    ]),
    ("9", "Budget & Stop Sanity", [
        ("K9  stopped by budget, not cap",      "budget_not_cap"),
        ("K5  failsafe ceiling",                "failsafe"),
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
            f"         KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
            f"mean_rel_diff={res.get('mean_rel_diff')} (<={res.get('mean_tol_rel')})",
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
            f"rel_diff={res.get('rounds_rel_diff')} (<={res.get('rounds_tol')})",
            f"         trainers: sim={res.get('sim_trainers_at_V')} real={res.get('real_trainers_at_V')} "
            f"rel_diff={res.get('trainers_rel_diff')} (<={res.get('trainers_tol')})",
        ]
    elif name == "total_commits":
        lines += [
            f"         V={res.get('matched_virtual_budget_s')}s  "
            f"sim={res.get('n_sim_commits')} real={res.get('n_real_commits')}  "
            f"rel_diff={res.get('rel_diff')} (<={res.get('tol')})",
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
            f"         matched_count_KS={res.get('matched_count_ks')} (<={res.get('ks_tol')})  "
            f"over n_rounds={res.get('n_rounds_matched')}",
            f"         share_KS(full)={res.get('share_ks')}  avg_diff={res.get('avg_diff')}  "
            f"max_diff={res.get('max_diff')} (diag)",
        ]
    elif name == "decision_determinism":
        if res.get("status") != "SKIP":
            lines += [
                f"         seed real={res.get('real_seed')} sim={res.get('sim_seed')}  "
                f"over n_rounds={res.get('n_rounds_compared')}",
                f"         eligible_match={res.get('eligible_match_frac')}  "
                f"decision_match={res.get('decision_match_frac')}  "
                f"chosen_match={res.get('chosen_match_frac')}",
                f"         -> {res.get('verdict')}",
            ]
    elif name == "trainer_speed":
        if res.get("n_real"):
            lines += [
                f"         grid_KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
                f"raw_KS={res.get('raw_ks_stat')}  "
                f"mean_overhead={res.get('mean_overhead_s')}s "
                f"(<={res.get('max_mean_overhead_s')})",
                f"         real_mean={res.get('real_mean_speed_s')}s "
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
            f"         pooled_KS={res.get('pooled_ks_stat')} (<={res.get('max_ks_tol')})  "
            f"[enforced]  gated(per-trainer)={res.get('gated')}",
            f"         per-trainer max_KS={res.get('max_ks_stat')} "
            f"(n>={res.get('min_samples')}: {res.get('n_trainers_well_sampled')}/"
            f"{res.get('n_trainers')})  avg_mean_diff={res.get('avg_mean_utility_diff')}",
        ]
    elif name == "convergence":
        lines += [
            f"         eval_rounds={res.get('eval_rounds_compared')}  "
            f"avg_acc_diff={res.get('avg_accuracy_diff')} (<={res.get('acc_tol')})  "
            f"avg_loss_diff={res.get('avg_loss_diff')}",
        ]
    elif name == "aggregation_sequence":
        lines += [
            f"         exact_set_match_frac={res.get('exact_set_match_frac')}  "
            f"rounds_compared={res.get('rounds_compared')}",
            f"         selector={res.get('selector')}  gated(WARN)={res.get('gated')}",
        ]
    elif name == "inter_arrival_order":
        lines += [
            f"         mean_spearman_rho={res.get('mean_spearman_rho')} "
            f"(>={res.get('min_rho')})  n_rounds={res.get('n_rounds')}",
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
    elif name == "avail_composition":
        per_state = res.get("per_state", {})
        for state, info in sorted(per_state.items()):
            lines.append(
                f"         {state:12s}: real={info.get('real_mean')}  "
                f"sim={info.get('sim_mean')}  rel_diff={info.get('rel_diff')}"
            )
        viol = res.get("violations")
        if viol:
            lines.append(f"         violations (>{res.get('tol_rel'):.0%}): {viol}")
    elif name == "eligibility":
        lines += [
            f"         num_eligible : KS={res.get('ks_eligible')} (<={res.get('warn_ks')})  "
            f"real_mean={res.get('real_mean_eligible')}  sim_mean={res.get('sim_mean_eligible')}",
            f"         num_candidates: KS={res.get('ks_candidates')}",
        ]
    elif name == "selection_detail":
        lines += [
            f"         num_chosen : real={res.get('real_mean_chosen')}  sim={res.get('sim_mean_chosen')}  "
            f"rel_diff={res.get('rel_diff_chosen')} (<={res.get('tol_chosen')})",
            f"         in_flight  : real={res.get('real_mean_inflight')}  sim={res.get('sim_mean_inflight')}  "
            f"rel_diff={res.get('rel_diff_inflight')} (<={res.get('tol_inflight')})",
            f"         effective_c: real={res.get('real_mean_effective_c')}  "
            f"sim={res.get('sim_mean_effective_c')}  (diagnostic only)",
        ]
    elif name == "residence":
        if res.get("status") == "SKIP":
            lines.append(f"         note: {res.get('note')}")
        else:
            lines += [
                f"         in_flight_after (carried): real={res.get('real_inflight_after')} "
                f"sim={res.get('sim_inflight_after')}  rel_diff={res.get('rel_diff_carry')} "
                f"(<={res.get('tol_rel')})",
                f"         committed_fresh: real={res.get('real_committed_fresh')} "
                f"sim={res.get('sim_committed_fresh')}  | stale_rejected: "
                f"real={res.get('real_stale_rejected')} sim={res.get('sim_stale_rejected')}",
                f"         residence_rounds: real={res.get('real_residence_rounds')} "
                f"sim={res.get('sim_residence_rounds')}  ({res.get('note')})",
            ]
    elif name == "trainer_phase":
        per_phase = res.get("per_phase", {})
        for phase, info in per_phase.items():
            lines.append(
                f"         {phase:22s}: real={info.get('real_mean_s')}s  "
                f"sim={info.get('sim_mean_s')}s  KS={info.get('ks')}"
            )
    elif name == "field_coverage":
        for label, info in res.get("matrix", {}).items():
            lines.append(
                f"         {label:34s}: real={info.get('real')}  "
                f"sim={info.get('sim')}  (expect {info.get('expect')})"
            )
        viol = res.get("violations")
        if viol:
            lines.append(f"         MISSING: {viol}")
    elif name == "modeled_compute_advance":
        lines += [
            f"         sim:  advance={res.get('sim_mean_advance_s')}s  "
            f"max_speed={res.get('sim_mean_max_speed_s')}s  "
            f"implied_overhead={res.get('sim_implied_overhead_s')}s",
            f"         real: advance={res.get('real_mean_advance_s')}s  "
            f"max_speed={res.get('real_mean_max_speed_s')}s  "
            f"implied_overhead={res.get('real_implied_overhead_s')}s",
        ]
    elif name == "overhead_residual":
        lines += [
            f"         real advance={res.get('real_mean_advance_s')}s  "
            f"sim advance={res.get('sim_mean_advance_s')}s  "
            f"residual={res.get('residual_s')}s  rel={res.get('rel')} "
            f"(<={res.get('tol_rel')})",
            f"         implied per-commit overhead="
            f"{res.get('implied_per_commit_overhead_s')}s "
            f"(agg_goal={res.get('agg_goal')})",
        ]
    elif name == "eligible_speed":
        lines += [
            f"         pool_speed_KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
            f"[{res.get('speed_source')}]  "
            f"real_mean={res.get('real_mean_pool_speed_s')}s sim_mean={res.get('sim_mean_pool_speed_s')}s",
            f"         observed (diag): real={res.get('real_observed_pool_speed_s')}s "
            f"sim={res.get('sim_observed_pool_speed_s')}s",
        ]
    elif name == "selection_bias":
        if res.get("status") == "SKIP":
            lines.append(f"         note: {res.get('note')}")
        else:
            lines += [
                f"         selected_speed_KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
                f"[{res.get('speed_source')}]  "
                f"selected r/s={res.get('real_selected_mean_s')}/{res.get('sim_selected_mean_s')}s  "
                f"pool r/s={res.get('real_pool_mean_s')}/{res.get('sim_pool_mean_s')}s",
                f"         bias(selected-pool) real={res.get('real_bias_s')}s sim={res.get('sim_bias_s')}s "
                f"-> pool-match+bias-diverge=selector; pool-diverge(A2b)=composition",
                f"         observed (diag): selected r/s={res.get('real_observed_selected_s')}/"
                f"{res.get('sim_observed_selected_s')}s pool r/s={res.get('real_observed_pool_s')}/"
                f"{res.get('sim_observed_pool_s')}s",
            ]
    elif name == "selector_score":
        if res.get("status") == "SKIP":
            lines.append(f"         note: {res.get('note')}")
        else:
            wk = res.get("worst_component")
            per = res.get("per_component") or {}
            lines.append(
                f"         worst term={wk} KS={res.get('worst_ks')} (<={res.get('ks_tol')})"
            )
            for k, v in per.items():
                lines.append(
                    f"           {k:14s} KS={v.get('ks')}  real_mean={v.get('real_mean')}  sim_mean={v.get('sim_mean')}"
                )
    elif name == "preferred_duration":
        if res.get("status") == "SKIP":
            lines.append(f"         note: {res.get('note')}")
        else:
            lines += [
                f"         frac_rounds_penalty_binds real={res.get('real_frac_binding')} "
                f"sim={res.get('sim_frac_binding')} (diff={res.get('frac_diff')} <={res.get('frac_tol')})",
                f"         reconstructed pref median r/s="
                f"{res.get('real_pref_median_s')}/{res.get('sim_pref_median_s')}s",
            ]
    elif name == "avail_timebase":
        lines += [
            f"         max_rel_diff={res.get('max_rel_diff')} "
            f"(<={res.get('tol_rel')})  per-decile={res.get('per_bin_rel_diff')}",
        ]
    elif name == "duty_cycle":
        if res.get("max_dutycycle_diff") is not None:
            lines.append(
                f"         max_dutycycle_diff={res.get('max_dutycycle_diff')}  "
                f"n_trainers={res.get('n_trainers')}"
            )
    elif name == "training_budget":
        if res.get("ks_stat") is not None:
            lines.append(
                f"         KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
                f"real_mean={res.get('real_mean_s')}s sim_mean={res.get('sim_mean_s')}s"
            )
    elif name.startswith("phase_"):
        if res.get("ks_stat") is not None:
            lines.append(
                f"         KS={res.get('ks_stat')} (<={res.get('ks_tol')})  "
                f"real_mean={res.get('real_mean_s')}s sim_mean={res.get('sim_mean_s')}s"
            )
    elif name == "convergence_loss":
        if res.get("avg_loss_diff") is not None:
            lines.append(
                f"         avg_loss_diff={res.get('avg_loss_diff')} "
                f"(<={res.get('loss_tol')})  "
                f"eval_rounds={res.get('eval_rounds_compared')}"
            )

    return [l for l in lines if l.strip()]


def _status_for(key: str, res: dict, root_set: set, down_set: set,
                warn_set: set) -> tuple:
    """Return (icon, status_word) for a single check line."""
    if res.get("status") == "SKIP" or (
            res.get("note", "").startswith("K10:") and key != "vclock_telemetry"):
        return _ICONS["SKIP"], "SKIP"
    # Low-confidence pass on a short run: a genuine fail still routes to FAIL/DOWN
    # below (it stays in root_set/down_set); only an otherwise-PASS is flagged.
    if res.get("low_confidence") and key not in root_set and key not in down_set:
        return _ICONS["LOWC"], "LOWC"
    if key in root_set:
        return _ICONS["FAIL"], "FAIL"
    if key in down_set:
        return _ICONS["DOWN"], "DOWN"
    if key in warn_set:
        return _ICONS["WARN"], "WARN"
    return _ICONS["PASS"], "PASS"


def print_report(results: dict, strict: bool = False, lenient: bool = False,
                 real_label: str = "", sim_label: str = "") -> bool:
    """Print stage-grouped ladder report to stdout.  Returns True if passed."""
    passed, roots, downstream, warnings = overall_verdict(
        results, strict=strict, lenient=lenient)
    root_set, down_set, warn_set = set(roots), set(downstream), set(warnings)

    width = 78
    print(f"\n{'='*width}")
    if real_label or sim_label:
        print(f"  PARITY REPORT  real={real_label}  sim={sim_label}")
    else:
        print("  PARITY REPORT: real vs simulated")
    print(f"{'='*width}")

    # ── root-cause banner: the lowest broken rung(s) ──
    if roots:
        print("  ROOT-CAUSE (lowest broken rung(s) with passing upstreams):")
        for r in roots:
            stage = check_stage(r)
            note = results.get(r, {}).get("note", "")
            tail = f" — {note}" if note else ""
            print(f"    -> stage {stage}: {r}{tail}")
        if downstream:
            print(f"  {len(downstream)} downstream failure(s) suppressed: "
                  f"{', '.join(downstream)}")
    print(f"{'='*width}\n")

    for sec_id, sec_title, checks in _SECTIONS:
        if not any(key in results for _, key in checks):
            continue
        print(f"  -- Stage {sec_id}  {sec_title}")
        for label, key in checks:
            res = results.get(key)
            if res is None:
                continue
            res["_name"] = key
            tier = _TIER_TAG.get(res.get("tier", ""), "     ")
            role = _ROLE_TAG.get(check_role(key), "    ")
            icon, status_word = _status_for(key, res, root_set, down_set, warn_set)
            print(f"  {icon} [{tier}|{role}] [{status_word:4s}] {label}")
            for detail in _fmt_metric(key, res):
                print(detail)
        print()

    verdict_word = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    if warnings and passed:
        verdict_word += f"  ({len(warnings)} warning(s))"
    icon = _ICONS["PASS"] if passed else _ICONS["FAIL"]
    print(f"{'='*width}")
    print(f"  {icon} {verdict_word}")
    if roots:
        print(f"  Root cause(s): {', '.join(roots)}")
    if downstream:
        print(f"  Downstream:    {', '.join(downstream)}")
    if warnings:
        print(f"  Warned:        {', '.join(warnings)}")
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
    print(f"[parity] JSON results -> {path}")


def write_plot(results: dict, path: str,
               real_label: str = "real", sim_label: str = "sim") -> None:
    """Write multi-panel PNG: virtual-time trajectory + per-round advance + overlap factor."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[parity] matplotlib not available - skipping plot")
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
    print(f"[parity] Plot -> {path}")


def roll_up_table(batch_results: dict) -> None:
    """Print a roll-up table for --batch mode: one row per baseline.

    The ROOT column names the lowest broken rung so a batch sweep shows where
    each baseline first diverges at a glance.
    """
    print(f"\n{'='*100}")
    print(f"  {'Baseline':<18} {'rounds(r/s)':<14} {'s/round(r/s)':<16} "
          f"{'K2':>3} {'K3':>3} {'K10':>3} {'P3':>3} {'overall':>7}  {'ROOT':<22}")
    print(f"  {'-'*18} {'-'*14} {'-'*16} {'-'*3} {'-'*3} {'-'*3} {'-'*3} "
          f"{'-'*7}  {'-'*22}")
    for baseline, res in sorted(batch_results.items()):
        results = res.get("results", {})
        th = results.get("throughput", {})
        passed, roots, downstream, _ = overall_verdict(results)

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
        root_str = roots[0] if roots else ("—" if passed else "?")
        print(f"  {baseline:<18} {rounds_str:<14} {spr_str:<16} "
              f"{_s('throughput'):>3} {_s('per_round_advance'):>3} "
              f"{_s('vclock_telemetry'):>3} {_s('trainer_speed'):>3} "
              f"{overall:>7}  {root_str:<22}")
    print(f"{'='*100}\n")
