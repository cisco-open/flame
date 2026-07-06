#!/usr/bin/env bash
# ============================================================================
# smoke_suite.sh  —  Sequential smoke campaign with per-run timeout guard.
#
# Runs five ordered steps.  Each individual (baseline, mode) invocation of
# debug_run.sh is wrapped in a wall-clock timeout: if the run does not
# self-terminate within (runtime_s + timeout_buffer_s), it is killed via
# SIGTERM → SIGKILL on the whole process group so no orphan trainers remain.
# The suite then continues with the next run.
#
# Steps
#   1  pytest          — unit tests (flame/tests/), expected: 536p/7s
#   2  syn_0  sim      — byte-identity regression, all 6 baselines
#   3  syn_20 sim      — all 6 baselines (availability active, fast mode)
#   4  syn_20 both     — all 6 baselines × sim + real (parity gate)
#   5  syn_50 starvation — feddance + oort, both modes (B2.0.2 regression)
#      Expected: [SIM_STARVATION] events present, no [SIM_WALL_CEILING],
#                self-stops via "stopping run" in both modes.
#
# Per-run checks (applied to the Python agg output log):
#   stopping_run  count of "stopping run" lines — must be > 0
#   wall_ceiling  count of [SIM_WALL_CEILING] lines — must be 0
#   starvation    count of [SIM_STARVATION] lines — informational
#
# Report is written to <output-dir>/report.txt and printed at the end.
#
# Usage:
#   smoke_suite.sh [OPTIONS]
#
# Options:
#   --runtime-syn0-s  N    Budget (wall/vclock) for syn_0 runs     [default: 900]
#   --runtime-syn20-s N    Budget for syn_20 runs                  [default: 1800]
#   --runtime-syn50-s N    Budget for syn_50 starvation runs       [default: 3600]
#   --timeout-buffer-s N   Extra wall-sec before force-kill        [default: 600]
#   --kill-settle-s N      GPU memory settle wait after force-kill [default: 20]
#   --steps LIST           Comma-separated steps to run (1–5)      [default: 1,2,3,4,5]
#   --baselines NAMES      Space-separated baseline list (steps 2–4) [default: all 6]
#   --starvation-baselines NAMES  Baselines for step 5            [default: feddance oort]
#   --num-trainers N       Shrink the cohort below the parity config's 300
#                          (forwarded to debug_run.sh --num-trainers on every
#                          run) [default: "" = use the parity config's 300]
#   --output-dir DIR       Log + report directory                  [default: /tmp/smoke_suite_<ts>]
#   --background           Re-exec via nohup+disown and return immediately;
#                          survives the launching shell/SSH session closing.
#                          Prints the PID, nohup log, and report path, then
#                          exits 0 right away — the suite keeps running
#                          detached. Use this for unattended/overnight runs.
#   --dry-run              Print commands without running them
#   --help
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # async_cifar10/
LIB_DIR="$(cd "$EX_DIR/../.." && pwd)"           # lib/python/
DEBUG_RUN="$SCRIPT_DIR/debug_run.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
RUNTIME_SYN0_S=900
RUNTIME_SYN20_S=1800
RUNTIME_SYN50_S=3600
TIMEOUT_BUFFER_S=600
KILL_SETTLE_S=20       # GPU settle wait (seconds) after a force-killed run
ALL_BASELINES="felix oort oort_star refl feddance fedbuff"
STARV_BASELINES="feddance oort"
STEPS="1,2,3,4,5"
OUTPUT_DIR="/tmp/smoke_suite_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
BACKGROUND=0
NUM_TRAINERS=""

# ── Progress counters (set after arg-parse via _compute_total_runs) ───────────
TOTAL_RUNS=0
COMPLETED_RUNS=0

# ── Arg parsing ──────────────────────────────────────────────────────────────
usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
  exit 0
}

ORIG_ARGS=("$@")   # preserved pre-shift for the --background re-exec below

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-syn0-s)       RUNTIME_SYN0_S="$2";    shift 2 ;;
    --runtime-syn20-s)      RUNTIME_SYN20_S="$2";   shift 2 ;;
    --runtime-syn50-s)      RUNTIME_SYN50_S="$2";   shift 2 ;;
    --timeout-buffer-s)     TIMEOUT_BUFFER_S="$2";  shift 2 ;;
    --kill-settle-s)        KILL_SETTLE_S="$2";     shift 2 ;;
    --steps)                STEPS="$2";             shift 2 ;;
    --baselines)            ALL_BASELINES="$2";     shift 2 ;;
    --starvation-baselines) STARV_BASELINES="$2";   shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2";        shift 2 ;;
    --num-trainers)         NUM_TRAINERS="$2";      shift 2 ;;
    --dry-run)              DRY_RUN=1;              shift   ;;
    --background)           BACKGROUND=1;           shift   ;;
    --help|-h)              usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

# ── --background: re-exec detached, return control immediately ───────────────
# Guarded by SMOKE_SUITE_BG so the re-exec'd child (which still sees
# --background in ORIG_ARGS) runs the real suite instead of looping.
if [[ "$BACKGROUND" == "1" && -z "${SMOKE_SUITE_BG:-}" ]]; then
  mkdir -p "$OUTPUT_DIR"
  NOHUP_LOG="$OUTPUT_DIR/nohup.log"
  SMOKE_SUITE_BG=1 nohup bash "$0" "${ORIG_ARGS[@]}" >"$NOHUP_LOG" 2>&1 < /dev/null &
  disown
  echo "[smoke_suite] backgrounded — PID $!  (survives this shell/SSH session closing)"
  echo "[smoke_suite] nohup log : $NOHUP_LOG"
  echo "[smoke_suite] report    : $OUTPUT_DIR/report.txt   (written when the suite finishes)"
  echo "[smoke_suite] progress  : tail -f $NOHUP_LOG"
  exit 0
fi

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/runs"
SUITE_LOG="$OUTPUT_DIR/suite.log"
REPORT="$OUTPUT_DIR/report.txt"
SUITE_START=$(date +%s)

_log()  { echo "[$(date '+%F %T')] $*" | tee -a "$SUITE_LOG"; }
_step() { _log ""; _log "══════ STEP $* ══════"; }

# ── Result tracking ──────────────────────────────────────────────────────────
# Each entry: "label|status|stopping_run|wall_ceiling|starvation"
declare -a RUN_RESULTS=()

_record() {
  RUN_RESULTS+=("${1}|${2}|${3}|${4}|${5}")
  COMPLETED_RUNS=$(( COMPLETED_RUNS + 1 ))
}

# ── Total-run count (computed once after arg-parse) ──────────────────────────
_compute_total_runs() {
  local total=0
  local n_all; n_all=$(echo "$ALL_BASELINES" | wc -w)
  local n_starv; n_starv=$(echo "$STARV_BASELINES" | wc -w)
  local _s
  IFS=',' read -ra _sa <<< "$STEPS"
  for _s in "${_sa[@]}"; do
    _s="${_s// /}"
    case "$_s" in
      1) total=$(( total + 1 )) ;;
      2) total=$(( total + n_all )) ;;
      3) total=$(( total + n_all )) ;;
      4) total=$(( total + 2 * n_all )) ;;
      5) total=$(( total + 2 * n_starv )) ;;
    esac
  done
  TOTAL_RUNS=$total
}

# ── Per-run timeout wrapper ───────────────────────────────────────────────────
# _run_baseline <label> <runtime_s> [debug_run_args...]
#
# Process tree (one per invocation):
#   smoke_suite.sh
#   └─ bash debug_run.sh          ← PGID = runner_pid (via set -m)
#      └─ python run_experiment    ← inherits PGID (plain Popen, no setsid)
#         ├─ python aggregator/pytorch/main_*.py   ← inherits PGID
#         └─ python trainer/pytorch/main.py × 300  ← inherits PGID
#
# Termination (timeout path):
#   1. kill -TERM -$runner_pid  → SIGTERM to whole group simultaneously.
#      ExperimentRunner._signal_handler fires on the runner → _cleanup()
#      (terminate_all trainers + terminate aggregator). Group members also
#      receive SIGTERM directly, so cleanup is redundant but harmless.
#      20s grace: gives Python time to flush files and release MQTT connections.
#   2. kill -KILL -$runner_pid  → SIGKILL to any survivors (hung GPU op,
#      stuck MQTT recv). OS reclaims GPU memory immediately after.
#   3. Post-kill sweep (this script): pkill the trainer/aggregator patterns
#      in case any process escaped the group (edge case). Then sleep
#      KILL_SETTLE_S to let GPU memory drain before the next run allocates.
#      Mirrors run_experiment_batch's _sweep_stragglers(), which is killed
#      mid-execution during a hard timeout.
#
# Grep checks scan experiments/run_*/*_aggregator.log (NOT debug_run.out).
# AggregatorSpawner redirects the aggregator subprocess's stdout/stderr to
# that dedicated file; debug_run.out is only the runner's own print()s.
#
# Returns 0 (PASS), 1 (FAIL/ERROR), 124 (TIMEOUT).
_run_baseline() {
  local label="$1" runtime_s="$2"; shift 2
  local run_dir="$OUTPUT_DIR/runs/$label"
  mkdir -p "$run_dir"
  local wall_timeout=$(( runtime_s + TIMEOUT_BUFFER_S ))

  _log "  [$label] start (runtime=${runtime_s}s  wall_timeout=${wall_timeout}s)"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: FLAME_LOGDIR=$run_dir bash $DEBUG_RUN $*" | tee -a "$SUITE_LOG"
    _record "$label" SKIP 0 0 0
    return 0
  fi

  local ts_start; ts_start=$(date +%s)
  # Marker for finding experiment dirs (and their aggregator logs) created
  # during this run. Touch before launch so any dir created after belongs here.
  local ts_marker="$run_dir/.ts_start"
  touch "$ts_marker"

  # set -m (job control) forces bash to assign PGID = runner_pid to the
  # background job regardless of whether the suite is running interactively
  # or not.  All descendants inherit that PGID (flame's spawner.py uses plain
  # subprocess.Popen with no start_new_session/os.setsid), so
  # kill -TERM/-KILL on -$runner_pid reliably reaches the whole process tree.
  # Rationale: setsid forks when the calling process is already a pg-leader
  # (interactive terminals do this), making runner_pid point to a dead parent
  # instead of the actual session leader → kill misses the tree entirely.
  set -m
  env FLAME_LOGDIR="$run_dir" bash "$DEBUG_RUN" "$@" \
    >"$run_dir/shell.log" 2>&1 &
  local runner_pid=$!
  set +m

  local deadline=$(( ts_start + wall_timeout ))
  local timed_out=0
  local _ela _kill_in _prun _pdone   # ticker temporaries
  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 5
    _ela=$(( $(date +%s) - ts_start ))
    _kill_in=$(( deadline - $(date +%s) )); [[ "$_kill_in" -lt 0 ]] && _kill_in=0
    _prun=0; [[ "$runtime_s" -gt 0 ]] && _prun=$(( _ela * 100 / runtime_s ))
    [[ "$_prun" -gt 100 ]] && _prun=100
    _pdone=0; [[ "$TOTAL_RUNS" -gt 0 ]] && _pdone=$(( COMPLETED_RUNS * 100 / TOTAL_RUNS ))
    printf '\r  %-52s  %4ds/%-4ds(%3d%%)  kill in %4ds  |  %d/%d done(%d%%)   ' \
      "[$label]" "$_ela" "$runtime_s" "$_prun" "$_kill_in" \
      "$COMPLETED_RUNS" "$TOTAL_RUNS" "$_pdone" >&2
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      printf '\n' >&2
      _log "  [$label] TIMEOUT after ${wall_timeout}s — killing process group $runner_pid"
      # SIGTERM first: lets ExperimentRunner._signal_handler call _cleanup()
      # (terminate_all trainers + terminate aggregator). 20s grace lets Python
      # flush open files and release MQTT connections before the hard kill.
      kill -TERM -"$runner_pid" 2>/dev/null || true
      sleep 20
      # SIGKILL for anything that survived (hung GPU op, stuck MQTT recv).
      kill -KILL -"$runner_pid" 2>/dev/null || true
      wait "$runner_pid" 2>/dev/null
      timed_out=1
      break
    fi
  done
  printf '\n' >&2
  [[ "$timed_out" == "0" ]] && wait "$runner_pid" 2>/dev/null
  local run_rc=$?

  # ── Post-kill cleanup ─────────────────────────────────────────────────────
  # When SIGKILL fires, run_experiment_batch's finally block (_sweep_stragglers)
  # is in the killed group and may not complete. Replicate it here: hard-kill
  # any surviving trainer/aggregator processes and wait for GPU memory to drain
  # so the next run starts from a clean slate.
  if [[ "$timed_out" == "1" ]]; then
    _log "  [$label] post-kill sweep: clearing straggler trainer/aggregator processes"
    pkill -9 -f "trainer/pytorch/main.py"  2>/dev/null || true
    pkill -9 -f "aggregator/pytorch/main_" 2>/dev/null || true
    _log "  [$label] waiting ${KILL_SETTLE_S}s for GPU memory to drain before next run"
    sleep "$KILL_SETTLE_S"
  fi

  local elapsed=$(( $(date +%s) - ts_start ))

  # ── Grep checks: scan the aggregator's dedicated log file ─────────────────
  # AggregatorSpawner redirects the aggregator subprocess's stdout/stderr to
  #   experiments/run_<ts>_<name>/<prefix>_aggregator.log
  # NOT to $FLAME_LOGDIR/debug_run.out (that file is only run_experiment.py's
  # own print()s — high-level orchestration, not FL logic messages).
  # Scan all aggregator logs created after ts_marker (handles batches of > 1).
  local stopping=0 ceiling=0 starv=0
  local found_agg_logs=()
  while IFS= read -r agg_log; do
    [[ -f "$agg_log" ]] || continue
    found_agg_logs+=("$agg_log")
    stopping=$(( stopping + $(grep -ic "stopping run"       "$agg_log" 2>/dev/null || true) ))
    ceiling=$((  ceiling  + $(grep -c  "SIM_WALL_CEILING"   "$agg_log" 2>/dev/null || true) ))
    starv=$((    starv    + $(grep -c  "\[SIM_STARVATION\]" "$agg_log" 2>/dev/null || true) ))
  done < <(find "$EX_DIR/experiments" -name "*_aggregator.log" -newer "$ts_marker" 2>/dev/null)

  # Record aggregator log paths in the run dir for easy post-mortem access.
  if [[ ${#found_agg_logs[@]} -gt 0 ]]; then
    printf '%s\n' "${found_agg_logs[@]}" > "$run_dir/agg_logs.txt"
  fi

  # ── Classify ─────────────────────────────────────────────────────────────
  local status
  if   [[ "$timed_out" == "1" ]];    then status="TIMEOUT"
  elif [[ "$run_rc"    != "0"  ]];   then status="ERROR(rc=$run_rc)"
  elif [[ "$stopping"  -eq 0   ]];   then status="FAIL(no_stop)"
  elif [[ "$ceiling"   -gt 0   ]];   then status="FAIL(wall_ceil)"
  else                                    status="PASS"
  fi

  _log "  [$label] $status  elapsed=${elapsed}s  stopping=${stopping}  wall_ceil=${ceiling}  starvation=${starv}"
  _record "$label" "$status" "$stopping" "$ceiling" "$starv"

  [[ "$status" == "PASS" ]]
}

# Convenience: run one baseline × one mode.
_run_one() {
  local baseline="$1" mode="$2" trace="$3" runtime_s="$4" step_pfx="$5"
  local label="${step_pfx}_${baseline}_${trace}_${mode}"
  local -a _nt_args=()
  [[ -n "$NUM_TRAINERS" ]] && _nt_args=(--num-trainers "$NUM_TRAINERS")
  _run_baseline "$label" "$runtime_s" \
    --baselines "$baseline" --mode "$mode" --trace "$trace" --runtime-s "$runtime_s" \
    "${_nt_args[@]}" \
    || true   # never abort the suite on a single run failure
}

# ── Step 1: Unit tests ───────────────────────────────────────────────────────
step1_pytest() {
  _step "1  Unit tests"
  local conda_env="${FLAME_CONDA_ENV:-dg_flame}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: conda run -n $conda_env python -m pytest $LIB_DIR/tests/ -q --tb=short" \
      | tee -a "$SUITE_LOG"
    _record "s1_pytest" SKIP 0 0 0
    return
  fi

  local pytest_log="$OUTPUT_DIR/runs/s1_pytest.log"
  mkdir -p "$OUTPUT_DIR/runs"
  conda run -n "$conda_env" \
    python -m pytest "$LIB_DIR/tests/" -q --tb=short \
    >"$pytest_log" 2>&1
  local rc=$?
  local summary; summary=$(tail -3 "$pytest_log")
  if [[ "$rc" == "0" ]]; then
    _log "  PASS  $summary"
    _record "s1_pytest" PASS 0 0 0
  else
    _log "  FAIL  $summary"
    _log "  Full log: $pytest_log"
    _record "s1_pytest" "FAIL(rc=$rc)" 0 0 0
  fi
}

# ── Step 2: syn_0 regression (sim only) ──────────────────────────────────────
step2_syn0_sim() {
  _step "2  syn_0 regression — sim, all 6 baselines"
  _log "  (gate-ON with always-available trace; sim completes in << runtime_s wall-seconds)"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim syn_0 "$RUNTIME_SYN0_S" s2
  done
}

# ── Step 3: syn_20 sim ───────────────────────────────────────────────────────
step3_syn20_sim() {
  _step "3  syn_20 — sim, all 6 baselines"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim syn_20 "$RUNTIME_SYN20_S" s3
  done
}

# ── Step 4: syn_20 both modes ────────────────────────────────────────────────
step4_syn20_both() {
  _step "4  syn_20 — sim then real, all 6 baselines"
  _log "  (real runs take ~${RUNTIME_SYN20_S}s wall each; total step ~$(( $(wc -w <<< "$ALL_BASELINES") * RUNTIME_SYN20_S / 60 ))+ min)"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim  syn_20 "$RUNTIME_SYN20_S" s4
    _run_one "$bl" real syn_20 "$RUNTIME_SYN20_S" s4
  done
}

# ── Step 5: syn_50 starvation regression ─────────────────────────────────────
step5_syn50_starvation() {
  _step "5  syn_50 starvation — $STARV_BASELINES, both modes (B2.0.2 regression)"
  _log "  Expected: [SIM_STARVATION] present, no [SIM_WALL_CEILING], self-stops cleanly."
  for bl in $STARV_BASELINES; do
    _run_one "$bl" sim  syn_50 "$RUNTIME_SYN50_S" s5
    _run_one "$bl" real syn_50 "$RUNTIME_SYN50_S" s5
  done
}

# ── Final report ─────────────────────────────────────────────────────────────
_final_report() {
  local total=${#RUN_RESULTS[@]}
  local pass=0 fail=0 timeout=0 skip=0 error=0

  {
    local divider; divider=$(printf '═%.0s' {1..68})
    echo "$divider"
    echo "  SMOKE SUITE REPORT"
    printf "  Generated  : %s\n"  "$(date '+%F %T')"
    printf "  Duration   : %ds\n" "$(( $(date +%s) - SUITE_START ))"
    printf "  Output     : %s\n"  "$OUTPUT_DIR"
    printf "  Runtimes   : syn_0=%ss  syn_20=%ss  syn_50=%ss  buffer=%ss  kill_settle=%ss\n" \
      "$RUNTIME_SYN0_S" "$RUNTIME_SYN20_S" "$RUNTIME_SYN50_S" "$TIMEOUT_BUFFER_S" "$KILL_SETTLE_S"
    echo "$divider"
    echo ""
    printf "%-48s  %-20s  %8s  %9s  %11s\n" \
      "RUN" "STATUS" "stop_run" "wall_ceil" "starvation"
    printf "%-48s  %-20s  %8s  %9s  %11s\n" \
      "$(printf '%0.s-' {1..48})" "$(printf '%0.s-' {1..20})" "--------" "---------" "-----------"

    for entry in "${RUN_RESULTS[@]}"; do
      IFS='|' read -r label status stopping ceiling starv <<< "$entry"
      case "$status" in
        PASS)    (( pass++    )) ;;
        SKIP)    (( skip++    )) ;;
        TIMEOUT) (( timeout++ )) ;;
        ERROR*)  (( error++   )) ;;
        FAIL*)   (( fail++    )) ;;
      esac
      printf "%-48s  %-20s  %8s  %9s  %11s\n" \
        "$label" "$status" "$stopping" "$ceiling" "$starv"
    done

    echo ""
    echo "TOTAL $total runs:  PASS=$pass  FAIL=$fail  ERROR=$error  TIMEOUT=$timeout  SKIP=$skip"
    echo ""

    # Attention list
    local bad=()
    for entry in "${RUN_RESULTS[@]}"; do
      IFS='|' read -r label status _ _ _ <<< "$entry"
      case "$status" in PASS|SKIP) ;; *) bad+=("  $label  →  $status") ;; esac
    done
    if [[ ${#bad[@]} -gt 0 ]]; then
      echo "Runs needing investigation:"
      printf '%s\n' "${bad[@]}"
      echo ""
      echo "Logs:  <label>/shell.log       — debug_run.sh + run_experiment.py stdout"
      echo "       <label>/debug_run.out   — run_experiment.py stdout (FL orchestration)"
      echo "       <label>/agg_logs.txt    — paths to aggregator log(s) for this run"
      echo "       (aggregator logs live in experiments/run_*/..._aggregator.log)"
      echo "       (grep 'stopping run'/'SIM_WALL_CEILING' in the aggregator log)"
      echo "  All under: $OUTPUT_DIR/runs/"
    else
      echo "All runs clean."
    fi
    echo "$divider"
  } | tee "$REPORT" | tee -a "$SUITE_LOG"

  # Return non-zero if anything went wrong
  (( fail + error + timeout == 0 ))
}

# ── Main ─────────────────────────────────────────────────────────────────────
_compute_total_runs
_log "Smoke suite started  (total runs: $TOTAL_RUNS)"
_log "  output      : $OUTPUT_DIR"
_log "  steps       : $STEPS"
_log "  baselines   : $ALL_BASELINES"
_log "  starvation  : $STARV_BASELINES"
_log "  num_trainers: ${NUM_TRAINERS:-300(parity config default)}"
_log "  runtimes    : syn_0=${RUNTIME_SYN0_S}s  syn_20=${RUNTIME_SYN20_S}s  syn_50=${RUNTIME_SYN50_S}s  buffer=${TIMEOUT_BUFFER_S}s  kill_settle=${KILL_SETTLE_S}s"

IFS=',' read -ra _steps_arr <<< "$STEPS"
for s in "${_steps_arr[@]}"; do
  s="${s// /}"   # strip any accidental spaces
  case "$s" in
    1) step1_pytest ;;
    2) step2_syn0_sim ;;
    3) step3_syn20_sim ;;
    4) step4_syn20_both ;;
    5) step5_syn50_starvation ;;
    *) _log "Unknown step '$s' (valid: 1–5)" ;;
  esac
done

_final_report
