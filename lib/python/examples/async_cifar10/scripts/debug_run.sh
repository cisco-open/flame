#!/bin/bash
# Configurable debug runner for targeted baseline comparison.
#
# Generates a single filtered+patched YAML on the fly from the parity template
# (felix_oort_refl_feddance_alpha0.1_parity.yaml — each baseline as a sim+real
# pair), keeping only the requested baselines and overriding runtime. Node- and
# duration-agnostic: the same invocation works on any machine — pick baselines,
# mode, duration.
#
# Usage (run anywhere):
#   debug_run.sh --baselines oort [--runtime-s 3600] [--mode sim|real|both]
#   debug_run.sh --baselines 'refl feddance' --runtime-s 1800
#   debug_run.sh smoke        # 48 trainers, 4 rounds, all baselines
#
# --baselines is matched against the 'baseline:' field in the parity config, so
# any baseline runs regardless of machine (felix/oort/refl/feddance). An unknown
# baseline is a no-op (not an error).
#
# Runtime:
#   --runtime-s sets max_experiment_runtime_s for BOTH real and sim variants.
#   Real mode:  wall-clock seconds (passes directly).
#   Sim mode:   virtual-clock seconds (vclock fix ensures sim stops at this
#               many virtual seconds, which completes in far less wall-clock
#               time for sync baselines like Refl).
set -u

# --- robust conda activation ---
ENVNAME="${FLAME_CONDA_ENV:-dg_flame}"
CB=""
if command -v conda >/dev/null 2>&1; then
  CB="$(conda info --base 2>/dev/null)"
elif [ -n "${CONDA_EXE:-}" ]; then
  CB="$(dirname "$(dirname "$CONDA_EXE")")"
fi
if [ -z "$CB" ] || [ ! -f "$CB/etc/profile.d/conda.sh" ]; then
  for c in "$HOME/miniconda3" "/coc/scratch/${USER%??}/miniconda3" \
           "/coc/scratch/$USER/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CB="$c" && break
  done
fi
if [ -z "$CB" ] || [ ! -f "$CB/etc/profile.d/conda.sh" ]; then
  echo "ERROR: conda not found. Activate '$ENVNAME' yourself or set CONDA_EXE." >&2; exit 1
fi
source "$CB/etc/profile.d/conda.sh"
conda activate "$ENVNAME" || { echo "ERROR: 'conda activate $ENVNAME' failed" >&2; exit 1; }
echo "conda: base=$CB env=$ENVNAME python=$(which python)"

# repo example dir (portable across nodes)
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EX" || exit 1
SCR=expt_scripts_2026
LOGDIR="${FLAME_LOGDIR:-/tmp/debug_run_logs}"; mkdir -p "$LOGDIR"
export FLAME_BATCH_CONTINUE_ON_ERROR=1

# defaults
RUNTIME_S=10800
BASELINES="felix refl"
SIM_WALL_CEILING_S=""  # empty = max_experiment_runtime_s (1×, tight guard; sim should be faster than real)
MODE="both"            # sim | real | both — which time_mode variant(s) of each baseline to run
NUM_TRAINERS=""        # empty = use whatever's in the parity config (300); non-smoke override only
ALPHA=""               # empty = use the parity config's dirichlet_alpha (0.1); e.g. 100 for homogeneous

usage() {
  echo "usage: $0 [--baselines 'felix refl'] [--runtime-s 3600] [--mode sim|real|both] [--sim-wall-ceiling-s 2700] [--trace syn_20]"
  echo "       $0 smoke [--baselines ...] [--mode sim|real|both] [--trace syn_20]"
  echo ""
  echo "  --baselines           which baselines to run (any of felix oort oort_star refl feddance fedbuff);"
  echo "                        filtered from the parity config, node-agnostic."
  echo "  --mode                which time_mode variant(s) to run for each baseline:"
  echo "                        'sim' (only the simulated run), 'real' (only the real run),"
  echo "                        or 'both' (default, runs both sequentially). Lets you split"
  echo "                        e.g. felix-sim on one machine and felix-real on another."
  echo "  --sim-wall-ceiling-s  wall-clock ceiling for sim mode (default: = runtime_s)."
  echo "                        A well-behaved sim finishes in <= real-mode wall time."
  echo "                        Fires [SIM_WALL_CEILING] warning + stops when exceeded."
  echo "  --trace               availability trace name(s) to substitute, space-separated for"
  echo "                        multiple (e.g. 'syn_20 syn_50' queues both, one experiment set each)."
  echo "                        Default: use whatever is in the parity config (syn_0)."
  echo "  --num-trainers        non-smoke only: shrink the cohort below the parity config's 300,"
  echo "                        scaling min_trainers_to_start down with it (gap of 8, same ratio as"
  echo "                        smoke). Use this instead of 'smoke' when you need a real --runtime-s"
  echo "                        budget (e.g. a vclock floor for an availability trace) that smoke's"
  echo "                        hardcoded rounds=4/runtime=240 would cut short."
  echo "  --alpha               Dirichlet alpha override (default: parity config's 0.1). Supported"
  echo "                        values have an n300 split: 0.1 / 1.0 / 10.0 / 100.0 (100=homogeneous)."
  echo "                        When set, the split lookup uses the n300 partition for that alpha."
  exit 2
}

# parse args
TRACE=""  # empty = use whatever is in the parity config (syn_0)
if [ "${1:-}" = "smoke" ]; then
  SMOKE=1; shift
  BASELINES="felix oort oort_star refl feddance fedbuff"   # smoke default: validate all
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baselines) BASELINES="$2"; shift 2 ;;
      --mode)      MODE="$2"; shift 2 ;;
      --trace)     TRACE="$2"; shift 2 ;;
      --alpha)     ALPHA="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
else
  SMOKE=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baselines)           BASELINES="$2"; shift 2 ;;
      --runtime-s)           RUNTIME_S="$2"; shift 2 ;;
      --mode)                MODE="$2"; shift 2 ;;
      --sim-wall-ceiling-s)  SIM_WALL_CEILING_S="$2"; shift 2 ;;
      --wall-runtime-s)      SIM_WALL_CEILING_S="$2"; shift 2 ;;  # backward compat alias
      --trace)               TRACE="$2"; shift 2 ;;
      --num-trainers)        NUM_TRAINERS="$2"; shift 2 ;;
      --alpha)               ALPHA="$2"; shift 2 ;;
      # --node is DEPRECATED (node1/node2 split removed): baselines are filtered
      # from a single node-agnostic parity config, so the node is irrelevant.
      # Accept+ignore so existing wrappers don't hard-error.
      --node)                echo "WARNING: --node '$2' is deprecated and ignored (runner is now node-agnostic)." >&2; shift 2 ;;
      *) usage ;;
    esac
  done
fi
case "$MODE" in sim|real|both) ;; *) echo "ERROR: --mode must be sim|real|both (got '$MODE')" >&2; exit 2 ;; esac

# Generate a single filtered+patched YAML from the parity source config.
# $1 = baselines (space-separated), $2 = runtime_s, $3 = output path,
# [$4 = smoke: 1|0], [$5 = sim_wall_ceiling_s: int or ""], [$6 = mode: sim|real|both],
# [$7 = trace: trace name or ""], [$8 = num_trainers override: int or "", non-smoke only],
# [$9 = alpha override: float or ""]
make_debug_yaml() {
  python - "$SCR" "$1" "$2" "$3" "${4:-0}" "${5:-}" "${6:-both}" "${7:-}" "${8:-}" "${9:-}" <<'PY'
import yaml, sys, copy, os
scr, baselines_str, runtime_s, outpath, smoke, ceil_arg = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
    sys.argv[6] if len(sys.argv) > 6 else ""
)
mode = (sys.argv[7] if len(sys.argv) > 7 else "both").lower()
# Space-separated list of trace names (e.g. "syn_20 syn_50"); "" -> [""] (no substitution).
trace_overrides = sys.argv[8].strip().split() if len(sys.argv) > 8 and sys.argv[8].strip() else [""]
num_trainers_override = int(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9].strip() else None
alpha_override = float(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10].strip() else None
requested = set(baselines_str.lower().split())
# Deterministic selection seed (same for real+sim). Default 1234; SEED=none disables.
_seed_env = os.environ.get("SEED", "1234").strip()
seed_val = None if _seed_env.lower() in ("none", "") else int(_seed_env)


def exp_mode(e):
    """sim | real for an experiment, from its time_mode field (preferred) or its
    name suffix (_sim / _real)."""
    tm = str((e.get("trainer") or {}).get("time_mode") or "").lower()
    if tm.startswith("sim"):
        return "sim"
    if tm == "real":
        return "real"
    n = e.get("name", "").lower()
    if n.endswith("_real"):
        return "real"
    if n.endswith("_sim"):
        return "sim"
    return "unknown"


# Single node-agnostic parity config holding every baseline (felix, oort, refl,
# feddance) × {sim, real}; filter it to the requested baselines/mode.
src = f"{scr}/felix_oort_refl_feddance_alpha0.1_parity.yaml"
try:
    # encoding="utf-8" explicit: the config has non-ASCII chars (e.g. "->" arrows
    # in comments/descriptions); without this, open() falls back to the node's
    # locale-preferred encoding, which mis-decodes them on non-UTF-8 locales
    # (e.g. C/POSIX) and yaml.safe_load then rejects the resulting control chars.
    cfg = yaml.safe_load(open(src, encoding="utf-8"))
except FileNotFoundError:
    print(f"ERROR: parity config not found: {src}", flush=True)
    sys.exit(1)

kept = []
for e_src in cfg.get("experiments", []):
    bl = e_src.get("baseline", "").lower()
    if bl not in requested:
        continue
    if mode != "both" and exp_mode(e_src) != mode:
        continue
    # One experiment per requested trace (trace_overrides has 1 entry, "", when
    # --trace wasn't given, so this loop is a no-op pass-through by default).
    for trace_override in trace_overrides:
        e = copy.deepcopy(e_src)
        h = e["aggregator"]["config_overrides"]["hyperparameters"]
        h["max_experiment_runtime_s"] = runtime_s
        # Deterministic seed: the SAME value for every experiment so the real and sim
        # variants of each baseline make identical selection draws (dedicated per-
        # selector RNG, PARITY "Determinism / seeding"). Without this, real vs sim are
        # two independent stochastic paths and participation/utility can never match.
        # Override per-invocation with SEED=<n>; SEED=none disables (legacy unseeded).
        if seed_val is not None:
            h["seed"] = seed_val
        # sim_wall_ceiling_s: tight wall guard -- sim must finish in <= this many
        # wall-seconds (default = max_experiment_runtime_s = 1x; a healthy sim is faster).
        h["sim_wall_ceiling_s"] = int(ceil_arg) if ceil_arg else runtime_s
        if smoke:
            e["trainer"]["num_trainers"] = 48
            h["rounds"] = 4
            h["min_trainers_to_start"] = 40
            h["min_trainers_join_timeout_s"] = 120
            e["name"] = "dbg_smoke_" + e["name"]
        else:
            # High round cap so the wall/vclock budget (max_experiment_runtime_s) is the
            # binding stop condition, not an early round-count termination.
            h["rounds"] = 20000
            if num_trainers_override:
                # Shrink the cohort but keep runtime_s as the real budget (unlike
                # smoke, which hardcodes rounds=4/runtime=240 -- too short for a
                # trace-driven vclock floor like syn_20's first UN_AVL at t=600s).
                # Same join-barrier slack ratio as smoke (gap of 8 below the count).
                # Preserve the config's native partition size as split_num_trainers
                # so the shrunk cohort reads the existing n<orig> split (e.g. n300)
                # instead of demanding a dedicated n<override> split file that may
                # not exist (there is no cifar10_alpha0.1_n10 split, only n48/50/300).
                # spawn_all spawns num_trainers trainers but keys the split lookup on
                # split_num_trainers -- the two are independent by design.
                orig_n = e["trainer"].get("num_trainers", 300)
                e["trainer"]["num_trainers"] = num_trainers_override
                e["trainer"]["split_num_trainers"] = orig_n
                h["min_trainers_to_start"] = max(1, num_trainers_override - 8)
            e["name"] = f"dbg_{e['name']}"
        # --alpha override: repoint dirichlet_alpha and the split lookup. Only n300
        # splits exist for every alpha (0.1/1.0/10.0/100.0=homogeneous); n48/n50
        # exist for alpha0.1 only. So read the n300 partition for the chosen alpha
        # (the cohort stays num_trainers, spawned as the first num_trainers of the
        # 300-way split via the split_num_trainers decoupling). Name gets an
        # alpha<..> tag so run dirs are distinguishable across alphas.
        if alpha_override is not None:
            e["trainer"].setdefault("dataset", {})["dirichlet_alpha"] = alpha_override
            e["trainer"]["split_num_trainers"] = 300
            e["name"] = f"{e['name']}_alpha{str(alpha_override).replace('.', 'p')}"
        # --trace override: substitute availability trace in trainer + aggregator config.
        if trace_override:
            # trainer.availability.mode is NOT read by anything (main.py/config.py never
            # touch config.availability) -- vestigial from an earlier design, kept
            # write-only here so as not to silently drop a field some other consumer may
            # still expect. The trainer's ACTUAL trace selection comes from
            # hyperparameters.client_notify.trace (see main.py's state_avl_event_ts
            # assignment), which lives under trainer.config_overrides.hyperparameters,
            # not trainer.hyperparameters (that block is base-model HP only: batchSize/
            # learningRate/etc, merged from configs/trainer_base.yaml's own client_notify
            # default of trace=syn_0). Before this fix, ONLY the aggregator's own trace
            # read (via `h` below) was ever overridden -- every debug_run.sh-launched
            # trainer, real and sim, ran with client_notify.trace stuck at the
            # trainer_base.yaml default (syn_0, always-available) regardless of the
            # requested --trace, silently no-op'ing the trainer-side avl_state machinery
            # (and hence the real-mode send-gate and all avail_change telemetry) for
            # every trace-driven run this project has ever launched. Root-caused Jul 1
            # via UNAVAILABILITY_DESIGN.md Batch 3 T3.1.
            avail = e["trainer"].setdefault("availability", {})
            old_trace = avail.get("mode", "syn_0")
            avail["mode"] = trace_override
            t_co_hp = e["trainer"].setdefault("config_overrides", {}).setdefault("hyperparameters", {})
            t_co_hp.setdefault("client_notify", {})["trace"] = trace_override
            if "trackTrainerAvail" in h:
                h["trackTrainerAvail"]["trace"] = trace_override
                # For baselines NOT on the ORACULAR legacy path (felix, feddance,
                # oracle, fedbuff): activate the new sim_unavailability gate so
                # _init_availability picks up the trace (Sec 7 felix master-gate).
                # ORACULAR baselines (oort, refl) already activate via the legacy path.
                if h["trackTrainerAvail"].get("type", "").upper() != "ORACULAR":
                    h["simUnavailability"] = True
                    # proactive_inflight_evict is set directly in each experiment's
                    # config_overrides HP (T1 two-axis split); no auto-detection needed
                    # here. The client_notify.enabled check below is always False
                    # (Stage H is future), so proactiveInflightEvict is never set by
                    # this branch -- the explicit YAML value is authoritative.
                    t_hp = e.get("trainer", {}).get("hyperparameters", {})
                    if str(t_hp.get("client_notify", {}).get("enabled", "False")).lower() == "true":
                        h["proactiveInflightEvict"] = True
            elif "client_notify" in h and isinstance(h["client_notify"], dict):
                h["client_notify"]["trace"] = trace_override
                h["simUnavailability"] = True
            elif e["aggregator"].get("tracking_mode", "oracular").lower() != "oracular":
                # Non-oracular baseline with no HP-level tracking block (e.g. feddance
                # in v1, which has no client_notify in HP and no trackTrainerAvail).
                # Inject trace via availability_trace so _init_availability finds it.
                h["availability_trace"] = trace_override
                h["simUnavailability"] = True
            # Rewrite syn_<digits> or syn<digits> in the name so run dirs are identifiable.
            import re
            e["name"] = re.sub(r"syn_?[0-9]+", trace_override, e["name"])
        e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
        kept.append(e)

if not kept:
    print(f"WARNING: no experiments matched baselines={baselines_str} mode={mode}",
          flush=True)
    sys.exit(0)

cfg["experiments"] = kept
yaml.safe_dump(cfg, open(outpath, "w", encoding="utf-8"), sort_keys=False)
print(f"Generated {outpath} with {len(kept)} experiment(s): "
      f"{[e['name'] for e in kept]}", flush=True)
PY
}

# Count experiments in a generated YAML (used to estimate budget and track progress).
_count_exps() {
  python3 - "$1" <<'PY'
import yaml, sys
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(len(d.get('experiments', [])))
PY
}

run_node() {
  local label="$1" cfg="$2" budget_s="${3:-0}" n_exps="${4:-1}"
  local start_ts; start_ts=$(date +%s)
  # Baseline run-dir count — new dirs that appear are newly-started experiments.
  local initial_runs; initial_runs=$(find experiments -maxdepth 1 -name "run_*" -type d 2>/dev/null | wc -l)

  echo "[$(date '+%F %T')] START $label ($n_exps exp(s), ~${budget_s}s budget)" | tee -a "$LOGDIR/debug_run.log"

  # Background progress ticker: fires every 30s, prints elapsed/remaining/percent
  # and how many experiments have started (each start creates a new run_* dir).
  (
    while true; do
      sleep 30
      local now; now=$(date +%s)
      local elapsed=$(( now - start_ts ))
      local pct=0 remaining=0
      if [ "$budget_s" -gt 0 ]; then
        pct=$(( elapsed * 100 / budget_s ))
        remaining=$(( budget_s - elapsed ))
        [ "$pct" -gt 100 ] && pct=100
        [ "$remaining" -lt 0 ] && remaining=0
      fi
      local curr; curr=$(find experiments -maxdepth 1 -name "run_*" -type d 2>/dev/null | wc -l)
      local started=$(( curr - initial_runs ))
      [ "$started" -lt 0 ] && started=0
      printf "  [%s] %s | %ds elapsed / ~%ds (%d%%) | exp started: %d/%d\n" \
        "$(date '+%T')" "$label" "$elapsed" "$budget_s" "$pct" "$started" "$n_exps"
    done
  ) &
  local ticker_pid=$!

  python -m flame.launch.run_experiment "$cfg" --example-dir "$EX" \
      < /dev/null >> "$LOGDIR/${label}.out" 2>&1
  local rc=$?

  kill "$ticker_pid" 2>/dev/null
  wait "$ticker_pid" 2>/dev/null

  local elapsed=$(( $(date +%s) - start_ts ))
  echo "[$(date '+%F %T')] DONE  $label exit=$rc (took ${elapsed}s / ~${budget_s}s budget)" | tee -a "$LOGDIR/debug_run.log"
}

# ---- smoke mode ----
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE DEBUG: 48 trainers, 4 rounds, baselines=${BASELINES} ==="
  cfg="$LOGDIR/dbg_smoke.yaml"
  # Clear any stale config from a previous invocation so a no-match run is
  # skipped (not silently re-running a leftover config).
  rm -f "$cfg"
  make_debug_yaml "$BASELINES" 240 "$cfg" 1 "$SIM_WALL_CEILING_S" "$MODE" "$TRACE" "" "$ALPHA"
  if [ -f "$cfg" ]; then
    _n=$(_count_exps "$cfg")
    run_node "dbg_smoke" "$cfg" $(( _n * 240 )) "$_n"
  fi
  echo "=== SMOKE RESULTS ==="
  for dd in experiments/run_*dbg_smoke_*; do
    [ -d "$dd" ] || continue
    t=$(ls "$dd"/telemetry/aggregator_*.jsonl 2>/dev/null | head -1)
    rounds=$(grep -c "agg_round" "$t" 2>/dev/null || echo 0)
    printf "  %-60s agg_round_events=%s\n" "$(basename "$dd")" "$rounds"
  done
  exit 0
fi

# ---- normal run mode ----
echo "=== DEBUG RUN: baselines='$BASELINES' mode=$MODE runtime_s=$RUNTIME_S sim_wall_ceiling_s=${SIM_WALL_CEILING_S:-auto(=runtime_s)} num_trainers=${NUM_TRAINERS:-300(default)} alpha=${ALPHA:-0.1(default)} ==="
cfg="$LOGDIR/debug_run.yaml"
# Clear any stale config so a no-match run is skipped (not silently re-running
# a previous baseline's leftover config).
rm -f "$cfg"
make_debug_yaml "$BASELINES" "$RUNTIME_S" "$cfg" 0 "$SIM_WALL_CEILING_S" "$MODE" "$TRACE" "$NUM_TRAINERS" "$ALPHA"

if [ ! -f "$cfg" ]; then
  echo "No experiments matched for baselines='$BASELINES'. Nothing to run."
  exit 0
fi

_n_exps=$(_count_exps "$cfg")
_budget=$(( _n_exps * RUNTIME_S ))
echo "  queued: $_n_exps exp(s), estimated budget ~${_budget}s (sim finishes faster than real)"
run_node "debug_run" "$cfg" "$_budget" "$_n_exps"
echo "Logs: $LOGDIR/debug_run.out"
echo "Run dirs: experiments/run_*dbg_*"
