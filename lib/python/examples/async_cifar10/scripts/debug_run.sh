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
#   --runtime-s sets max_runtime_s for BOTH real and sim variants.
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
LOGDIR=/tmp/debug_run_logs; mkdir -p "$LOGDIR"
export FLAME_BATCH_CONTINUE_ON_ERROR=1

# defaults
RUNTIME_S=10800
BASELINES="felix refl"
SIM_WALL_CEILING_S=""  # empty = max_runtime_s (1×, tight guard; sim should be faster than real)
MODE="both"            # sim | real | both — which time_mode variant(s) of each baseline to run

usage() {
  echo "usage: $0 [--baselines 'felix refl'] [--runtime-s 3600] [--mode sim|real|both] [--sim-wall-ceiling-s 2700]"
  echo "       $0 smoke [--baselines ...] [--mode sim|real|both]"
  echo ""
  echo "  --baselines           which baselines to run (any of felix oort refl feddance);"
  echo "                        filtered from the parity config, node-agnostic."
  echo "  --mode                which time_mode variant(s) to run for each baseline:"
  echo "                        'sim' (only the simulated run), 'real' (only the real run),"
  echo "                        or 'both' (default, runs both sequentially). Lets you split"
  echo "                        e.g. felix-sim on one machine and felix-real on another."
  echo "  --sim-wall-ceiling-s  wall-clock ceiling for sim mode (default: = runtime_s)."
  echo "                        A well-behaved sim finishes in <= real-mode wall time."
  echo "                        Fires [SIM_WALL_CEILING] warning + stops when exceeded."
  exit 2
}

# parse args
if [ "${1:-}" = "smoke" ]; then
  SMOKE=1; shift
  BASELINES="felix oort refl feddance"   # smoke default: validate all
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --baselines) BASELINES="$2"; shift 2 ;;
      --mode)      MODE="$2"; shift 2 ;;
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
# [$4 = smoke: 1|0], [$5 = sim_wall_ceiling_s: int or ""], [$6 = mode: sim|real|both]
make_debug_yaml() {
  python - "$SCR" "$1" "$2" "$3" "${4:-0}" "${5:-}" "${6:-both}" <<'PY'
import yaml, sys, copy, os
scr, baselines_str, runtime_s, outpath, smoke, ceil_arg = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
    sys.argv[6] if len(sys.argv) > 6 else ""
)
mode = (sys.argv[7] if len(sys.argv) > 7 else "both").lower()
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
    cfg = yaml.safe_load(open(src))
except FileNotFoundError:
    print(f"ERROR: parity config not found: {src}", flush=True)
    sys.exit(1)

kept = []
for e in cfg.get("experiments", []):
    bl = e.get("baseline", "").lower()
    if bl not in requested:
        continue
    if mode != "both" and exp_mode(e) != mode:
        continue
    e = copy.deepcopy(e)
    h = e["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = runtime_s
    # Deterministic seed: the SAME value for every experiment so the real and sim
    # variants of each baseline make identical selection draws (dedicated per-
    # selector RNG, PARITY "Determinism / seeding"). Without this, real vs sim are
    # two independent stochastic paths and participation/utility can never match.
    # Override per-invocation with SEED=<n>; SEED=none disables (legacy unseeded).
    if seed_val is not None:
        h["seed"] = seed_val
    # sim_wall_ceiling_s: tight wall guard — sim must finish in <= this many
    # wall-seconds (default = max_runtime_s = 1×; a healthy sim is faster).
    h["sim_wall_ceiling_s"] = int(ceil_arg) if ceil_arg else runtime_s
    if smoke:
        e["trainer"]["num_trainers"] = 48
        h["rounds"] = 4
        h["min_trainers_to_start"] = 40
        h["min_trainers_join_timeout_s"] = 120
        e["name"] = "dbg_smoke_" + e["name"]
    else:
        # High round cap so the wall/vclock budget (max_runtime_s) is the
        # binding stop condition, not an early round-count termination.
        h["rounds"] = 20000
        e["name"] = f"dbg_{e['name']}"
    e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
    kept.append(e)

if not kept:
    print(f"WARNING: no experiments matched baselines={baselines_str} mode={mode}",
          flush=True)
    sys.exit(0)

cfg["experiments"] = kept
yaml.safe_dump(cfg, open(outpath, "w"), sort_keys=False)
print(f"Generated {outpath} with {len(kept)} experiment(s): "
      f"{[e['name'] for e in kept]}", flush=True)
PY
}

run_node() {
  local label="$1" cfg="$2"
  echo "[$(date '+%F %T')] START $label -> $cfg" | tee -a "$LOGDIR/debug_run.log"
  python -m flame.launch.run_experiment "$cfg" --example-dir "$EX" \
      < /dev/null >> "$LOGDIR/${label}.out" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE  $label exit=$rc" | tee -a "$LOGDIR/debug_run.log"
}

# ---- smoke mode ----
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE DEBUG: 48 trainers, 4 rounds, baselines=${BASELINES} ==="
  cfg="$LOGDIR/dbg_smoke.yaml"
  # Clear any stale config from a previous invocation so a no-match run is
  # skipped (not silently re-running a leftover config).
  rm -f "$cfg"
  make_debug_yaml "$BASELINES" 240 "$cfg" 1 "$SIM_WALL_CEILING_S" "$MODE"
  [ -f "$cfg" ] && run_node "dbg_smoke" "$cfg"
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
echo "=== DEBUG RUN: baselines='$BASELINES' mode=$MODE runtime_s=$RUNTIME_S sim_wall_ceiling_s=${SIM_WALL_CEILING_S:-auto(=runtime_s)} ==="
cfg="$LOGDIR/debug_run.yaml"
# Clear any stale config so a no-match run is skipped (not silently re-running
# a previous baseline's leftover config).
rm -f "$cfg"
make_debug_yaml "$BASELINES" "$RUNTIME_S" "$cfg" 0 "$SIM_WALL_CEILING_S" "$MODE"

if [ ! -f "$cfg" ]; then
  echo "No experiments matched for baselines='$BASELINES'. Nothing to run."
  exit 0
fi

run_node "debug_run" "$cfg"
echo "Logs: $LOGDIR/debug_run.out"
echo "Run dirs: experiments/run_*dbg_*"
