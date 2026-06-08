#!/bin/bash
# Configurable debug runner for targeted baseline comparison.
#
# Generates per-node YAML configs on the fly from the OVERNIGHT templates,
# filtering to the requested baselines and overriding runtime. Designed for
# short 1h debugging runs to isolate overrun root causes between baselines.
#
# Usage (run on each node separately):
#   debug_run.sh --node node1 [--baselines felix] [--runtime-s 1800]
#   debug_run.sh --node node2 [--baselines refl]  [--runtime-s 1800]
#   debug_run.sh smoke        # 48 trainers, 4 rounds, all baselines
#
# Node→baseline assignment (matches OVERNIGHT config split):
#   node1: felix, oort
#   node2: refl, feddance
#
# --baselines is matched against the 'baseline:' field in the OVERNIGHT YAML
# for that node, so only baselines present on this node's config actually run.
# Specifying a baseline from the other node is a no-op (not an error).
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
NODE=""
RUNTIME_S=10800
BASELINES="felix refl"
SIM_WALL_CEILING_S=""  # empty = max_runtime_s (1×, tight guard; sim should be faster than real)

usage() {
  echo "usage: $0 --node node1|node2 [--baselines 'felix refl'] [--runtime-s 3600] [--sim-wall-ceiling-s 2700]"
  echo "       $0 smoke"
  echo ""
  echo "  --sim-wall-ceiling-s  wall-clock ceiling for sim mode (default: = runtime_s)."
  echo "                        A well-behaved sim finishes in <= real-mode wall time."
  echo "                        Fires [SIM_WALL_CEILING] warning + stops when exceeded."
  exit 2
}

# parse args
if [ "${1:-}" = "smoke" ]; then
  NODE="smoke"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --node)                NODE="$2"; shift 2 ;;
      --baselines)           BASELINES="$2"; shift 2 ;;
      --runtime-s)           RUNTIME_S="$2"; shift 2 ;;
      --sim-wall-ceiling-s)  SIM_WALL_CEILING_S="$2"; shift 2 ;;
      --wall-runtime-s)      SIM_WALL_CEILING_S="$2"; shift 2 ;;  # backward compat alias
      *) usage ;;
    esac
  done
  [ -z "$NODE" ] && usage
fi

# Generate a filtered+patched YAML from the OVERNIGHT source configs.
# $1 = node (node1|node2), $2 = baselines (space-separated), $3 = runtime_s,
# $4 = output path, [$5 = smoke: 1|0], [$6 = sim_wall_ceiling_s: int or ""]
make_debug_yaml() {
  python - "$SCR" "$1" "$2" "$3" "$4" "${5:-0}" "${6:-}" <<'PY'
import yaml, sys, copy
scr, node, baselines_str, runtime_s, outpath, smoke, ceil_arg = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6] == "1",
    sys.argv[7] if len(sys.argv) > 7 else ""
)
requested = set(baselines_str.lower().split())

src = f"{scr}/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_{node}.yaml"
try:
    d = yaml.safe_load(open(src))
except FileNotFoundError:
    print(f"SKIP: no config for {node}", flush=True)
    sys.exit(0)

kept = []
for e in d["experiments"]:
    bl = e.get("baseline", "").lower()
    if bl not in requested:
        continue
    e = copy.deepcopy(e)
    h = e["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = runtime_s
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
        # High round cap so the 3h wall/vclock budget (max_runtime_s) is the
        # binding stop condition, not an early round-count termination.
        h["rounds"] = 20000
        e["name"] = f"dbg_{e['name']}"
    e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
    kept.append(e)

if not kept:
    print(f"WARNING: no experiments matched baselines={baselines_str} on {node}", flush=True)
    sys.exit(0)

d["experiments"] = kept
yaml.safe_dump(d, open(outpath, "w"), sort_keys=False)
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
if [ "$NODE" = "smoke" ]; then
  echo "=== SMOKE DEBUG: 48 trainers, 4 rounds, baselines=${BASELINES} ==="
  for node in node1 node2; do
    cfg="$LOGDIR/dbg_smoke_${node}.yaml"
    make_debug_yaml "$node" "$BASELINES" 240 "$cfg" 1 "$SIM_WALL_CEILING_S"
    [ -f "$cfg" ] && run_node "dbg_smoke_$node" "$cfg"
  done
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
echo "=== DEBUG RUN: node=$NODE baselines='$BASELINES' runtime_s=$RUNTIME_S sim_wall_ceiling_s=${SIM_WALL_CEILING_S:-auto(=runtime_s)} ==="
cfg="$LOGDIR/debug_${NODE}.yaml"
make_debug_yaml "$NODE" "$BASELINES" "$RUNTIME_S" "$cfg" 0 "$SIM_WALL_CEILING_S"

if [ ! -f "$cfg" ]; then
  echo "No experiments matched for node=$NODE baselines='$BASELINES'. Nothing to run."
  exit 0
fi

run_node "debug_$NODE" "$cfg"
echo "Logs: $LOGDIR/debug_${NODE}.out"
echo "Run dirs: experiments/run_*dbg_*"
