#!/bin/bash
# Overnight runner + smoke test for the n300 OVERNIGHT node configs (each runs
# sim THEN real per baseline, both with the join barrier).
#
#   overnight_run.sh smoke           # short, reduced-scale run of ALL node exps
#   overnight_run.sh run node1       # full overnight, node1 (felix+oort, sim+real)
#   overnight_run.sh run node2       # full overnight, node2 (refl+feddance)
#
# The launcher (run_experiment) is now non-interactive (FLAME_BATCH_CONTINUE_ON_ERROR)
# and hard-sweeps stragglers + waits for GPU to drain between experiments, so one
# failed/hung run is cleaned off the system and the next proceeds. Logs per node.
set -u

# --- robust conda activation (handles non-default install locations) ---
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

# repo example dir, derived from this script's location (portable across nodes)
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EX" || exit 1
SCR=expt_scripts_2026
LOGDIR=/tmp/overnight_logs; mkdir -p "$LOGDIR"
export FLAME_BATCH_CONTINUE_ON_ERROR=1

run_node() {  # $1 = node1|node2  $2 = config yaml
  local node="$1" cfg="$2"
  echo "[$(date '+%F %T')] START $node -> $cfg" | tee -a "$LOGDIR/overnight.log"
  python -m flame.launch.run_experiment "$cfg" --example-dir "$EX" \
      < /dev/null >> "$LOGDIR/${node}.out" 2>&1
  echo "[$(date '+%F %T')] DONE  $node exit=$?" | tee -a "$LOGDIR/overnight.log"
}

make_smoke() {  # writes short reduced configs to $LOGDIR/smoke_*.yaml
  python - "$SCR" "$LOGDIR" <<'PY'
import yaml, sys, copy, glob, os
scr, out = sys.argv[1], sys.argv[2]
for node in (1, 2):
    src = f"{scr}/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_node{node}.yaml"
    d = yaml.safe_load(open(src))
    for e in d["experiments"]:
        e["trainer"]["num_trainers"] = 48
        h = e["aggregator"]["config_overrides"]["hyperparameters"]
        h["rounds"] = 4
        h["max_runtime_s"] = 240
        h["min_trainers_to_start"] = 40
        h["min_trainers_join_timeout_s"] = 120
        e["name"] = "smoke_" + e["name"]
        e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
    fn = f"{out}/smoke_node{node}.yaml"
    yaml.safe_dump(d, open(fn, "w"), sort_keys=False)
    print(fn)
PY
}

case "${1:-}" in
  smoke)
    echo "=== SMOKE: 48 trainers, 4 rounds, all 8 node experiments ==="
    make_smoke
    for node in 1 2; do
      run_node "smoke_node$node" "$LOGDIR/smoke_node$node.yaml"
    done
    echo "=== SMOKE RESULTS ==="
    for d in experiments/run_*smoke_*; do
      [ -d "$d" ] || continue
      t=$(ls "$d"/telemetry/aggregator_*.jsonl 2>/dev/null | head -1)
      rounds=$(grep -c "agg_round" "$t" 2>/dev/null || echo 0)
      printf "  %-50s agg_round_events=%s\n" "$(basename "$d")" "$rounds"
    done
    ;;
  run)
    node="${2:?usage: overnight_run.sh run node1|node2}"
    run_node "$node" "$SCR/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_${node}.yaml"
    ;;
  *)
    echo "usage: $0 smoke | run node1 | run node2"; exit 2;;
esac
