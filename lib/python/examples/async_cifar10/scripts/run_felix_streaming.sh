#!/bin/bash
# Felix streaming-misprioritization experiment driver (n=50, sim, UNIFORM).
#
#   run_felix_streaming.sh smoke         # tiny single-node sanity pass (all arms)
#   run_felix_streaming.sh node <1-4>    # ONE node: its baseline + oracle (uniform)
#   run_felix_streaming.sh run           # all arms on THIS node (single-node fallback)
#   run_felix_streaming.sh analyze       # oracle replay + figures on pooled runs
#
# 4-NODE PLAN (phase 1, uniform):
#   node1 (128c/500GB) felix+oracle   node2 (128c/500GB) feddance+oracle
#   node3 (96c/250GB)  oort+oracle     node4 (96c/250GB)  refl+oracle
#   Per node:   FELIX_NUM_GPUS=<n> run_felix_streaming.sh node <i>
#   Then pool ALL <node>/experiments/run_* dirs onto one box and: ... analyze
# See docs/EXPERIMENT_felix_streaming.md (Distributed execution & pooling).
set -u

# --- robust conda activation (mirrors debug_run.sh) ---
ENVNAME="${FLAME_CONDA_ENV:-dg_flame}"
CB=""
if command -v conda >/dev/null 2>&1; then
  CB="$(conda info --base 2>/dev/null)"
elif [ -n "${CONDA_EXE:-}" ]; then
  CB="$(dirname "$(dirname "$CONDA_EXE")")"
fi
if [ -z "$CB" ] || [ ! -f "$CB/etc/profile.d/conda.sh" ]; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CB="$c" && break
  done
fi
[ -f "$CB/etc/profile.d/conda.sh" ] || { echo "ERROR: conda not found; activate '$ENVNAME' yourself." >&2; exit 1; }
source "$CB/etc/profile.d/conda.sh"
conda activate "$ENVNAME" || { echo "ERROR: conda activate $ENVNAME failed" >&2; exit 1; }
echo "conda: base=$CB env=$ENVNAME python=$(which python)"

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # async_cifar10 example dir
REPO="$(cd "$EX/../../../.." && pwd)"                    # repo root
cd "$EX" || exit 1
SCR=expt_scripts_2026
CFG="$SCR/n50_alpha0.1_syn0_stream_unif_sim.yaml"
ORACLE="$REPO/scripts/analysis/oracle_misselection.py"
FIGS="$REPO/scripts/analysis/felix_streaming_figures.py"
TARGET="${FELIX_TARGET:-0.60}"
export FLAME_BATCH_CONTINUE_ON_ERROR=1

# Per-node GPU count flows into the generated YAMLs (sim falls back to CPU if 0).
export NUM_GPUS="${FELIX_NUM_GPUS:-8}"
regen() { python scripts/gen_dirichlet_split.py --alpha 0.1 --num-trainers 50 --seed 0
          python scripts/gen_n50_experiment.py; }

# Oracle replay on the 4 practical baselines (oracle arm IS the ground truth).
replay() {
  for d in experiments/run_*_n50_alpha0.1_syn0_*; do
    [ -d "$d/checkpoints" ] || { echo "skip (no checkpoints): $d"; continue; }
    case "$(basename "$d")" in *_oracle_*) continue;; esac
    echo "[oracle] $d"
    python "$ORACLE" --run-dir "$d" --target "$TARGET" >/dev/null 2>&1 \
      || echo "  oracle replay FAILED for $d"
  done
}

case "${1:-}" in
  smoke)
    echo "=== SMOKE: all uniform arms, tiny ==="
    SMOKE="/tmp/n50_felix_smoke.yaml"
    regen >/dev/null
    python - "$CFG" "$SMOKE" <<'PY'
import sys, yaml
src, out = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(src))
for e in d["experiments"]:
    h = e["aggregator"]["config_overrides"]["hyperparameters"]
    h.update(rounds=6, max_runtime_s=180, evalEveryNRounds=2,
             targetAccuracy=0.2, stableEvalsAboveTarget=2,
             min_trainers_to_start=48, min_trainers_join_timeout_s=120)
    h["checkpoint"]["every_n_rounds"] = 2
    e["name"] = "smoke_" + e["name"]
    e["aggregator"]["config_overrides"]["job"]["id"] = e["name"]
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
print("wrote", out)
PY
    python -m flame.launch.run_experiment "$SMOKE" --example-dir "$EX" < /dev/null
    replay
    python "$FIGS" --runs-root "$EX/experiments" --target 0.2 || true
    ;;
  node)
    i="${2:?usage: run_felix_streaming.sh node <1-4>}"
    NCFG="$SCR/n50_alpha0.1_syn0_stream_unif_node${i}.yaml"
    echo "=== NODE $i (gpus=$NUM_GPUS): $NCFG ==="
    regen
    [ -f "$NCFG" ] || { echo "ERROR: $NCFG not found" >&2; exit 1; }
    python -m flame.launch.run_experiment "$NCFG" --example-dir "$EX" < /dev/null
    echo "node $i done. Pool experiments/run_* dirs onto the analysis box, then: $0 analyze"
    ;;
  run)
    echo "=== FULL: all uniform arms on THIS node (single-node fallback) ==="
    regen
    python -m flame.launch.run_experiment "$CFG" --example-dir "$EX" < /dev/null
    replay
    python "$FIGS" --runs-root "$EX/experiments" --target "$TARGET"
    ;;
  analyze)
    replay
    python "$FIGS" --runs-root "$EX/experiments" --target "$TARGET"
    ;;
  *)
    echo "usage: $0 smoke | node <1-4> | run | analyze"; exit 2;;
esac
echo "[$(date '+%F %T')] done."
