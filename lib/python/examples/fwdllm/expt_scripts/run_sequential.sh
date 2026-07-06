#!/bin/bash
# Run multiple fwdllm YAMLs (fwdllm, fwdllm_plus, fluxtune) one after
# another, in a single conda env, logging each run separately.
#
# Each YAML auto-terminates once data_id reaches a threshold or after a
# wall-time cap, whichever comes first. This script overrides those caps
# per invocation via --max-runtime-s/--max-data-id, generating a patched
# copy of each YAML rather than editing the originals.
#
# Usage (from anywhere):
#   run_sequential.sh [--max-runtime-s 600] [--max-data-id 10]
#       [--num-trainers N] [--num-gpus N] [--c C] [--k K] [--stop-on-fail]
#       [--partition-method NAME] [--only name1,name2]
#
#   --max-runtime-s  wall-clock cap in seconds for each run (default: 600 = 10 min)
#   --max-data-id    stop a run once data_id reaches this value (default: 10)
#   --num-trainers   override trainer.num_trainers (default: each YAML's own, 10)
#   --num-gpus       override execution.num_gpus (default: each YAML's own, 1).
#                     Scale this with --num-trainers -- each YAML's default of
#                     1 GPU is sized for its own default 10-trainer count.
#   --c              override selector.kwargs.c + minInitialTrainers + agg_goal
#                     (agg_goal matches c so no selected trainer goes stranded)
#                     -- superseded per-field by --agg-goal/--min-initial-trainers
#                     when those are also passed (see below).
#   --c-async        override selector.kwargs.c only for the async baseline
#                     (fluxtune) -- lets sync baselines run concurrency==agg_goal
#                     via --c/--agg-goal while fluxtune overcommits concurrency
#                     independent of agg_goal (matches async_oort's design: c
#                     ends in flight, agg_goal of them counted per round).
#   --agg-goal       override aggregator.agg_goal directly (fans into
#                     hyperparameters.aggGoal + selector.kwargs.aggGoal/aggr_num
#                     per runner.py) independent of --c/--c-async. When --c is
#                     also given without this, legacy behavior (agg_goal==c)
#                     still applies.
#   --min-initial-trainers  override selector.kwargs.minInitialTrainers
#                     directly, independent of --num-trainers/--c.
#   --avail-trace    override the availability trace used by ALL baselines:
#                     trainer.availability.mode (cosmetic/consistency),
#                     trainer hyperparameters.client_notify.trace (fluxtune's
#                     real signal), and aggregator
#                     hyperparameters.trackTrainerAvail.trace (fwdllm_plus's
#                     real ORACULAR signal). Use e.g. "syn_0" (always
#                     available) to isolate selection/aggregation bugs from
#                     trace-driven scarcity/churn.
#   --avail-traces   comma-separated list of traces, e.g. "syn_0,syn_20" --
#                     runs the ENTIRE --only baseline sequence once per trace,
#                     back to back, in this one invocation/process (for an
#                     unattended overnight multi-trace comparison; no need to
#                     babysit and launch the next trace by hand). Takes
#                     precedence over --avail-trace if both are given. Each
#                     (baseline, trace) run's name/log/results are
#                     disambiguated by trace -- see the run-name note below.
#   --k              override selector.kwargs.k
#   --stop-on-fail   abort the remaining runs as soon as one exits non-zero
#                    (default: run all three regardless, report at the end)
#   --partition-method  override hyperparameters.partition_method on both the
#                     trainer and aggregator sides (default: each YAML's own,
#                     "uniform" -- IID, chosen for smoke tests to isolate
#                     launcher-mechanics validation from data-skew effects).
#                     Must be one of agnews_partition.h5's own group names,
#                     e.g. "niid_label_clients=100_alpha=0.1" for the most
#                     heterogeneous split available in the 100-client group
#                     (smaller alpha = more skewed/non-IID).
#   --only           comma-separated subset of baselines to execute, e.g.
#                     --only fwdllm_plus,fluxtune
#                     (default: all three -- fwdllm, fwdllm_plus, fluxtune)
#                     These are plain baseline names, independent of
#                     --num-trainers -- the "n10" in each source YAML's
#                     filename is just that file's own default trainer
#                     count, not part of the run's identity.
set -u

# --- robust conda activation (same pattern as scripts/debug_run.sh) ---
# Env choice: FLAME_CONDA_ENV overrides; otherwise use whatever conda env is
# already active in the launching shell (CONDA_DEFAULT_ENV). No hardcoded
# fallback -- activate an env before calling this script, or set
# FLAME_CONDA_ENV explicitly.
ENVNAME="${FLAME_CONDA_ENV:-${CONDA_DEFAULT_ENV:-}}"
if [ -z "$ENVNAME" ]; then
  echo "ERROR: no conda env active in this shell and FLAME_CONDA_ENV not set." >&2
  echo "       Activate an env first (conda activate <name>) or pass FLAME_CONDA_ENV=<name>." >&2
  exit 1
fi
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

# repo paths (portable across nodes/checkouts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # .../examples/fwdllm
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"       # flame/

# Force this checkout's flame package ahead of anything already on
# sys.path (e.g. a stale `pip install -e` editable pointing at a
# different clone) so the code that actually runs matches this repo.
export PYTHONPATH="$REPO_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"

# defaults
MAX_RUNTIME_S=600   # 10 minutes
MAX_DATA_ID=10
STOP_ON_FAIL=0
NUM_TRAINERS=""   # empty = leave each YAML's own value
NUM_GPUS=""       # empty = leave each YAML's own value
SEL_C=""
SEL_C_ASYNC=""
SEL_K=""
AGG_GOAL=""
MIN_INIT_TRAINERS=""
AVAIL_TRACE=""
AVAIL_TRACES=""
PARTITION_METHOD=""   # empty = leave each YAML's own value ("uniform")
ONLY=""           # empty = run all three

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runtime-s)     MAX_RUNTIME_S="$2"; shift 2 ;;
    --max-data-id)       MAX_DATA_ID="$2"; shift 2 ;;
    --num-trainers)      NUM_TRAINERS="$2"; shift 2 ;;
    --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
    --c)                 SEL_C="$2"; shift 2 ;;
    --c-async)           SEL_C_ASYNC="$2"; shift 2 ;;
    --k)                 SEL_K="$2"; shift 2 ;;
    --agg-goal)          AGG_GOAL="$2"; shift 2 ;;
    --min-initial-trainers) MIN_INIT_TRAINERS="$2"; shift 2 ;;
    --avail-trace)       AVAIL_TRACE="$2"; shift 2 ;;
    --avail-traces)      AVAIL_TRACES="$2"; shift 2 ;;
    --stop-on-fail)      STOP_ON_FAIL=1; shift ;;
    --partition-method)  PARTITION_METHOD="$2"; shift 2 ;;
    --only)              ONLY="$2"; shift 2 ;;
    *) echo "usage: $0 [--max-runtime-s SECONDS] [--max-data-id N] [--num-trainers N] [--num-gpus N] [--c C] [--c-async C] [--k K] [--agg-goal N] [--min-initial-trainers N] [--avail-trace NAME | --avail-traces NAME1,NAME2,...] [--stop-on-fail] [--partition-method NAME] [--only name1,name2]" >&2; exit 2 ;;
  esac
done

# --avail-traces takes precedence; otherwise fall back to the single
# --avail-trace (may be empty, meaning "leave each YAML's own trace").
if [ -n "$AVAIL_TRACES" ]; then
  IFS=',' read -ra TRACE_LIST <<< "$AVAIL_TRACES"
else
  TRACE_LIST=("$AVAIL_TRACE")
fi
MULTI_TRACE=0
[ "${#TRACE_LIST[@]}" -gt 1 ] && MULTI_TRACE=1

LOGDIR="$SCRIPT_DIR/smoke_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOGDIR"

# Patch hyperparameters.max_runtime_s / max_data_id_progress, and optionally
# num_trainers / selector c+k+minInitialTrainers+agg_goal, in a copy of the
# YAML rather than the original -- keeps the checked-in smoke configs stable
# while letting this script's caller pick the scale per invocation.
patch_yaml() {
  python - "$1" "$2" "$3" "$MAX_RUNTIME_S" "$MAX_DATA_ID" "$NUM_TRAINERS" "$NUM_GPUS" "$SEL_C" "$SEL_K" "$PARTITION_METHOD" "$SEL_C_ASYNC" "$AGG_GOAL" "$MIN_INIT_TRAINERS" "$AVAIL_TRACE" <<'PY'
import sys, yaml
(src, dst, run_key, max_runtime_s, max_data_id, num_trainers, num_gpus, sel_c,
 sel_k, partition_method, sel_c_async, agg_goal, min_init_trainers,
 avail_trace) = sys.argv[1:15]
# Only baseline in ALL_RUNS below that's async; --c-async targets it
# specifically so one invocation can decouple sync concurrency (==agg_goal)
# from async concurrency (overcommitted vs agg_goal) -- see async_oort.py.
IS_ASYNC_BASELINE = run_key == "fluxtune"
cfg = yaml.safe_load(open(src))
for exp in cfg.get("experiments", []):
    h = exp["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = int(max_runtime_s)
    h["max_data_id_progress"] = int(max_data_id)
    if partition_method:
        h["partition_method"] = partition_method
        exp["trainer"]["config_overrides"]["hyperparameters"]["partition_method"] = partition_method
    if num_trainers:
        exp["trainer"]["num_trainers"] = int(num_trainers)
        # exp["name"] feeds the run directory name (run_<ts>_<name>); derive
        # it from run_key + the actual trainer count rather than copying the
        # source YAML's own checked-in name, which only reflects that file's
        # default count. job.id must track exp["name"] (every checked-in
        # YAML keeps them equal; it's the MQTT job/task id shared with
        # trainers via runner.py). Include avail_trace when set so runs
        # launched back-to-back under different traces (--avail-traces)
        # don't produce identically-named run dirs/job ids.
        new_name = (
            f"{run_key}_n{num_trainers}_{avail_trace}_smoke"
            if avail_trace else f"{run_key}_n{num_trainers}_smoke"
        )
        exp["name"] = new_name
        exp["aggregator"]["config_overrides"]["job"]["id"] = new_name
    if num_gpus:
        exp["execution"]["num_gpus"] = int(num_gpus)
    kwargs = exp["aggregator"]["config_overrides"]["selector"]["kwargs"]
    if sel_c:
        kwargs["c"] = int(sel_c)
        if not min_init_trainers:
            kwargs["minInitialTrainers"] = int(num_trainers) if num_trainers else int(sel_c)
        if not agg_goal:
            # legacy behavior: agg_goal matches c so no selected trainer goes
            # uncounted/stranded. Superseded by --agg-goal below when given.
            exp["aggregator"]["agg_goal"] = int(sel_c)
    if sel_c_async and IS_ASYNC_BASELINE:
        kwargs["c"] = int(sel_c_async)
    if sel_k:
        kwargs["k"] = int(sel_k)
    if agg_goal:
        exp["aggregator"]["agg_goal"] = int(agg_goal)
    if min_init_trainers:
        kwargs["minInitialTrainers"] = int(min_init_trainers)
    if avail_trace:
        # Cosmetic/consistency: trainer-side self-reported mode.
        exp["trainer"].setdefault("availability", {})["mode"] = avail_trace
        # Real signal for fluxtune (client_notify) and fwdllm/fwdllm_plus
        # (dormant unless trackTrainerAvail below is ORACULAR).
        t_hp = exp["trainer"].setdefault("config_overrides", {}).setdefault("hyperparameters", {})
        t_hp.setdefault("client_notify", {})["trace"] = avail_trace
        # Real signal for fwdllm_plus (ORACULAR tracking reads this trace
        # directly rather than waiting on trainer self-reports).
        a_hp = exp["aggregator"]["config_overrides"]["hyperparameters"]
        a_hp.setdefault("trackTrainerAvail", {})["trace"] = avail_trace
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
}

# Keys are plain baseline names -- independent of --num-trainers and of
# whatever scale is baked into each source YAML's own filename/checked-in
# default. The mapping to the actual YAML file lives only here.
ALL_RUNS=(
  "fwdllm:$SCRIPT_DIR/fwdllm_n10_smoke.yaml"
  "fwdllm_plus:$SCRIPT_DIR/fwdllm_plus_n10_smoke.yaml"
  "fluxtune:$SCRIPT_DIR/fluxtune_n10_smoke.yaml"
)

if [ -n "$ONLY" ]; then
  RUNS=()
  IFS=',' read -ra ONLY_NAMES <<< "$ONLY"
  for want in "${ONLY_NAMES[@]}"; do
    found=0
    for entry in "${ALL_RUNS[@]}"; do
      if [ "${entry%%:*}" = "$want" ]; then
        RUNS+=("$entry")
        found=1
        break
      fi
    done
    if [ "$found" = "0" ]; then
      echo "ERROR: --only name '$want' not recognized. Valid names: ${ALL_RUNS[*]%%:*}" >&2
      exit 2
    fi
  done
else
  RUNS=("${ALL_RUNS[@]}")
fi

declare -A RESULT
declare -A DURATION_S
ORDERED_KEYS=()   # (name or name@trace) in the order actually run, for the summary

CHILD_PID=""
cleanup() {
  echo ""
  echo "Interrupted. Killing child (PID=${CHILD_PID:-none})..."
  [ -n "$CHILD_PID" ] && kill -- -"$CHILD_PID" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

cd "$REPO_ROOT" || exit 1
echo "=== fwdllm sequential run: ${#RUNS[@]} runs (${RUNS[*]%%:*}) x ${#TRACE_LIST[@]} trace(s) (${TRACE_LIST[*]:-<yaml default>}), max_runtime_s=$MAX_RUNTIME_S max_data_id=$MAX_DATA_ID num_trainers=${NUM_TRAINERS:-<yaml default>} num_gpus=${NUM_GPUS:-<yaml default>} c=${SEL_C:-<yaml default>} k=${SEL_K:-<yaml default>} partition_method=${PARTITION_METHOD:-<yaml default>}, logs in $LOGDIR ==="

STOP_ALL=0
for trace in "${TRACE_LIST[@]}"; do
  AVAIL_TRACE="$trace"   # read by patch_yaml() via the outer AVAIL_TRACE var
  [ "$MULTI_TRACE" = "1" ] && echo "--- trace: ${trace:-<yaml default>} ---"

  for entry in "${RUNS[@]}"; do
    name="${entry%%:*}"
    src_cfg="${entry#*:}"
    # Disambiguate by trace only when actually looping multiple traces, so a
    # single-trace (or no-trace) invocation keeps today's exact file/key names.
    if [ "$MULTI_TRACE" = "1" ]; then
      key="${name}@${trace:-default}"
    else
      key="$name"
    fi
    cfg="$LOGDIR/${key}.yaml"
    log="$LOGDIR/${key}.out"
    patch_yaml "$src_cfg" "$cfg" "$name"

    start_ts=$(date +%s)
    python -m flame.launch.run_experiment "$cfg" --example-dir "$EXAMPLE_DIR" \
        < /dev/null > "$log" 2>&1 &
    CHILD_PID=$!
    echo "[$(date '+%F %T')] START $key (PID=$CHILD_PID) -> $cfg (log: $log)"
    echo "  (to kill: kill -9 $CHILD_PID   or Ctrl+C)"
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""
    end_ts=$(date +%s)
    DURATION_S[$key]=$((end_ts - start_ts))
    ORDERED_KEYS+=("$key")
    if [ $rc -eq 0 ]; then
      RESULT[$key]="PASS"
    else
      RESULT[$key]="FAIL(exit=$rc)"
    fi
    echo "[$(date '+%F %T')] DONE  $key -> ${RESULT[$key]} (${DURATION_S[$key]}s)"

    if [ $rc -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
      echo "--stop-on-fail set; aborting remaining runs (including remaining traces)."
      STOP_ALL=1
      break
    fi
  done
  [ "$STOP_ALL" = "1" ] && break
done

echo ""
echo "=== Summary ==="
for key in "${ORDERED_KEYS[@]}"; do
  printf "  %-35s %-15s %ss\n" "$key" "${RESULT[$key]:-SKIPPED}" "${DURATION_S[$key]:-0}"
done
echo "Logs: $LOGDIR"
echo "Run dirs: $EXAMPLE_DIR/experiments/run_*"
