#!/bin/bash
# Post-run comparison for the overnight n300 runs. Two views:
#   1. PER-BASELINE sim-vs-real tracking (the only apples-to-apples pairing):
#      parity_check.py for each baseline's sim run dir vs its real run dir.
#   2. CROSS-BASELINE, separately for sim and for real (one plot set each):
#      analyze_run.py --compare-streaming over the 4 runs of a mode.
#
#   compare_overnight.sh            # auto-discovers latest run dir per (baseline,mode)
# Output: /tmp/overnight_compare/{parity_<baseline>.txt, parity_<baseline>.json, sim_cross/, real_cross/}
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

# paths derived from this script's location (portable across nodes)
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EX" || exit 1
ROOT="$(cd "$EX/../../../.." && pwd)"
OUT=/tmp/overnight_compare; mkdir -p "$OUT"
BASELINES="felix oort refl feddance"

# latest run dir for a (baseline, mode) — names end in _stream_<mode>
rundir() { ls -dt experiments/run_*_${1}_n300_*_stream_${2}* 2>/dev/null | head -1; }

echo "### 1. PER-BASELINE sim-vs-real parity ###"
for b in $BASELINES; do
  dr=$(rundir "$b" real); ds=$(rundir "$b" sim)
  if [ -z "$dr" ] || [ -z "$ds" ]; then
    echo "  $b: missing run dir (real='$dr' sim='$ds')"; continue
  fi
  echo "  $b: real=$(basename "$dr") sim=$(basename "$ds")"
  python scripts/parity_check.py \
    --real "$dr" --sim "$ds" \
    --json-out "$OUT/parity_${b}.json" \
    > "$OUT/parity_${b}.txt" 2>&1
  grep -E "\[OK\]|\[!!\]|\[XX\]|CHECKS" "$OUT/parity_${b}.txt" | sed 's/^/      /'
done

echo "### 2. CROSS-BASELINE (sim plots, then real plots) ###"
for mode in sim real; do
  dirs=(); labels=()
  for b in $BASELINES; do
    d=$(rundir "$b" "$mode"); [ -z "$d" ] && continue
    dirs+=("$d/telemetry"); labels+=("$b")
  done
  if [ ${#dirs[@]} -ge 2 ]; then
    echo "  $mode: ${labels[*]}"
    python "$ROOT"/scripts/analysis/analyze_run.py \
      --compare-streaming "${dirs[@]}" --labels "${labels[@]}" \
      --out "$OUT/${mode}_cross" > "$OUT/${mode}_cross.log" 2>&1
    echo "    -> $OUT/${mode}_cross/ ($(find "$OUT/${mode}_cross" -name '*.pdf' 2>/dev/null | wc -l) plots)"
  else
    echo "  $mode: <2 run dirs, skipping cross-baseline"
  fi
done
echo "DONE -> $OUT"
