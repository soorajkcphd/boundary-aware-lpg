#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LPG_WORKERS="${LPG_WORKERS:-8}"
LPG_TORCH_THREADS="${LPG_TORCH_THREADS:-1}"

case "${1:-}" in
  --serial)
    LPG_WORKERS=1
    ;;
  --parallel|"")
    ;;
  *)
    echo "Usage: $0 [--parallel|--serial]" >&2
    exit 2
    ;;
esac

if ! [[ "$LPG_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: LPG_WORKERS must be a positive integer; got '$LPG_WORKERS'." >&2
  exit 2
fi
if ! [[ "$LPG_TORCH_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: LPG_TORCH_THREADS must be a positive integer; got '$LPG_TORCH_THREADS'." >&2
  exit 2
fi

export LPG_WORKERS LPG_TORCH_THREADS
export LPG_FIGURE_DIR="$ROOT/figures"
export LPG_RESULT_DIR="$ROOT/results"
export LPG_TABLE_DIR="$ROOT/tables"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Seed-level multiprocessing is already used by the experiments. Keeping each
# worker single-threaded avoids 8 workers x 8 BLAS threads oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

required=(
  experiments/core_experiments.py
  experiments/state_dependent_policy.py
  experiments/conditioning_experiments.py
  experiments/pullback_armijo.py
  experiments/fixed_ratio_fisher.py
  experiments/verify_dexp.py
  experiments/build_tables.py
  experiments/validate_results.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

"$PYTHON_BIN" - <<'PYENV'
import importlib
import sys

required = ("numpy", "scipy", "matplotlib", "torch")
errors = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # report import failures, not only missing modules
        errors.append(f"{name}: {exc}")
if errors:
    raise SystemExit(
        "Python dependency check failed:\n  " + "\n  ".join(errors)
        + "\nInstall missing packages with: python -m pip install -r requirements.txt"
    )

print("Python executable:", sys.executable)
for name in required:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'available')}")
import torch
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PYENV

# Syntax-check without leaving Python-version-specific bytecode in the source tree.
"$PYTHON_BIN" - <<'PYCHECK'
import ast
from pathlib import Path
for path in sorted(Path("experiments").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checks: PASS")
PYCHECK

rm -rf figures results tables manuscript_figures logs
mkdir -p figures results tables manuscript_figures logs
RUN_START="$(date +%s)"
SUMMARY_LOG="$ROOT/logs/run_summary.log"
: > "$SUMMARY_LOG"

run_stage() {
  local name="$1"
  local log_file="$2"
  shift 2
  local started ended status
  started="$(date +%s)"
  printf '\n============================================================\n'
  printf 'START: %s\n' "$name"
  printf '============================================================\n'

  # The pipeline runs in this shell, where pipefail is active; a Python failure
  # cannot be hidden by a successful tee process.
  if "$@" 2>&1 | tee "$log_file"; then
    status=0
  else
    status=$?
  fi
  ended="$(date +%s)"
  printf '%s\t%s\n' "$name" "$((ended-started))" >> "$SUMMARY_LOG"
  if (( status != 0 )); then
    echo "ERROR: stage failed: $name (exit $status). See $log_file" >&2
    exit "$status"
  fi
  printf 'DONE: %s (%s seconds)\n' "$name" "$((ended-started))"
}

run_stage "Lie exponential convention check" logs/verify_dexp.log \
  "$PYTHON_BIN" -u experiments/verify_dexp.py

run_stage "Core diagnostics" logs/core_experiments.log \
  "$PYTHON_BIN" -u experiments/core_experiments.py

run_stage "Pullback Armijo calibration" logs/pullback_armijo.log \
  "$PYTHON_BIN" -u experiments/pullback_armijo.py

run_stage "Fixed-ratio Fisher experiment" logs/fixed_ratio_fisher.log \
  "$PYTHON_BIN" -u experiments/fixed_ratio_fisher.py

run_stage "State-dependent projected policy" logs/state_dependent_policy.log \
  "$PYTHON_BIN" -u experiments/state_dependent_policy.py

run_stage "Learning-rate and conditioning experiments" logs/conditioning_experiments.log \
  "$PYTHON_BIN" -u experiments/conditioning_experiments.py --experiment all

run_stage "Publication tables" logs/build_tables.log \
  "$PYTHON_BIN" -u experiments/build_tables.py

run_stage "Numerical and file validation" logs/validation.log \
  "$PYTHON_BIN" -u experiments/validate_results.py

# Current manuscript-compatible aliases. These are generated, not source files.
declare -A aliases=(
  [state_dependent_policy.png]=state_dependent_lpg_127.png
  [theory_aligned_diagnostics.png]=theory_aligned_diagnostics.png
  [pullback_armijo.png]=lie_pullback_armijo_multiblock.png
  [se3_radius_control.png]=exp_se3_radius_projection.png
  [fisher_alignment_histogram.png]=exp1_fisher_alignment_hist.png
  [fisher_isotropy_tracking.png]=isotropy_tracking.png
  [controlled_anisotropy.png]=controlled_anisotropy.png
  [conditioning_stress_test.png]=three_arm_conditioning_126.png
)
for source_name in "${!aliases[@]}"; do
  source_path="figures/$source_name"
  target_path="manuscript_figures/${aliases[$source_name]}"
  [[ -s "$source_path" ]] || { echo "ERROR: missing figure $source_path" >&2; exit 1; }
  cp "$source_path" "$target_path"
  [[ -s "$target_path" ]] || { echo "ERROR: failed to create $target_path" >&2; exit 1; }
done

# Reject stale outputs from previous runs.
while IFS= read -r -d '' file; do
  mtime="$(stat -c %Y "$file")"
  if (( mtime < RUN_START )); then
    echo "ERROR: stale output detected: $file" >&2
    exit 1
  fi
done < <(find figures results tables manuscript_figures logs -type f -print0)

TOTAL="$(( $(date +%s) - RUN_START ))"
printf 'TOTAL\t%s\n' "$TOTAL" >> "$SUMMARY_LOG"
printf '\nFull reproduction completed in %s seconds.\n' "$TOTAL"
printf 'Figures:            %s\n' "$ROOT/figures"
printf 'Manuscript aliases: %s\n' "$ROOT/manuscript_figures"
printf 'Results:            %s\n' "$ROOT/results"
printf 'Tables:             %s\n' "$ROOT/tables"
printf 'Logs:               %s\n' "$ROOT/logs"
