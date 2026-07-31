# Boundary-Aware Lie-Projected Policy Optimization

Reproducibility code for the manuscript:

**Boundary-Aware Lie-Projected Policy Optimization in Matrix Lie-Algebra Coordinates**

The repository provides one command that regenerates the eight manuscript
figures, the numerical result files, and publication-facing tables from source.

## Repository contents

```text
boundary-aware-lpg/
├── README.md
├── requirements.txt
├── reproduce_all.sh
├── LICENSE
├── CITATION.cff
├── .gitignore
└── experiments/
    ├── core_experiments.py
    ├── state_dependent_policy.py
    ├── conditioning_experiments.py
    ├── pullback_armijo.py
    ├── fixed_ratio_fisher.py
    ├── verify_dexp.py
    ├── build_tables.py
    └── validate_results.py
```

Generated figures, results, tables, aliases, and logs are intentionally excluded
from version control and are recreated by `reproduce_all.sh`.

## Installation

An existing Python environment may be used. The validated experiment code
requires NumPy, SciPy, Matplotlib, and PyTorch.

```bash
conda activate sppg
python -m pip install -r requirements.txt
```

When a CUDA-enabled PyTorch installation is already present, do not reinstall
PyTorch merely for this repository. The numerical experiments use CPU execution
for reproducibility; CUDA availability does not change the protocol. The
requirements file uses version ranges, so a compatible existing CUDA-enabled
PyTorch installation is not replaced.

Verify the active interpreter:

```bash
python - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

## Full reproduction

Parallel execution uses up to eight worker processes:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" ./reproduce_all.sh --parallel
```

A one-worker audit run uses the identical scientific workload:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" ./reproduce_all.sh --serial
```

The launcher deletes all generated output directories before execution, records
a log and elapsed time for every stage, propagates Python failures through the
logging pipeline, validates the expected numerical values, and rejects stale
artifacts. Parallel runs keep each BLAS worker single-threaded to avoid CPU
oversubscription.

## Canonical experiment order

The full run follows the experiment order used to produce the manuscript:

1. Lie-exponential convention check
2. Retained core diagnostics
3. Pullback and projected-Armijo calibration
4. Fixed-ratio Fisher experiment
5. State-dependent radius-projected policy experiment
6. Extended learning-rate ablation
7. Three-arm coordinate-conditioning stress test
8. Table construction and numerical validation

The removed dense-Fisher benchmark is not part of the retained manuscript
protocol and is not included.

## Manuscript figures

The code generates professional figure filenames in `figures/`:

| Figure | Generator |
|---|---|
| `state_dependent_policy.png` | `state_dependent_policy.py` |
| `theory_aligned_diagnostics.png` | `core_experiments.py` |
| `pullback_armijo.png` | `pullback_armijo.py` |
| `se3_radius_control.png` | `core_experiments.py` |
| `fisher_alignment_histogram.png` | `core_experiments.py` |
| `fisher_isotropy_tracking.png` | `core_experiments.py` |
| `controlled_anisotropy.png` | `core_experiments.py` |
| `conditioning_stress_test.png` | `conditioning_experiments.py` |

For direct use with the current LaTeX manuscript, the launcher also creates
`manuscript_figures/` with the exact filenames referenced by the paper.

## Numerical tables

Fresh CSV tables are written to `tables/`, including:

- theory-aligned stochastic projected-mapping rates;
- state-dependent constrained and unconstrained policy summaries;
- pullback Armijo iterations;
- SE(3) Fisher and radius-control diagnostics;
- joint-count scaling;
- Fisher alignment and controlled anisotropy;
- fixed-ratio Fisher estimation;
- learning-rate selection;
- three-arm conditioning comparisons.

`tables/manuscript_table_rows.tex` provides formatted rows for checking the
numbers in the manuscript. The manuscript itself does not depend on a generated
LaTeX macro file.

## Expected checks

The validator checks values at the manuscript reporting precision, including:

```text
Boundary raw-gradient norm        approximately 1.0712
Theory-aligned fitted slope       approximately -0.519
State-dependent selected step     0.0025
State-dependent selected radius   2.0
Constrained final norm            2.000
Unconstrained final norm          approximately 3.339
Selected rate for all three arms  0.015625
Direct final return               approximately -897.15
Redundant final return            approximately -1326.06
Metric-corrected final return     approximately -897.14
```

The complete validation report, including selected hyperparameters and output
inventory checks, is written to:

```text
results/validation_report.txt
```

## Consolidation from the research working directory

The public repository uses generic, descriptive filenames. The scientific
protocol was consolidated as follows:

| Research filename | Public repository file |
|---|---|
| `experiments_amai_submission.py` and its retained launcher logic | `experiments/core_experiments.py` and `reproduce_all.sh` |
| `run_state_dependent_lpg_126.py` | `experiments/state_dependent_policy.py` |
| `run_extended_lr_ng_ablation_126_fast.py` | `experiments/conditioning_experiments.py` |
| `run_three_arm_conditioning_126.py` | `experiments/conditioning_experiments.py` |
| `lie_pullback_armijo_multiblock.py` | `experiments/pullback_armijo.py` |
| `fisher_population_fixed_ratio.py` | `experiments/fixed_ratio_fisher.py` |
| `verify_dexp_convention.py` | `experiments/verify_dexp.py` |

The consolidation changes naming, output locations, orchestration, and reporting
only. Seeds, iteration counts, episode counts, learning rates, radius grid,
Fisher sample counts, and evaluation formulas are retained.

## Output directories

```text
figures/             professional figure filenames
manuscript_figures/  aliases matching the current LaTeX source
results/             JSON, NPZ, CSV, summaries, and validation report
tables/              publication-facing CSV and LaTeX rows
logs/                complete stage logs
```

## Reproducibility notes

- Random seeds are fixed in the source.
- Five evaluation seeds are used for the state-dependent and conditioning
  experiments.
- The state-dependent experiment runs the full 25-point pilot grid before
  held-out evaluation.
- The learning-rate grid is
  `{0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5}`.
- The smallest grid value is selected for all three arms; therefore the
  experiment is not interpreted as a tuned-rate ranking.
- The fixed-step conditioning comparison uses learning rate `0.25`.
- The repository does not include manuscript drafts, journal submission files,
  archived plots, old outputs, internal review documents, or dense-benchmark
  code.

## Citation

Citation metadata are provided in `CITATION.cff`.
