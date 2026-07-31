#!/usr/bin/env python3
#Validate the complete reproduction output inventory and key numerical checks.
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = Path(os.environ.get("LPG_FIGURE_DIR", ROOT / "figures")).resolve()
RESULT_DIR = Path(os.environ.get("LPG_RESULT_DIR", ROOT / "results")).resolve()
TABLE_DIR = Path(os.environ.get("LPG_TABLE_DIR", ROOT / "tables")).resolve()

FIGURES = [
    "state_dependent_policy.png",
    "theory_aligned_diagnostics.png",
    "pullback_armijo.png",
    "se3_radius_control.png",
    "fisher_alignment_histogram.png",
    "fisher_isotropy_tracking.png",
    "controlled_anisotropy.png",
    "conditioning_stress_test.png",
]

TABLES = [
    "theory_aligned_rate.csv",
    "state_dependent_policy.csv",
    "state_dependent_paired_difference.csv",
    "pullback_armijo_selected_iterations.csv",
    "pullback_armijo_all_iterations.csv",
    "se3_fisher.csv",
    "se3_radius_control.csv",
    "scalability.csv",
    "fisher_alignment.csv",
    "controlled_anisotropy.csv",
    "fixed_ratio_fisher.csv",
    "conditioning_stress_test.csv",
    "conditioning_comparisons.csv",
    "learning_rate_study.csv",
    "learning_rate_selected.csv",
    "manuscript_table_rows.tex",
]

JSON_RESULTS = [
    "core_experiments.json",
    "state_dependent_policy.json",
    "learning_rate_ablation.json",
    "conditioning_stress_test.json",
]

BINARY_RESULTS = [
    "state_dependent_policy.npz",
    "conditioning_stress_test.npz",
    "pullback_armijo.csv",
    "pullback_armijo_summary.txt",
    "fixed_ratio_fisher.csv",
]


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise RuntimeError(f"missing or empty output: {path}")


def close(actual: float, expected: float, tolerance: float, label: str) -> str:
    difference = abs(actual - expected)
    if difference > tolerance:
        raise RuntimeError(
            f"{label}: got {actual:.8g}, expected approximately {expected:.8g}, "
            f"tolerance {tolerance:.3g}"
        )
    return f"PASS {label}: {actual:.8g}"


def main() -> None:
    report: list[str] = []
    for name in FIGURES:
        path = FIGURE_DIR / name
        require_file(path, 5_000)
        report.append(f"PASS figure {name}: {path.stat().st_size} bytes")
    for name in TABLES:
        path = TABLE_DIR / name
        require_file(path, 5)
        report.append(f"PASS table {name}: {path.stat().st_size} bytes")
    for name in JSON_RESULTS:
        path = RESULT_DIR / name
        require_file(path, 20)
        json.loads(path.read_text(encoding="utf-8"))
        report.append(f"PASS JSON {name}")
    for name in BINARY_RESULTS:
        path = RESULT_DIR / name
        require_file(path, 20)
        report.append(f"PASS result {name}: {path.stat().st_size} bytes")

    core = json.loads((RESULT_DIR / "core_experiments.json").read_text(encoding="utf-8"))
    theory = core["theory_aligned"]
    report.append(close(theory["boundary_final_gradient_norm"], 1.0712, 5e-4, "boundary raw gradient"))
    report.append(close(theory["boundary_final_mapping_norm"], 0.0, 1e-10, "boundary mapping"))
    report.append(close(theory["slope"], -0.519, 0.01, "stochastic mapping slope"))

    fisher = core["fisher_alignment"]["summary"]
    report.append(close(fisher["alignment_mean"], 0.970, 0.01, "Fisher alignment mean"))
    report.append(close(fisher["condition_mean"], 2.53, 0.15, "Fisher condition mean"))

    anisotropy = core["controlled_anisotropy"]
    report.append(close(anisotropy[-1]["alignment"], 0.819, 0.02, "high-anisotropy alignment"))
    report.append(close(anisotropy[-1]["return_degradation_percent"], 15.2, 2.0, "high-anisotropy return degradation"))

    state = json.loads((RESULT_DIR / "state_dependent_policy.json").read_text(encoding="utf-8"))
    report.append(close(state["design"]["tau"], 0.0025, 1e-12, "selected state-dependent step"))
    report.append(close(state["design"]["R0"], 2.0, 1e-12, "selected state-dependent radius"))
    report.append(close(state["constrained"]["active_mean"], 0.67, 0.03, "active projection fraction"))
    report.append(close(state["constrained"]["norm_mean"], 2.0, 1e-10, "constrained final norm"))
    report.append(close(state["unconstrained"]["norm_mean"], 3.339, 0.02, "unconstrained final norm"))
    report.append(close(state["constrained"]["final_mean"], -29.22, 0.15, "constrained return"))
    report.append(close(state["unconstrained"]["final_mean"], -18.84, 0.15, "unconstrained return"))

    learning_rate = json.loads((RESULT_DIR / "learning_rate_ablation.json").read_text(encoding="utf-8"))
    for arm in ("direct", "redundant", "redundant_ng"):
        report.append(close(learning_rate[arm]["selected_lr"], 0.015625, 1e-12, f"{arm} selected rate"))

    conditioning = json.loads((RESULT_DIR / "conditioning_stress_test.json").read_text(encoding="utf-8"))
    report.append(close(conditioning["arms"]["direct"]["final_mean"], -897.15, 0.1, "direct stress-test return"))
    report.append(close(conditioning["arms"]["redundant"]["final_mean"], -1326.06, 0.1, "redundant stress-test return"))
    report.append(close(conditioning["arms"]["redundant_ng"]["final_mean"], -897.14, 0.1, "metric-corrected return"))

    fixed_rows = list(csv.DictReader((RESULT_DIR / "fixed_ratio_fisher.csv").open(encoding="utf-8")))
    report.append(close(float(fixed_rows[0]["empirical_kappa_mean"]), 2.242, 0.01, "fixed-ratio d=15 condition"))

    summary = {}
    for line in (RESULT_DIR / "pullback_armijo_summary.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            summary[key] = value
    report.append(close(float(summary["final_mapping_norm"]), 1.76e-7, 1e-8, "Armijo terminal mapping"))

    output = RESULT_DIR / "validation_report.txt"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
