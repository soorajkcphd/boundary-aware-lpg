#!/usr/bin/env python3
# Population-calibrated fixed-ratio Fisher diagnostic.

# For a state-independent direct Gaussian mean with covariance sigma^2 I,
# score vectors have population covariance F_pop = sigma^{-2} I. Scaling by
# sigma^{-2} does not change condition number, isotropy deviation, or alignment.
# This script keeps d/n_F = 0.05 exactly by using n_F = 20d and reports empirical
# Wishart spread against the analytical population values and the
# Marchenko-Pastur edge-ratio reference.
#
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

SEED = 20260728
REPLICATIONS = 200
SAMPLE_RATIO = 20
DIMENSIONS = (15, 30, 60, 90)
RIDGE = 1e-4
SIGMA = float(np.exp(-0.5))
OUTPUT_DIR = Path(os.environ.get("LPG_RESULT_DIR", Path(__file__).resolve().parents[1] / "results")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "fixed_ratio_fisher.csv"


def isotropy_deviation(matrix: np.ndarray) -> float:
    d = matrix.shape[0]
    mean_eigenvalue = float(np.trace(matrix) / d)
    return float(
        np.linalg.norm(matrix - mean_eigenvalue * np.eye(d), ord="fro")
        / np.linalg.norm(matrix, ord="fro")
    )


def alignment(gradient: np.ndarray, fisher: np.ndarray) -> float:
    direction = np.linalg.solve(
        fisher + RIDGE * np.eye(fisher.shape[0]), gradient
    )
    return float(
        gradient @ direction
        / (np.linalg.norm(gradient) * np.linalg.norm(direction))
    )


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    population_scale = SIGMA**-2
    gamma = 1.0 / SAMPLE_RATIO
    mp_condition = ((1.0 + np.sqrt(gamma)) / (1.0 - np.sqrt(gamma))) ** 2

    rows: list[dict[str, float | int]] = []
    for dimension in DIMENSIONS:
        sample_count = SAMPLE_RATIO * dimension
        condition_numbers: list[float] = []
        deviations: list[float] = []
        alignments: list[float] = []

        for _ in range(REPLICATIONS):
            scores = rng.normal(size=(sample_count, dimension)) * np.sqrt(
                population_scale
            )
            fisher = scores.T @ scores / sample_count
            eigenvalues = np.linalg.eigvalsh(fisher)
            condition_numbers.append(float(eigenvalues[-1] / eigenvalues[0]))
            deviations.append(isotropy_deviation(fisher))
            gradient = rng.normal(size=dimension)
            alignments.append(alignment(gradient, fisher))

        kappa_mean, kappa_sd = summarize(condition_numbers)
        epsilon_mean, epsilon_sd = summarize(deviations)
        alignment_mean, alignment_sd = summarize(alignments)
        rows.append(
            {
                "dimension": dimension,
                "sample_count": sample_count,
                "population_kappa": 1.0,
                "population_epsilon": 0.0,
                "population_alignment": 1.0,
                "empirical_kappa_mean": kappa_mean,
                "empirical_kappa_sd": kappa_sd,
                "empirical_epsilon_mean": epsilon_mean,
                "empirical_epsilon_sd": epsilon_sd,
                "empirical_alignment_mean": alignment_mean,
                "empirical_alignment_sd": alignment_sd,
                "mp_condition_reference": float(mp_condition),
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"seed={SEED}, replications={REPLICATIONS}, d/n={gamma:.6f}")
    print(f"Marchenko-Pastur covariance condition reference={mp_condition:.10f}")
    for row in rows:
        print(
            f"d={row['dimension']:>3}, n={row['sample_count']:>4}, "
            f"kappa={row['empirical_kappa_mean']:.3f}+/-{row['empirical_kappa_sd']:.3f}, "
            f"epsilon={row['empirical_epsilon_mean']:.3f}+/-{row['empirical_epsilon_sd']:.3f}, "
            f"alignment={row['empirical_alignment_mean']:.3f}+/-{row['empirical_alignment_sd']:.3f}"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
