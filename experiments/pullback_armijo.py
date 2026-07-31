#!/usr/bin/env python3
# Reproducible multi-block Lie-pullback Armijo diagnostic.

# The intrinsic objective is defined on so(3)^K:

#     J(theta) = sum_j w_j <Q_star_j, exp(theta_j)>_F,

# with a global Frobenius-ball constraint.  Targets and the initial point are
# misaligned and generated from fixed seeds.  An ambient extension adds

#     <S_j(theta_j), Sym(Theta_j)>_F,
#     S_j(theta_j) = S0_j + alpha theta_j^2,

# which vanishes on so(3)^K but produces a normal gradient that varies along the
# feasible trajectory.  The script verifies:

# * nonzero, trajectory-varying normal ambient components;
# * noncommutativity of ambient clipping and algebra projection;
# * an active global radius projection;
# * genuine projected-Armijo backtracking over multiple iterations;
# * monotone objective increase and decay of the projected-gradient mapping;
# * the slack between the analytical L <= 3/2 single-block certificate and a
#   dense numerical Hessian search on the radius-0.75 calibration ball.
#
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import differential_evolution

SEED = 20260728
NORMAL_SEED = 20260729
K = 4
WEIGHTS = np.full(K, 0.5)
RADIUS = 4.0
INITIAL_NORM = 1.2
TRIAL_STEP = 16.0
BACKTRACK = 0.5
ARMIJO_C = 0.2
ITERATIONS = 8
NORMAL_ALPHA = 0.35
CALIBRATION_RADIUS = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path(os.environ.get("LPG_RESULT_DIR", PROJECT_ROOT / "results")).resolve()
FIGURE_DIR = Path(os.environ.get("LPG_FIGURE_DIR", PROJECT_ROOT / "figures")).resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULT_DIR / "pullback_armijo.csv"
PNG_PATH = FIGURE_DIR / "pullback_armijo.png"
SUMMARY_PATH = RESULT_DIR / "pullback_armijo_summary.txt"


def so3_basis() -> np.ndarray:
    """Frobenius-orthonormal basis E_i = hat(e_i)/sqrt(2)."""
    elements: list[np.ndarray] = []
    for i, j in ((2, 1), (0, 2), (1, 0)):
        matrix = np.zeros((3, 3), dtype=float)
        matrix[i, j] = 1.0 / np.sqrt(2.0)
        matrix[j, i] = -1.0 / np.sqrt(2.0)
        elements.append(matrix)
    return np.stack(elements)


BASIS = so3_basis()


def to_matrix(x: np.ndarray) -> np.ndarray:
    return np.tensordot(x, BASIS, axes=(0, 0))


def to_coordinates(matrix: np.ndarray) -> np.ndarray:
    return np.array([np.sum(matrix * element) for element in BASIS])


def skew(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix - matrix.T)


def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def project_ball(array: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(array))
    return array.copy() if norm <= radius else array * (radius / norm)


def objective(x: np.ndarray, targets: np.ndarray) -> float:
    return float(
        sum(
            WEIGHTS[j] * np.sum(targets[j] * expm(to_matrix(x[j])))
            for j in range(K)
        )
    )


def intrinsic_gradient(x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(x)
    for j in range(K):
        theta = to_matrix(x[j])
        for ell, basis_element in enumerate(BASIS):
            derivative = expm_frechet(
                theta, basis_element, compute_expm=False
            )
            gradient[j, ell] = WEIGHTS[j] * np.sum(
                targets[j] * derivative
            )
    return gradient


def normal_gradient_blocks(
    x: np.ndarray, normal_offsets: np.ndarray
) -> np.ndarray:
    blocks = []
    for j in range(K):
        theta = to_matrix(x[j])
        candidate = normal_offsets[j] + NORMAL_ALPHA * (theta @ theta)
        blocks.append(sym(candidate))
    return np.stack(blocks)


def ambient_gradient_blocks(
    x: np.ndarray, targets: np.ndarray, normal_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent_coordinates = intrinsic_gradient(x, targets)
    tangent_blocks = np.stack([to_matrix(row) for row in tangent_coordinates])
    normal_blocks = normal_gradient_blocks(x, normal_offsets)
    return tangent_coordinates, tangent_blocks + normal_blocks, normal_blocks


def projected_mapping(
    x: np.ndarray, gradient: np.ndarray, step: float = 1.0
) -> np.ndarray:
    return (project_ball(x + step * gradient, RADIUS) - x) / step


def ordered_and_reversed_first_trial(
    x: np.ndarray, ambient_gradient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta_blocks = np.stack([to_matrix(row) for row in x])
    ambient_trial = theta_blocks + TRIAL_STEP * ambient_gradient

    algebra_trial = np.stack([skew(block) for block in ambient_trial])
    ordered = project_ball(algebra_trial, RADIUS)

    clipped_ambient = project_ball(ambient_trial, RADIUS)
    reversed_projection = np.stack([skew(block) for block in clipped_ambient])
    return ordered, reversed_projection


def make_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    target_coordinates = []
    targets = []
    for _ in range(K):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        coordinate_norm = rng.uniform(1.2, 2.0)
        coordinates = coordinate_norm * axis
        target_coordinates.append(coordinates)
        targets.append(expm(to_matrix(coordinates)))

    x0 = rng.normal(size=(K, 3))
    x0 = project_ball(x0, INITIAL_NORM)

    normal_rng = np.random.default_rng(NORMAL_SEED)
    offsets = []
    for _ in range(K):
        raw = normal_rng.normal(size=(3, 3))
        offsets.append(sym(raw))
    return np.stack(targets), x0, np.stack(offsets)


def run_armijo(
    targets: np.ndarray, x0: np.ndarray, normal_offsets: np.ndarray
) -> tuple[list[dict[str, float | int | bool]], np.ndarray, float]:
    x = x0.copy()
    rows: list[dict[str, float | int | bool]] = []
    previous_objective = objective(x, targets)

    for iteration in range(ITERATIONS):
        gradient, ambient_gradient, normal_blocks = ambient_gradient_blocks(
            x, targets, normal_offsets
        )
        current_objective = objective(x, targets)
        mapping_norm = float(np.linalg.norm(projected_mapping(x, gradient)))
        normal_norm = float(np.linalg.norm(normal_blocks))

        step = TRIAL_STEP
        reductions = 0
        while True:
            unprojected = x + step * gradient
            candidate = project_ball(unprojected, RADIUS)
            displacement = candidate - x
            rhs = current_objective + (ARMIJO_C / step) * float(
                np.sum(displacement * displacement)
            )
            if objective(candidate, targets) + 1e-13 >= rhs:
                break
            step *= BACKTRACK
            reductions += 1
            if reductions > 80:
                raise RuntimeError("Armijo backtracking failed to terminate")

        candidate_objective = objective(candidate, targets)
        if candidate_objective + 1e-12 < current_objective:
            raise AssertionError("Accepted objective is not monotone")

        rows.append(
            {
                "iteration": iteration,
                "objective": current_objective,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "mapping_norm": mapping_norm,
                "accepted_step": step,
                "backtracking_reductions": reductions,
                "parameter_norm": float(np.linalg.norm(x)),
                "unprojected_trial_norm": float(np.linalg.norm(unprojected)),
                "radius_projection_active": bool(
                    np.linalg.norm(unprojected) > RADIUS + 1e-12
                ),
                "normal_component_norm": normal_norm,
            }
        )
        x = candidate
        previous_objective = candidate_objective

    final_gradient = intrinsic_gradient(x, targets)
    final_mapping = float(np.linalg.norm(projected_mapping(x, final_gradient)))
    return rows, x, final_mapping


def single_block_objective(x: np.ndarray) -> float:
    q_star = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return float(np.sum(q_star * expm(to_matrix(x))))


def single_block_gradient(x: np.ndarray) -> np.ndarray:
    q_star = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    gradient = np.zeros(3)
    theta = to_matrix(x)
    for ell, basis_element in enumerate(BASIS):
        derivative = expm_frechet(theta, basis_element, compute_expm=False)
        gradient[ell] = np.sum(q_star * derivative)
    return gradient


def numerical_hessian(x: np.ndarray, step: float = 2e-5) -> np.ndarray:
    hessian = np.empty((3, 3), dtype=float)
    for column in range(3):
        direction = np.zeros(3)
        direction[column] = step
        hessian[:, column] = (
            single_block_gradient(x + direction)
            - single_block_gradient(x - direction)
        ) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def estimate_single_block_lipschitz() -> tuple[float, np.ndarray]:
    """Numerically maximize ||H(x)||_2 over ||x|| <= 0.75.

    This is an empirical calibration, not a mathematical upper bound.
    """

    def negative_norm(raw: np.ndarray) -> float:
        x = np.asarray(raw, dtype=float)
        norm = np.linalg.norm(x)
        if norm > CALIBRATION_RADIUS:
            x = x * (CALIBRATION_RADIUS / norm)
        return -float(np.linalg.norm(numerical_hessian(x), ord=2))

    result = differential_evolution(
        negative_norm,
        bounds=[(-CALIBRATION_RADIUS, CALIBRATION_RADIUS)] * 3,
        seed=SEED,
        popsize=18,
        maxiter=80,
        tol=1e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    maximizer = np.asarray(result.x, dtype=float)
    norm = np.linalg.norm(maximizer)
    if norm > CALIBRATION_RADIUS:
        maximizer *= CALIBRATION_RADIUS / norm
    estimate = float(np.linalg.norm(numerical_hessian(maximizer), ord=2))
    return estimate, maximizer


def write_outputs(
    rows: list[dict[str, float | int | bool]],
    targets: np.ndarray,
    x0: np.ndarray,
    normal_offsets: np.ndarray,
    final_x: np.ndarray,
    final_mapping: float,
    lipschitz_estimate: float,
    lipschitz_point: np.ndarray,
) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    iterations = np.array([int(row["iteration"]) for row in rows])
    objectives = np.array([float(row["objective"]) for row in rows])
    mappings = np.array([float(row["mapping_norm"]) for row in rows])
    steps = np.array([float(row["accepted_step"]) for row in rows])
    normal_norms = np.array(
        [float(row["normal_component_norm"]) for row in rows]
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(iterations, objectives, marker="o")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("Projected Armijo ascent")

    axes[1].semilogy(iterations, np.maximum(mappings, 1e-16), marker="o")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mapping norm")
    axes[1].set_title("Stationarity diagnostic")

    axes[2].plot(iterations, steps, marker="o", label="accepted step")
    axes[2].plot(iterations, normal_norms, marker="s", label="normal norm")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Backtracking and ambient normal")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(figure)

    _, ambient0, normals0 = ambient_gradient_blocks(x0, targets, normal_offsets)
    ordered, reversed_projection = ordered_and_reversed_first_trial(x0, ambient0)
    composition_difference = float(np.linalg.norm(ordered - reversed_projection))

    final_objective = objective(final_x, targets)
    clipped_count = sum(bool(row["radius_projection_active"]) for row in rows)
    reductions = [int(row["backtracking_reductions"]) for row in rows]
    normal_min = min(float(row["normal_component_norm"]) for row in rows)
    normal_max = max(float(row["normal_component_norm"]) for row in rows)
    certificate = 1.5
    slack_ratio = certificate / lipschitz_estimate

    summary = f"""seed={SEED}\nnormal_seed={NORMAL_SEED}\nK={K}\nweights={WEIGHTS.tolist()}\nradius={RADIUS}\ntrial_step={TRIAL_STEP}\narmijo_c={ARMIJO_C}\nbacktrack={BACKTRACK}\niterations={ITERATIONS}\ninitial_objective={objective(x0, targets):.12f}\nfinal_objective={final_objective:.12f}\ninitial_mapping_norm={float(rows[0]['mapping_norm']):.12e}\nfinal_mapping_norm={final_mapping:.12e}\nradius_projection_active_count={clipped_count}\nbacktracking_reductions={reductions}\nnormal_component_norm_min={normal_min:.12f}\nnormal_component_norm_max={normal_max:.12f}\nfirst_trial_ordered_norm={np.linalg.norm(ordered):.12f}\nfirst_trial_reverse_norm={np.linalg.norm(reversed_projection):.12f}\nfirst_trial_composition_difference={composition_difference:.12f}\nsingle_block_certified_L={certificate:.12f}\nsingle_block_numerical_Hessian_sup_estimate={lipschitz_estimate:.12f}\ncertificate_to_estimate_ratio={slack_ratio:.12f}\nnumerical_maximizer={lipschitz_point.tolist()}\n"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    assert objective(final_x, targets) > objective(x0, targets)
    assert final_mapping < 1e-6
    assert clipped_count >= 1
    assert max(reductions) >= 1
    assert normal_max - normal_min > 1e-3
    assert composition_difference > 1e-3
    assert 0.63 < lipschitz_estimate < 0.68


def main() -> None:
    targets, x0, normal_offsets = make_problem()
    rows, final_x, final_mapping = run_armijo(targets, x0, normal_offsets)
    lipschitz_estimate, lipschitz_point = estimate_single_block_lipschitz()
    write_outputs(
        rows,
        targets,
        x0,
        normal_offsets,
        final_x,
        final_mapping,
        lipschitz_estimate,
        lipschitz_point,
    )
    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(f"wrote {CSV_PATH}")
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
