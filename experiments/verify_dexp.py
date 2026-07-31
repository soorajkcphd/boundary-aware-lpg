#!/usr/bin/env python3
# Numerically verify the exact dexp convention used in the manuscript.

# The manuscript defines
#     dexp^L_theta = integral_0^1 exp(-s ad_theta) ds
# and claims
#     D exp_theta[v] = exp(theta) dexp^L_theta(v),
#     grad(J~ o exp)(theta) = (dexp^L_theta)^* u(exp(theta)).

# This script checks both identities in a Frobenius-orthonormal basis of so(3)
# against SciPy's matrix-exponential Frechet derivative.
#
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm, expm_frechet

SEED = 20260728
TRIALS = 100


def basis_so3() -> np.ndarray:
    out = []
    for i, j in ((2, 1), (0, 2), (1, 0)):
        e = np.zeros((3, 3))
        e[i, j] = 1.0 / math.sqrt(2.0)
        e[j, i] = -1.0 / math.sqrt(2.0)
        out.append(e)
    return np.stack(out)


BASIS = basis_so3()


def mat(x: np.ndarray) -> np.ndarray:
    return np.tensordot(x, BASIS, axes=(0, 0))


def coords(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x * e) for e in BASIS])


def project_so3(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x - x.T)


def ad_matrix(theta: np.ndarray) -> np.ndarray:
    matrix = np.empty((3, 3))
    for j, e_j in enumerate(BASIS):
        bracket = theta @ e_j - e_j @ theta
        matrix[:, j] = coords(bracket)
    return matrix


def dexp_left_matrix(theta: np.ndarray, terms: int = 40) -> np.ndarray:
    a = ad_matrix(theta)
    result = np.eye(3)
    power = np.eye(3)
    factorial = 1.0
    # sum_{k>=0} (-A)^k/(k+1)!
    for k in range(1, terms):
        power = power @ (-a)
        factorial *= (k + 1)
        result += power / factorial
    return result


def main() -> None:
    rng = np.random.default_rng(SEED)
    q_star = expm(mat(np.array([0.7, -0.4, 1.1])))
    max_derivative_error = 0.0
    max_gradient_error = 0.0
    max_compact_ratio = 0.0

    for _ in range(TRIALS):
        x = rng.normal(size=3)
        x *= rng.uniform(0.0, 0.75) / max(np.linalg.norm(x), 1e-15)
        v = rng.normal(size=3)
        theta = mat(x)
        direction = mat(v)

        frechet = expm_frechet(theta, direction, compute_expm=False)
        t = dexp_left_matrix(theta)
        predicted = expm(theta) @ mat(t @ v)
        derivative_error = np.linalg.norm(frechet - predicted, ord="fro")
        max_derivative_error = max(max_derivative_error, derivative_error)

        q = expm(theta)
        u = coords(project_so3(q.T @ q_star))
        predicted_gradient = t.T @ u
        direct_gradient = np.array(
            [
                np.sum(
                    q_star
                    * expm_frechet(theta, element, compute_expm=False)
                )
                for element in BASIS
            ]
        )
        gradient_error = np.linalg.norm(predicted_gradient - direct_gradient)
        max_gradient_error = max(max_gradient_error, gradient_error)

        y = rng.normal(size=3)
        y *= rng.uniform(0.0, 0.75) / max(np.linalg.norm(y), 1e-15)
        theta_y = mat(y)
        q_y = expm(theta_y)
        u_y = coords(project_so3(q_y.T @ q_star))
        t_y = dexp_left_matrix(theta_y)
        grad_y = t_y.T @ u_y
        denominator = np.linalg.norm(x - y)
        if denominator > 1e-12:
            max_compact_ratio = max(
                max_compact_ratio,
                np.linalg.norm(predicted_gradient - grad_y) / denominator,
            )

    print(f"max Dexp convention error = {max_derivative_error:.3e}")
    print(f"max pullback-gradient error = {max_gradient_error:.3e}")
    print(f"max sampled gradient-Lipschitz ratio = {max_compact_ratio:.6f}")
    print("analytical compact certificate = 1.500000")

    assert max_derivative_error < 2e-12
    assert max_gradient_error < 2e-12
    assert max_compact_ratio < 1.5


if __name__ == "__main__":
    main()
