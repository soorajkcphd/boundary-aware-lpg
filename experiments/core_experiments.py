#!/usr/bin/env python3
# Generate the retained core diagnostic figures and numerical tables.

# The module implements the controlled SO(3)^K, SE(3), constrained-quadratic,
# Fisher-alignment, anisotropy, and joint-count experiments reported in the
# manuscript. All random seeds and protocol constants are fixed below.
#
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import ttest_1samp

SEED = 42
N_SEEDS = 5
BASE_STD = float(np.exp(-0.5))
FISHER_SAMPLES = 500
FISHER_RIDGE = 1e-4

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = Path(os.environ.get("LPG_FIGURE_DIR", ROOT / "figures")).resolve()
RESULT_DIR = Path(os.environ.get("LPG_RESULT_DIR", ROOT / "results")).resolve()
TABLE_DIR = Path(os.environ.get("LPG_TABLE_DIR", ROOT / "tables")).resolve()
for directory in (FIGURE_DIR, RESULT_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> None:
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {name}")
    path = TABLE_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


class SO3:
    @staticmethod
    def hat(w: np.ndarray) -> np.ndarray:
        return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])

    @staticmethod
    def vee(x: np.ndarray) -> np.ndarray:
        return np.array([x[2, 1], x[0, 2], x[1, 0]])

    @staticmethod
    def exp(x: np.ndarray) -> np.ndarray:
        w = SO3.vee(x)
        theta = float(np.linalg.norm(w))
        if theta < 1e-10:
            return np.eye(3) + x
        k = x / theta
        return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)

    @staticmethod
    def project(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x - x.T)


class SE3:
    @staticmethod
    def hat(xi: np.ndarray) -> np.ndarray:
        x = np.zeros((4, 4))
        x[:3, :3] = SO3.hat(xi[:3])
        x[:3, 3] = xi[3:]
        return x

    @staticmethod
    def vee(x: np.ndarray) -> np.ndarray:
        return np.concatenate([SO3.vee(x[:3, :3]), x[:3, 3]])

    @staticmethod
    def exp(x: np.ndarray) -> np.ndarray:
        w = SO3.vee(x[:3, :3])
        v = x[:3, 3]
        theta = float(np.linalg.norm(w))
        t = np.eye(4)
        if theta < 1e-10:
            t[:3, :3] = np.eye(3) + x[:3, :3]
            t[:3, 3] = v
            return t
        k = x[:3, :3] / theta
        r = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
        v_matrix = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta) * k
            + ((theta - np.sin(theta)) / theta) * (k @ k)
        )
        t[:3, :3] = r
        t[:3, 3] = v_matrix @ v
        return t

    @staticmethod
    def log(t: np.ndarray) -> np.ndarray:
        r = t[:3, :3]
        translation = t[:3, 3]
        cosine = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(cosine))
        x = np.zeros((4, 4))
        if theta < 1e-6:
            x[:3, :3] = 0.5 * (r - r.T)
            x[:3, 3] = translation
            return x
        omega = theta / (2.0 * np.sin(theta)) * (r - r.T)
        x[:3, :3] = omega
        coefficient = (1.0 / theta**2) * (
            1.0 - theta * np.cos(theta / 2.0) / (2.0 * np.sin(theta / 2.0))
        )
        v_inverse = np.eye(3) - 0.5 * omega + coefficient * (omega @ omega)
        x[:3, 3] = v_inverse @ translation
        return x


class SO3ProductEnvironment:
    def __init__(self, joints: int = 10, horizon: int = 30):
        self.joints = joints
        self.horizon = horizon
        self.gamma = 0.99
        self.state: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.step_index = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = [SO3.exp(SO3.hat(np.random.randn(3) * 0.3)) for _ in range(self.joints)]
        self.target = [SO3.exp(SO3.hat(np.random.randn(3) * 0.8)) for _ in range(self.joints)]
        self.step_index = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        values: list[float] = []
        for state, target in zip(self.state, self.target):
            error = target.T @ state
            cosine = np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0)
            angle = float(np.arccos(cosine))
            if angle < 1e-6:
                vector = np.zeros(3)
            else:
                vector = angle / (2.0 * np.sin(angle) + 1e-8) * SO3.vee(error - error.T)
            values.extend(vector)
        return np.asarray(values)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        action = np.clip(action, -2.0, 2.0)
        for joint in range(self.joints):
            vector = 0.1 * action[3 * joint : 3 * (joint + 1)]
            updated = self.state[joint] @ SO3.exp(SO3.hat(vector))
            u, _, vh = la.svd(updated)
            self.state[joint] = u @ vh
        reward = 0.0
        for state, target in zip(self.state, self.target):
            error = target.T @ state
            cosine = np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0)
            reward -= float(np.arccos(cosine) ** 2)
        self.step_index += 1
        return self.observation(), reward, self.step_index >= self.horizon

    @property
    def observation_dimension(self) -> int:
        return 3 * self.joints

    @property
    def action_dimension(self) -> int:
        return 3 * self.joints


class SE3Environment:
    def __init__(self, horizon: int = 30, increment_radius: float = 2.0, rotation_scale: float = 0.7, translation_scale: float = 1.0):
        self.horizon = horizon
        self.gamma = 0.99
        self.increment_radius = increment_radius
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale
        self.state = np.eye(4)
        self.target = np.eye(4)
        self.step_index = 0
        self.reset()

    def reset(self) -> np.ndarray:
        initial = np.zeros(6)
        initial[:3] = np.random.randn(3) * 0.2
        initial[3:] = np.random.randn(3) * 0.3
        target = np.zeros(6)
        target[:3] = np.random.randn(3) * self.rotation_scale
        target[3:] = np.random.randn(3) * self.translation_scale
        self.state = SE3.exp(SE3.hat(initial))
        self.target = SE3.exp(SE3.hat(target))
        self.step_index = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        error = np.linalg.solve(self.state, np.eye(4)) @ self.target
        return SE3.vee(SE3.log(error))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        xi = 0.1 * np.clip(action, -2.0, 2.0)
        norm = float(np.linalg.norm(xi))
        if norm > self.increment_radius:
            xi *= self.increment_radius / norm
        self.state = self.state @ SE3.exp(SE3.hat(xi))
        error = np.linalg.solve(self.state, np.eye(4)) @ self.target
        reward = -float(np.linalg.norm(SE3.log(error), ord="fro") ** 2)
        self.step_index += 1
        return self.observation(), reward, self.step_index >= self.horizon

    @property
    def observation_dimension(self) -> int:
        return 6

    @property
    def action_dimension(self) -> int:
        return 6


class ConstantMeanPolicy:
    def __init__(self, dimension: int, sigma: np.ndarray | None = None):
        self.theta = np.zeros(dimension)
        self.sigma = BASE_STD * np.ones(dimension) if sigma is None else np.asarray(sigma, dtype=float)

    @property
    def parameter_dimension(self) -> int:
        return self.theta.size

    def sample(self, state: np.ndarray) -> np.ndarray:
        del state
        return self.theta + self.sigma * np.random.randn(self.theta.size)

    def score(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        del state
        return (action - self.theta) / self.sigma**2


class DiagonalFeaturePolicy:
    def __init__(self, dimension: int):
        self.theta = np.zeros(dimension)
        self.sigma = BASE_STD * np.ones(dimension)

    @property
    def parameter_dimension(self) -> int:
        return self.theta.size

    def mean(self, state: np.ndarray) -> np.ndarray:
        return self.theta * state[: self.theta.size]

    def sample(self, state: np.ndarray) -> np.ndarray:
        return self.mean(state) + self.sigma * np.random.randn(self.theta.size)

    def score(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return (action - self.mean(state)) / self.sigma**2 * state[: self.theta.size]


class RedundantPolicy:
    def __init__(self, action_dimension: int, redundancy: int = 15, seed: int = 999):
        self.theta = np.zeros(action_dimension * redundancy)
        rng = np.random.RandomState(seed)
        matrix = rng.randn(action_dimension, self.theta.size)
        matrix /= np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8
        matrix *= np.linspace(0.1, 5.0, matrix.shape[1])
        self.matrix = np.diag(np.linspace(0.3, 3.0, action_dimension)) @ matrix
        self.sigma = BASE_STD * np.ones(action_dimension)

    def mean(self) -> np.ndarray:
        return self.matrix @ self.theta

    def sample(self, state: np.ndarray) -> np.ndarray:
        del state
        return self.mean() + self.sigma * np.random.randn(self.sigma.size)

    def score(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        del state
        return self.matrix.T @ ((action - self.mean()) / self.sigma**2)


def policy_iteration(policy: Any, environment: Any, episodes: int = 4) -> tuple[np.ndarray, list[float]]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    returns: list[float] = []
    episode_totals: list[float] = []
    for _ in range(episodes):
        state = environment.reset()
        done = False
        episode_states: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        rewards: list[float] = []
        while not done:
            action = policy.sample(state)
            next_state, reward, done = environment.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            rewards.append(reward)
            state = next_state
        running = 0.0
        episode_returns: list[float] = []
        for reward in reversed(rewards):
            running = reward + environment.gamma * running
            episode_returns.append(running)
        episode_returns.reverse()
        states.extend(episode_states)
        actions.extend(episode_actions)
        returns.extend(episode_returns)
        episode_totals.append(float(sum(rewards)))
    advantages = np.asarray(returns)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    gradient = np.zeros_like(policy.theta)
    for state, action, advantage in zip(states, actions, advantages):
        gradient += advantage * policy.score(state, action)
    gradient /= max(len(states), 1)
    return gradient, episode_totals


def train_policy(policy: Any, environment: Any, iterations: int, learning_rate: float, episodes: int = 8) -> np.ndarray:
    curve = []
    for _ in range(iterations):
        gradient, returns = policy_iteration(policy, environment, episodes)
        policy.theta += learning_rate * gradient
        curve.append(float(np.mean(returns)))
    return np.asarray(curve)


def estimate_fisher(policy: Any, environment: Any, samples: int = FISHER_SAMPLES) -> np.ndarray:
    # For a state-independent Gaussian mean, the score distribution is known
    # exactly and does not require environment resets. Sampling it in one batch
    # is mathematically identical and substantially reduces runtime.
    if isinstance(policy, ConstantMeanPolicy):
        scores = np.random.randn(samples, policy.parameter_dimension) / policy.sigma
        return scores.T @ scores / samples
    if isinstance(policy, DiagonalFeaturePolicy):
        return diagonal_feature_fisher(policy, environment, samples)
    fisher = np.zeros((policy.parameter_dimension, policy.parameter_dimension))
    for _ in range(samples):
        state = environment.reset()
        action = policy.sample(state)
        score = policy.score(state, action)
        fisher += np.outer(score, score)
    return fisher / samples


def fisher_metrics(fisher: np.ndarray) -> dict[str, float]:
    dimension = fisher.shape[0]
    eigenvalues = np.linalg.eigvalsh(fisher)
    positive = eigenvalues[eigenvalues > 1e-10]
    condition = float(positive[-1] / positive[0])
    mean_eigenvalue = float(np.trace(fisher) / dimension)
    epsilon = float(np.linalg.norm(fisher - mean_eigenvalue * np.eye(dimension), ord="fro") / np.linalg.norm(fisher, ord="fro"))
    effective_rank = float(positive.sum() ** 2 / np.sum(positive**2))
    return {"condition": condition, "epsilon": epsilon, "effective_rank": effective_rank}


def fisher_alignment(policy: Any, environment: Any, gradient: np.ndarray, samples: int = FISHER_SAMPLES) -> tuple[dict[str, float], float]:
    fisher = estimate_fisher(policy, environment, samples)
    metrics = fisher_metrics(fisher)
    regularized = fisher + FISHER_RIDGE * np.eye(fisher.shape[0])
    metrics["regularized_condition"] = fisher_metrics(regularized)["condition"]
    natural = la.solve(regularized, gradient, assume_a="pos")
    alignment = float(gradient @ natural / (np.linalg.norm(gradient) * np.linalg.norm(natural) + 1e-10))
    return metrics, alignment


def constant_mean_iteration(
    theta: np.ndarray,
    sigma: np.ndarray,
    joints: int,
    episodes: int = 4,
    horizon: int = 30,
) -> tuple[np.ndarray, list[float]]:
    """Vectorized REINFORCE batch with the scalar generator's RNG order."""
    action_dimension = 3 * joints
    # Each original episode consumes reset draws first (state, target), followed
    # by one action-noise vector per time step. Drawing one row per episode
    # preserves that exact pseudorandom sequence while vectorizing the algebra.
    draws = np.random.randn(episodes, 6 * joints + horizon * action_dimension)
    state_vectors = draws[:, : 3 * joints].reshape(episodes, joints, 3) * 0.3
    target_vectors = draws[:, 3 * joints : 6 * joints].reshape(episodes, joints, 3) * 0.8
    standardized_noise = draws[:, 6 * joints :].reshape(episodes, horizon, action_dimension)
    state = batch_exp_so3(state_vectors)
    target = batch_exp_so3(target_vectors)
    residuals = standardized_noise * sigma[None, None, :]
    rewards = np.empty((episodes, horizon))
    for time_index in range(horizon):
        action = np.clip(theta[None, :] + residuals[:, time_index, :], -2.0, 2.0)
        increments = batch_exp_so3((0.1 * action).reshape(episodes, joints, 3))
        state = state @ increments
        error = np.swapaxes(target, -1, -2) @ state
        traces = np.trace(error, axis1=-2, axis2=-1)
        cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        rewards[:, time_index] = -np.sum(np.arccos(cosine) ** 2, axis=1)
    return_to_go = np.empty_like(rewards)
    running = np.zeros(episodes)
    for time_index in range(horizon - 1, -1, -1):
        running = rewards[:, time_index] + 0.99 * running
        return_to_go[:, time_index] = running
    advantages = return_to_go.reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    gradient = np.mean(
        advantages[:, None]
        * (residuals.reshape(-1, action_dimension) / (sigma[None, :] ** 2)),
        axis=0,
    )
    return gradient, rewards.sum(axis=1).tolist()


def constant_mean_fisher(
    sigma: np.ndarray,
    joints: int,
    samples: int = FISHER_SAMPLES,
) -> np.ndarray:
    # The scalar implementation resets SO(3)^K before every score draw. The
    # reset values do not enter a state-independent score, but they do advance
    # the seeded RNG. Retaining those draws reproduces the original sequence.
    draws = np.random.randn(samples, 6 * joints + sigma.size)
    standardized_noise = draws[:, 6 * joints :]
    scores = standardized_noise / sigma[None, :]
    return scores.T @ scores / samples


def constant_mean_metrics(
    sigma: np.ndarray,
    gradient: np.ndarray,
    joints: int,
    samples: int = FISHER_SAMPLES,
) -> tuple[dict[str, float], float]:
    fisher = constant_mean_fisher(sigma, joints, samples)
    metrics = fisher_metrics(fisher)
    regularized = fisher + FISHER_RIDGE * np.eye(fisher.shape[0])
    metrics["regularized_condition"] = fisher_metrics(regularized)["condition"]
    natural = la.solve(regularized, gradient, assume_a="pos")
    alignment = float(
        gradient @ natural
        / (np.linalg.norm(gradient) * np.linalg.norm(natural) + 1e-10)
    )
    return metrics, alignment


def run_fisher_alignment() -> dict[str, Any]:
    joints = 10
    iterations = 40
    alignments: list[float] = []
    conditions: list[float] = []
    regularized_conditions: list[float] = []
    epsilons: list[float] = []
    ranks: list[float] = []
    per_epsilon = np.zeros((N_SEEDS, iterations))
    per_condition = np.zeros((N_SEEDS, iterations))
    per_alignment = np.zeros((N_SEEDS, iterations))
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        np.random.randn(6 * joints)
        theta = np.zeros(3 * joints)
        sigma = BASE_STD * np.ones(3 * joints)
        for iteration in range(iterations):
            gradient, _ = constant_mean_iteration(theta, sigma, joints)
            metrics, alignment = constant_mean_metrics(sigma, gradient, joints)
            alignments.append(alignment)
            conditions.append(metrics["condition"])
            regularized_conditions.append(metrics["regularized_condition"])
            epsilons.append(metrics["epsilon"])
            ranks.append(metrics["effective_rank"])
            per_epsilon[seed, iteration] = metrics["epsilon"]
            per_condition[seed, iteration] = metrics["condition"]
            per_alignment[seed, iteration] = alignment
            theta += 0.05 * gradient
    a = np.asarray(alignments)
    k = np.asarray(conditions)
    kr = np.asarray(regularized_conditions)
    e = np.asarray(epsilons)
    r = np.asarray(ranks)
    bound = float(2.0 * np.sqrt(kr.mean()) / (kr.mean() + 1.0))
    _, p_two_sided = ttest_1samp(a, 0.9)
    summary = {
        "measurements": int(a.size),
        "condition_mean": float(k.mean()),
        "condition_sd": float(k.std()),
        "epsilon_mean": float(e.mean()),
        "epsilon_sd": float(e.std()),
        "alignment_mean": float(a.mean()),
        "alignment_sd": float(a.std()),
        "alignment_min": float(a.min()),
        "alignment_max": float(a.max()),
        "effective_rank_mean": float(r.mean()),
        "effective_rank_sd": float(r.std()),
        "kantorovich_bound": bound,
        "one_sided_p_against_0_9": float(p_two_sided / 2.0),
    }
    plt.figure(figsize=(5, 4))
    plt.hist(a, bins=20, alpha=0.8)
    plt.axvline(a.mean(), linestyle="-", linewidth=2, label=f"mean={a.mean():.3f}")
    plt.axvline(bound, linestyle="--", linewidth=2, label=f"bound={bound:.3f}")
    plt.xlabel("cosine(natural gradient, ordinary gradient)")
    plt.ylabel("count")
    plt.title("Fisher-Metric Alignment")
    plt.legend()
    save_figure("fisher_alignment_histogram.png")
    figure, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)
    for axis, values, label, title in (
        (axes[0], per_epsilon, r"$\varepsilon_F$", "(a) Isotropy deviation"),
        (axes[1], per_condition, r"$\kappa$", "(b) Condition number"),
        (axes[2], per_alignment, "Alignment", "(c) Gradient alignment"),
    ):
        mean = values.mean(axis=0)
        sd = values.std(axis=0)
        x = np.arange(iterations)
        axis.plot(x, mean)
        axis.fill_between(x, mean - sd, mean + sd, alpha=0.2)
        axis.set_ylabel(label)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Iteration")
    figure.suptitle("Isotropy Tracking During Training", fontsize=12)
    save_figure("fisher_isotropy_tracking.png")
    write_csv("fisher_alignment.csv", [summary])
    return {"summary": summary, "alignments": a.tolist(), "conditions": k.tolist(), "epsilons": e.tolist()}


def population_geometry(sigma: np.ndarray) -> tuple[float, float]:
    fisher = np.diag(1.0 / sigma**2)
    values = np.linalg.eigvalsh(fisher)
    condition = float(values[-1] / values[0])
    mean_value = float(np.trace(fisher) / fisher.shape[0])
    epsilon = float(np.linalg.norm(fisher - mean_value * np.eye(fisher.shape[0]), ord="fro") / np.linalg.norm(fisher, ord="fro"))
    return condition, epsilon


def run_controlled_anisotropy() -> list[dict[str, Any]]:
    joints = 10
    dimension = 3 * joints
    iterations = 40
    axis_sigma = BASE_STD * np.ones(dimension)
    for joint in range(joints):
        axis_sigma[3 * joint + 2] *= np.sqrt(1.5)
    conditions = [
        ("Uniform", BASE_STD * np.ones(dimension)),
        ("Axis-biased", axis_sigma),
        ("Diagonal spread (kappa=5)", BASE_STD * np.sqrt(np.linspace(1.0, 5.0, dimension))),
        ("Diagonal spread (kappa=10)", BASE_STD * np.sqrt(np.linspace(1.0, 10.0, dimension))),
    ]
    rows: list[dict[str, Any]] = []
    for label, sigma in conditions:
        alignments: list[float] = []
        conditions_empirical: list[float] = []
        epsilons: list[float] = []
        final_returns: list[float] = []
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            np.random.randn(6 * joints)
            theta = np.zeros(dimension)
            episode_returns: list[float] = []
            for _ in range(iterations):
                gradient, returns = constant_mean_iteration(theta, sigma, joints)
                episode_returns.extend(returns)
                metrics, alignment = constant_mean_metrics(sigma, gradient, joints)
                alignments.append(alignment)
                conditions_empirical.append(metrics["condition"])
                epsilons.append(metrics["epsilon"])
                theta += 0.05 * gradient
            final_returns.append(float(np.mean(episode_returns[-40:])))
        population_condition, population_epsilon = population_geometry(sigma)
        rows.append({
            "condition": label,
            "population_condition": population_condition,
            "population_epsilon": population_epsilon,
            "empirical_condition": float(np.mean(conditions_empirical)),
            "empirical_epsilon": float(np.mean(epsilons)),
            "alignment": float(np.mean(alignments)),
            "final_return_mean": float(np.mean(final_returns)),
            "final_return_sd": float(np.std(final_returns, ddof=1)),
        })
    baseline = abs(float(rows[0]["final_return_mean"]))
    for row in rows:
        row["return_degradation_percent"] = 100.0 * (abs(float(row["final_return_mean"])) - baseline) / baseline
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    empirical_conditions = np.asarray([row["empirical_condition"] for row in rows])
    alignments = np.asarray([row["alignment"] for row in rows])
    epsilons = np.asarray([row["empirical_epsilon"] for row in rows])
    degradations = np.asarray([row["return_degradation_percent"] for row in rows])
    axes[0].plot(empirical_conditions, alignments, "o-")
    grid = np.linspace(empirical_conditions.min() * 0.9, empirical_conditions.max() * 1.1, 100)
    axes[0].plot(grid, 2.0 * np.sqrt(grid) / (grid + 1.0), "--", label="Kantorovich bound")
    axes[0].set(xlabel=r"Empirical $\kappa$", ylabel="Alignment", title="(a) Alignment vs. condition number")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epsilons, degradations, "s-")
    axes[1].set(xlabel=r"Empirical $\varepsilon_F$", ylabel="Return degradation (%)", title="(b) Within-sweep degradation")
    axes[1].grid(alpha=0.3)
    figure.suptitle("Controlled Anisotropy")
    save_figure("controlled_anisotropy.png")
    write_csv("controlled_anisotropy.csv", rows)
    return rows


def run_se3_diagnostics() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, environment_factory in (
        ("so(3)", lambda: SO3ProductEnvironment(joints=1)),
        ("se(3)", lambda: SE3Environment()),
    ):
        alignments: list[float] = []
        conditions: list[float] = []
        epsilons: list[float] = []
        dimension = 0
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            environment = environment_factory()
            dimension = environment.action_dimension
            policy = DiagonalFeaturePolicy(dimension)
            theta = policy.theta
            for _ in range(40):
                gradient, _ = diagonal_policy_iteration(theta, environment)
                policy.theta = theta
                metrics, alignment = fisher_alignment(policy, environment, gradient)
                alignments.append(alignment)
                conditions.append(metrics["condition"])
                epsilons.append(metrics["epsilon"])
                theta += 0.05 * gradient
        rows.append({
            "algebra": label,
            "dimension": dimension,
            "alignment_mean": float(np.mean(alignments)),
            "alignment_sd": float(np.std(alignments)),
            "condition_mean": float(np.mean(conditions)),
            "condition_sd": float(np.std(conditions)),
            "epsilon_mean": float(np.mean(epsilons)),
        })
    no_projection: list[list[float]] = []
    projection: list[list[float]] = []
    no_gradient: list[list[float]] = []
    projection_gradient: list[list[float]] = []
    learning_rate = 3.0
    radius = 2.0
    iterations = 150
    for seed in range(N_SEEDS):
        np.random.seed(200 + seed)
        environment = SE3Environment()
        policy = DiagonalFeaturePolicy(6)
        norms: list[float] = []
        gradients: list[float] = []
        for _ in range(iterations):
            gradient, _ = diagonal_policy_iteration(policy.theta, environment)
            gradients.append(float(np.linalg.norm(gradient)))
            policy.theta += learning_rate * gradient
            norms.append(float(np.linalg.norm(policy.theta)))
        no_projection.append(norms)
        no_gradient.append(gradients)
        np.random.seed(200 + seed)
        environment = SE3Environment()
        policy = DiagonalFeaturePolicy(6)
        norms = []
        gradients = []
        for _ in range(iterations):
            gradient, _ = diagonal_policy_iteration(policy.theta, environment)
            gradients.append(float(np.linalg.norm(gradient)))
            policy.theta += learning_rate * gradient
            norm = float(np.linalg.norm(policy.theta))
            if norm > radius:
                policy.theta *= radius / norm
            norms.append(float(np.linalg.norm(policy.theta)))
        projection.append(norms)
        projection_gradient.append(gradients)
    no_array = np.asarray(no_projection)
    projection_array = np.asarray(projection)
    no_gradient_array = np.asarray(no_gradient)
    projection_gradient_array = np.asarray(projection_gradient)
    x = np.arange(iterations)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for values, label, linestyle in ((no_array, "No parameter projection", "-"), (projection_array, "Parameter radius 2", "--")):
        mean = values.mean(axis=0)
        sd = values.std(axis=0)
        axes[0].plot(x, mean, linestyle=linestyle, label=label)
        axes[0].fill_between(x, np.maximum(mean - sd, 0), mean + sd, alpha=0.2)
    axes[0].axhline(radius, linestyle=":", label="Radius 2")
    axes[0].set(xlabel="Iteration", ylabel=r"$\Vert\theta\Vert_F$", title="(a) Parameter norm")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for values, label, linestyle in ((no_array, "No parameter projection", "-"), (projection_array, "Parameter radius 2", "--")):
        proxy = values.mean(axis=0) ** 2
        proxy /= proxy[0] + 1e-10
        axes[1].semilogy(x, proxy, linestyle=linestyle, label=label)
    axes[1].set(xlabel="Iteration", ylabel="Relative quadratic radius proxy", title="(b) Radius-growth proxy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    figure.suptitle("SE(3) Parameter-Radius Control")
    save_figure("se3_radius_control.png")
    radius_summary = {
        "final_parameter_norm_no_projection": float(no_array[:, -1].mean()),
        "final_parameter_norm_projection": float(projection_array[:, -1].mean()),
        "final_gradient_norm_no_projection": float(no_gradient_array[:, -1].mean()),
        "final_gradient_norm_projection": float(projection_gradient_array[:, -1].mean()),
        "seeds": N_SEEDS,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "radius": radius,
    }
    write_csv("se3_fisher.csv", rows)
    write_csv("se3_radius_control.csv", [radius_summary])
    return {"fisher": rows, "radius_control": radius_summary}


def project_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= radius else radius * vector / norm


def run_theory_aligned() -> dict[str, Any]:
    radius = 1.0
    step = 0.1
    b = np.array([2.0, 0.5, -0.2])
    theta = np.zeros(3)
    raw_norms: list[float] = []
    mapping_norms: list[float] = []
    for _ in range(30):
        gradient = b - theta
        projected = project_ball(theta + step * gradient, radius)
        mapping = (projected - theta) / step
        raw_norms.append(float(np.linalg.norm(gradient)))
        mapping_norms.append(float(np.linalg.norm(mapping)))
        theta = projected
    final_gradient = b - theta
    final_mapping = (project_ball(theta + step * final_gradient, radius) - theta) / step
    dimension = 10
    matrix = np.diag(np.linspace(1.0, 2.0, dimension))
    stochastic_step = 0.25
    stochastic_b = 3.0 * np.ones(dimension) / np.sqrt(dimension)
    sigma = 2.0
    horizons = np.array([100, 200, 400, 800, 1600], dtype=int)
    rows: list[dict[str, Any]] = []
    means: list[float] = []
    stds: list[float] = []
    for horizon in horizons:
        batch = int(np.ceil(np.sqrt(horizon)))
        values: list[float] = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            point = np.zeros(dimension)
            squared: list[float] = []
            for _ in range(int(horizon)):
                gradient = stochastic_b - matrix @ point
                mapping = (project_ball(point + stochastic_step * gradient, radius) - point) / stochastic_step
                squared.append(float(mapping @ mapping))
                noise = rng.normal(scale=sigma, size=(batch, dimension)).mean(axis=0)
                point = project_ball(point + stochastic_step * (gradient + noise), radius)
            values.append(float(np.mean(squared)))
        mean = float(np.mean(values))
        sd = float(np.std(values))
        means.append(mean)
        stds.append(sd)
        rows.append({"horizon": int(horizon), "batch_size": batch, "mapping_squared_mean": mean, "mapping_squared_sd": sd})
    slope = float(np.polyfit(np.log(horizons.astype(float)), np.log(np.asarray(means)), 1)[0])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(raw_norms, label=r"$\Vert\nabla J\Vert$")
    axes[0].plot(mapping_norms, label=r"$\Vert G_\eta\Vert$")
    axes[0].set(xlabel="Iteration", ylabel="Norm", title="(a) Boundary-active stationarity")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    means_array = np.asarray(means)
    stds_array = np.asarray(stds)
    axes[1].loglog(horizons, means_array, "o-", label=f"slope {slope:.3f}")
    axes[1].fill_between(horizons, np.maximum(means_array - stds_array, 1e-12), means_array + stds_array, alpha=0.2)
    axes[1].loglog(horizons, means_array[0] * (horizons / horizons[0]) ** (-0.5), "--", label=r"$T^{-1/2}$ reference")
    axes[1].set(xlabel="Horizon T", ylabel="Averaged squared mapping", title="(b) Mini-batch stochastic ascent")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    save_figure("theory_aligned_diagnostics.png")
    write_csv("theory_aligned_rate.csv", rows)
    return {
        "boundary_final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "boundary_final_mapping_norm": float(np.linalg.norm(final_mapping)),
        "slope": slope,
        "rate_rows": rows,
    }


def batch_hat(vectors: np.ndarray) -> np.ndarray:
    output = np.zeros(vectors.shape[:-1] + (3, 3), dtype=float)
    output[..., 0, 1] = -vectors[..., 2]
    output[..., 0, 2] = vectors[..., 1]
    output[..., 1, 0] = vectors[..., 2]
    output[..., 1, 2] = -vectors[..., 0]
    output[..., 2, 0] = -vectors[..., 1]
    output[..., 2, 1] = vectors[..., 0]
    return output


def batch_exp_so3(vectors: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(vectors, axis=-1)
    matrices = batch_hat(vectors)
    angle_squared = angles**2
    nonzero = angles > 1e-8
    coefficient_one = np.empty_like(angles)
    coefficient_two = np.empty_like(angles)
    np.divide(np.sin(angles), angles, out=coefficient_one, where=nonzero)
    np.divide(
        1.0 - np.cos(angles),
        angle_squared,
        out=coefficient_two,
        where=nonzero,
    )
    coefficient_one[~nonzero] = (
        1.0 - angle_squared[~nonzero] / 6.0
        + angle_squared[~nonzero] ** 2 / 120.0
    )
    coefficient_two[~nonzero] = (
        0.5 - angle_squared[~nonzero] / 24.0
        + angle_squared[~nonzero] ** 2 / 720.0
    )
    identity = np.broadcast_to(np.eye(3), matrices.shape)
    return (
        identity
        + coefficient_one[..., None, None] * matrices
        + coefficient_two[..., None, None] * (matrices @ matrices)
    )


def batch_so3_log_vectors(rotations: np.ndarray) -> np.ndarray:
    traces = np.trace(rotations, axis1=-2, axis2=-1)
    cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angles = np.arccos(cosine)
    skew_part = rotations - np.swapaxes(rotations, -1, -2)
    vee = np.stack(
        [skew_part[..., 2, 1], skew_part[..., 0, 2], skew_part[..., 1, 0]],
        axis=-1,
    )
    factors = np.where(
        angles > 1e-6,
        angles / (2.0 * np.sin(angles) + 1e-15),
        0.5,
    )
    return factors[..., None] * vee


def batch_exp_se3(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = coordinates[..., :3]
    translation_coordinates = coordinates[..., 3:]
    angles = np.linalg.norm(omega, axis=-1)
    omega_hat = batch_hat(omega)
    angle_squared = angles**2
    nonzero = angles > 1e-8
    coefficient_a = np.empty_like(angles)
    coefficient_b = np.empty_like(angles)
    np.divide(
        1.0 - np.cos(angles),
        angle_squared,
        out=coefficient_a,
        where=nonzero,
    )
    np.divide(
        angles - np.sin(angles),
        angles**3,
        out=coefficient_b,
        where=nonzero,
    )
    coefficient_a[~nonzero] = (
        0.5 - angle_squared[~nonzero] / 24.0
        + angle_squared[~nonzero] ** 2 / 720.0
    )
    coefficient_b[~nonzero] = (
        1.0 / 6.0 - angle_squared[~nonzero] / 120.0
        + angle_squared[~nonzero] ** 2 / 5040.0
    )
    identity = np.broadcast_to(np.eye(3), omega_hat.shape)
    rotation = batch_exp_so3(omega)
    v_matrix = (
        identity
        + coefficient_a[..., None, None] * omega_hat
        + coefficient_b[..., None, None] * (omega_hat @ omega_hat)
    )
    translation = (v_matrix @ translation_coordinates[..., None])[..., 0]
    return rotation, translation


def batch_se3_log_vectors(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    omega = batch_so3_log_vectors(rotation)
    omega_hat = batch_hat(omega)
    angles = np.linalg.norm(omega, axis=-1)
    coefficient = np.empty_like(angles)
    small = angles < 1e-6
    coefficient[small] = 1.0 / 12.0 + angles[small] ** 2 / 720.0
    large = ~small
    coefficient[large] = (
        1.0
        - angles[large] * np.cos(angles[large] / 2.0)
        / (2.0 * np.sin(angles[large] / 2.0))
    ) / (angles[large] ** 2)
    identity = np.broadcast_to(np.eye(3), omega_hat.shape)
    inverse_v = identity - 0.5 * omega_hat + coefficient[..., None, None] * (omega_hat @ omega_hat)
    vector = (inverse_v @ translation[..., None])[..., 0]
    return np.concatenate([omega, vector], axis=-1)


def sample_reset_observations(environment: Any, samples: int) -> np.ndarray:
    if isinstance(environment, SO3ProductEnvironment):
        joints = environment.joints
        state = batch_exp_so3(np.random.randn(samples, joints, 3) * 0.3)
        target = batch_exp_so3(np.random.randn(samples, joints, 3) * 0.8)
        error = np.swapaxes(target, -1, -2) @ state
        return batch_so3_log_vectors(error).reshape(samples, 3 * joints)
    if isinstance(environment, SE3Environment):
        initial = np.zeros((samples, 6))
        initial[:, :3] = np.random.randn(samples, 3) * 0.2
        initial[:, 3:] = np.random.randn(samples, 3) * 0.3
        target_coordinates = np.zeros((samples, 6))
        target_coordinates[:, :3] = np.random.randn(samples, 3) * environment.rotation_scale
        target_coordinates[:, 3:] = np.random.randn(samples, 3) * environment.translation_scale
        state_rotation, state_translation = batch_exp_se3(initial)
        target_rotation, target_translation = batch_exp_se3(target_coordinates)
        inverse_rotation = np.swapaxes(state_rotation, -1, -2)
        error_rotation = inverse_rotation @ target_rotation
        error_translation = (
            inverse_rotation
            @ (target_translation - state_translation)[..., None]
        )[..., 0]
        return batch_se3_log_vectors(error_rotation, error_translation)
    raise TypeError(f"unsupported environment type: {type(environment)!r}")


def so3_observation(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.swapaxes(target, -1, -2) @ state
    return batch_so3_log_vectors(error)


def se3_observation(
    state_rotation: np.ndarray,
    state_translation: np.ndarray,
    target_rotation: np.ndarray,
    target_translation: np.ndarray,
) -> np.ndarray:
    inverse_rotation = np.swapaxes(state_rotation, -1, -2)
    error_rotation = inverse_rotation @ target_rotation
    error_translation = (
        inverse_rotation
        @ (target_translation - state_translation)[..., None]
    )[..., 0]
    return batch_se3_log_vectors(error_rotation, error_translation)


def diagonal_feature_fisher(
    policy: DiagonalFeaturePolicy,
    environment: Any,
    samples: int,
) -> np.ndarray:
    dimension = policy.parameter_dimension
    if isinstance(environment, SO3ProductEnvironment):
        if environment.joints != 1:
            raise ValueError("diagonal Fisher helper expects one SO(3) block")
        draws = np.random.randn(samples, 6 + dimension)
        state = batch_exp_so3(draws[:, :3] * 0.3)
        target = batch_exp_so3(draws[:, 3:6] * 0.8)
        observations = so3_observation(state, target)
        standardized_noise = draws[:, 6:]
    elif isinstance(environment, SE3Environment):
        draws = np.random.randn(samples, 12 + dimension)
        initial = np.empty((samples, 6))
        initial[:, :3] = draws[:, :3] * 0.2
        initial[:, 3:] = draws[:, 3:6] * 0.3
        target_coordinates = np.empty((samples, 6))
        target_coordinates[:, :3] = draws[:, 6:9] * environment.rotation_scale
        target_coordinates[:, 3:] = draws[:, 9:12] * environment.translation_scale
        state_rotation, state_translation = batch_exp_se3(initial)
        target_rotation, target_translation = batch_exp_se3(target_coordinates)
        observations = se3_observation(
            state_rotation,
            state_translation,
            target_rotation,
            target_translation,
        )
        standardized_noise = draws[:, 12:]
    else:
        raise TypeError(f"unsupported environment type: {type(environment)!r}")
    scores = standardized_noise / policy.sigma[None, :] * observations
    return scores.T @ scores / samples


def diagonal_policy_iteration(
    theta: np.ndarray,
    environment: Any,
    episodes: int = 4,
    horizon: int = 30,
) -> tuple[np.ndarray, list[float]]:
    dimension = theta.size
    if isinstance(environment, SO3ProductEnvironment):
        draws = np.random.randn(episodes, 6 + horizon * dimension)
        state = batch_exp_so3(draws[:, :3] * 0.3)
        target = batch_exp_so3(draws[:, 3:6] * 0.8)
        standardized_noise = draws[:, 6:].reshape(episodes, horizon, dimension)
        rewards = np.empty((episodes, horizon))
        scores = np.empty((episodes, horizon, dimension))
        for time_index in range(horizon):
            observation = so3_observation(state, target)
            mean = theta[None, :] * observation
            action = mean + BASE_STD * standardized_noise[:, time_index, :]
            processed = np.clip(action, -2.0, 2.0)
            state = state @ batch_exp_so3(0.1 * processed)
            next_observation = so3_observation(state, target)
            rewards[:, time_index] = -np.sum(next_observation**2, axis=1)
            scores[:, time_index, :] = standardized_noise[:, time_index, :] / BASE_STD * observation
    elif isinstance(environment, SE3Environment):
        draws = np.random.randn(episodes, 12 + horizon * dimension)
        initial = np.empty((episodes, 6))
        initial[:, :3] = draws[:, :3] * 0.2
        initial[:, 3:] = draws[:, 3:6] * 0.3
        target_coordinates = np.empty((episodes, 6))
        target_coordinates[:, :3] = draws[:, 6:9] * environment.rotation_scale
        target_coordinates[:, 3:] = draws[:, 9:12] * environment.translation_scale
        state_rotation, state_translation = batch_exp_se3(initial)
        target_rotation, target_translation = batch_exp_se3(target_coordinates)
        standardized_noise = draws[:, 12:].reshape(episodes, horizon, dimension)
        rewards = np.empty((episodes, horizon))
        scores = np.empty((episodes, horizon, dimension))
        for time_index in range(horizon):
            observation = se3_observation(
                state_rotation,
                state_translation,
                target_rotation,
                target_translation,
            )
            mean = theta[None, :] * observation
            action = mean + BASE_STD * standardized_noise[:, time_index, :]
            coordinates = 0.1 * np.clip(action, -2.0, 2.0)
            coordinate_norm = np.linalg.norm(coordinates, axis=1)
            active = coordinate_norm > environment.increment_radius
            if np.any(active):
                coordinates[active] *= (
                    environment.increment_radius / coordinate_norm[active]
                )[:, None]
            increment_rotation, increment_translation = batch_exp_se3(coordinates)
            state_translation = state_translation + (
                state_rotation @ increment_translation[..., None]
            )[..., 0]
            state_rotation = state_rotation @ increment_rotation
            next_observation = se3_observation(
                state_rotation,
                state_translation,
                target_rotation,
                target_translation,
            )
            rewards[:, time_index] = -(
                2.0 * np.sum(next_observation[:, :3] ** 2, axis=1)
                + np.sum(next_observation[:, 3:] ** 2, axis=1)
            )
            scores[:, time_index, :] = standardized_noise[:, time_index, :] / BASE_STD * observation
    else:
        raise TypeError(f"unsupported environment type: {type(environment)!r}")
    return_to_go = np.empty_like(rewards)
    running = np.zeros(episodes)
    for time_index in range(horizon - 1, -1, -1):
        running = rewards[:, time_index] + 0.99 * running
        return_to_go[:, time_index] = running
    advantages = return_to_go.reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    gradient = np.mean(
        advantages[:, None] * scores.reshape(-1, dimension),
        axis=0,
    )
    return gradient, rewards.sum(axis=1).tolist()


def redundant_map(action_dimension: int, redundancy: int = 15, seed: int = 999) -> np.ndarray:
    rng = np.random.RandomState(seed)
    matrix = rng.randn(action_dimension, action_dimension * redundancy)
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8
    matrix *= np.linspace(0.1, 5.0, matrix.shape[1])
    return np.diag(np.linspace(0.3, 3.0, action_dimension)) @ matrix


def vectorized_auc(kind: str, joints: int, seed: int, iterations: int = 200, episodes: int = 8) -> float:
    rng = np.random.RandomState(seed)
    action_dimension = 3 * joints
    matrix = redundant_map(action_dimension) if kind == "redundant" else None
    theta = np.zeros(action_dimension if matrix is None else matrix.shape[1])
    variance = BASE_STD**2
    curve: list[float] = []
    for _ in range(iterations):
        state = batch_exp_so3(rng.randn(episodes, joints, 3) * 0.3)
        target = batch_exp_so3(rng.randn(episodes, joints, 3) * 0.8)
        mean = theta if matrix is None else matrix @ theta
        residuals = np.empty((episodes, 30, action_dimension))
        rewards = np.empty((episodes, 30))
        for time_index in range(30):
            noise = rng.randn(episodes, action_dimension) * BASE_STD
            action = np.clip(mean[None, :] + noise, -2.0, 2.0)
            increments = batch_exp_so3((0.1 * action).reshape(episodes, joints, 3))
            state = state @ increments
            error = np.swapaxes(target, -1, -2) @ state
            traces = np.trace(error, axis1=-2, axis2=-1)
            cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
            rewards[:, time_index] = -np.sum(np.arccos(cosine) ** 2, axis=1)
            residuals[:, time_index, :] = noise
        return_to_go = np.empty_like(rewards)
        running = np.zeros(episodes)
        for time_index in range(29, -1, -1):
            running = rewards[:, time_index] + 0.99 * running
            return_to_go[:, time_index] = running
        curve.append(float(rewards.sum(axis=1).mean()))
        advantage = return_to_go.reshape(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        mean_gradient = np.mean(
            advantage[:, None] * (residuals.reshape(-1, action_dimension) / variance),
            axis=0,
        )
        theta += 0.25 * (mean_gradient if matrix is None else matrix.T @ mean_gradient)
    return float(np.sum(curve))


def vectorized_auc_worker(arguments: tuple[str, int, int]) -> float:
    kind, joints, seed = arguments
    return vectorized_auc(kind, joints, seed)


def run_scalability() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joints in (5, 10, 15, 20, 30):
        arguments = (
            [("direct", joints, seed) for seed in range(N_SEEDS)]
            + [("redundant", joints, seed + 1000) for seed in range(N_SEEDS)]
        )
        workers = min(len(arguments), max(1, int(os.environ.get("LPG_WORKERS", str(N_SEEDS)))))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(vectorized_auc_worker, arguments))
        direct_aucs = values[:N_SEEDS]
        redundant_aucs = values[N_SEEDS:]
        direct_mean = float(np.mean(direct_aucs))
        redundant_mean = float(np.mean(redundant_aucs))
        rows.append({
            "joints": joints,
            "lie_dimension": 3 * joints,
            "direct_parameters": 3 * joints,
            "redundant_parameters": 45 * joints,
            "auc_magnitude_ratio": abs(direct_mean) / abs(redundant_mean),
            "direct_auc_mean": direct_mean,
            "redundant_auc_mean": redundant_mean,
        })
        print(
            f"K={joints}: direct AUC={direct_mean:.3f}, "
            f"redundant AUC={redundant_mean:.3f}, "
            f"ratio={rows[-1]['auc_magnitude_ratio']:.3f}",
            flush=True,
        )
    write_csv("scalability.csv", rows)
    return rows


def main() -> None:
    np.random.seed(SEED)
    started = time.time()
    payload = {
        "metadata": {
            "seed": SEED,
            "seeds": N_SEEDS,
            "fisher_samples": FISHER_SAMPLES,
            "fisher_ridge": FISHER_RIDGE,
            "base_standard_deviation": BASE_STD,
        },
        "theory_aligned": run_theory_aligned(),
        "fisher_alignment": run_fisher_alignment(),
        "controlled_anisotropy": run_controlled_anisotropy(),
        "se3": run_se3_diagnostics(),
        "scalability": run_scalability(),
    }
    payload["metadata"]["elapsed_seconds"] = time.time() - started
    path = RESULT_DIR / "core_experiments.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
