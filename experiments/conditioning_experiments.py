#!/usr/bin/env python3
#Run learning-rate and coordinate-conditioning experiments.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

STD = float(np.exp(-0.5))
VAR = STD**2
GRID = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
TUNING_SEEDS = [20, 21, 22]
HELDOUT_SEEDS = [30, 31, 32, 33, 34]
RIDGE_REL = 1e-4
J = 10
H = 30
GAMMA = 0.99

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("LPG_RESULT_DIR", ROOT / "results")).resolve()
FIGURE_DIR = Path(os.environ.get("LPG_FIGURE_DIR", ROOT / "figures")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def hat_batch(w):
    out = np.zeros(w.shape[:-1] + (3,3), dtype=float)
    out[...,0,1] = -w[...,2]; out[...,0,2] = w[...,1]
    out[...,1,0] = w[...,2]; out[...,1,2] = -w[...,0]
    out[...,2,0] = -w[...,1]; out[...,2,1] = w[...,0]
    return out


def exp_so3_batch(w):
    """Batched Rodrigues map without divide-by-zero warnings."""
    th = np.linalg.norm(w, axis=-1)
    K = hat_batch(w)
    th2 = th * th
    nonzero = th > 1e-8
    a = np.empty_like(th)
    b = np.empty_like(th)
    np.divide(np.sin(th), th, out=a, where=nonzero)
    np.divide(1.0 - np.cos(th), th2, out=b, where=nonzero)
    a[~nonzero] = 1.0 - th2[~nonzero] / 6.0 + th2[~nonzero] ** 2 / 120.0
    b[~nonzero] = 0.5 - th2[~nonzero] / 24.0 + th2[~nonzero] ** 2 / 720.0
    identity = np.broadcast_to(np.eye(3), K.shape)
    return identity + a[..., None, None] * K + b[..., None, None] * (K @ K)


def make_P(act_dim=30, k=15, seed=999):
    rng = np.random.RandomState(seed)
    P = rng.randn(act_dim, act_dim*k)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
    P *= np.linspace(0.1, 5.0, P.shape[1])
    P = np.diag(np.linspace(0.3, 3.0, act_dim)) @ P
    return P

P_FIXED = make_P()
PPt = P_FIXED @ P_FIXED.T
RIDGE = RIDGE_REL * np.trace(P_FIXED.T @ P_FIXED) / P_FIXED.shape[1]
NG_MEAN_SOLVE = np.linalg.inv(PPt + RIDGE*np.eye(PPt.shape[0]))


def reset_batch(rng, E):
    init_w = rng.randn(E,J,3)*0.3
    tgt_w = rng.randn(E,J,3)*0.8
    return exp_so3_batch(init_w), exp_so3_batch(tgt_w)


def train_one(kind, lr, seed, n_iters, n_episodes):
    rng = np.random.RandomState(seed)
    act_dim = 3*J
    if kind == 'direct':
        theta = np.zeros(act_dim)
    else:
        theta = np.zeros(P_FIXED.shape[1])
    curve = []
    for _ in range(n_iters):
        state, target = reset_batch(rng, n_episodes)
        if kind == 'direct':
            mu = theta
        else:
            mu = P_FIXED @ theta
        residuals = np.empty((n_episodes,H,act_dim))
        rewards = np.empty((n_episodes,H))
        for t in range(H):
            eps = rng.randn(n_episodes, act_dim)*STD
            action = mu[None,:] + eps
            # Preserve the original controlled experiment's action clipping.
            action_proc = np.clip(action, -2.0, 2.0)
            inc = exp_so3_batch((0.1*action_proc).reshape(n_episodes,J,3))
            state = state @ inc
            Rerr = np.swapaxes(target, -1, -2) @ state
            tr = np.trace(Rerr, axis1=-2, axis2=-1)
            c = np.clip((tr-1.0)/2.0, -1.0, 1.0)
            rewards[:,t] = -np.sum(np.arccos(c)**2, axis=1)
            residuals[:,t,:] = eps
        # Discounted return-to-go, matching the existing generator.
        rtg = np.empty_like(rewards)
        G = np.zeros(n_episodes)
        for t in range(H-1,-1,-1):
            G = rewards[:,t] + GAMMA*G
            rtg[:,t] = G
        curve.append(float(rewards.sum(axis=1).mean()))
        adv = rtg.reshape(-1)
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        g_mu = np.mean(adv[:,None]*(residuals.reshape(-1,act_dim)/VAR), axis=0)
        if kind == 'direct':
            theta += lr*g_mu
        elif kind == 'redundant':
            theta += lr*(P_FIXED.T@g_mu)
        elif kind == 'redundant_ng':
            # Ridge geometry correction: (P^T P + lambda I)^-1 P^T g
            # = P^T (P P^T + lambda I)^-1 g.
            theta += lr*(P_FIXED.T@(NG_MEAN_SOLVE@g_mu))
        else:
            raise ValueError(kind)
    return np.asarray(curve)


def _worker(args):
    return train_one(*args)


def run_model(kind, lr, seeds, n_iters, n_episodes):
    args=[(kind,lr,s,n_iters,n_episodes) for s in seeds]
    workers = min(len(args), max(1, int(os.environ.get("LPG_WORKERS", "8"))))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        curves=np.asarray(list(ex.map(_worker,args)))
    finals=curves[:,-40:].mean(axis=1) if n_iters>=40 else curves[:,-10:].mean(axis=1)
    aucs=curves.sum(axis=1)
    return {
      'lr':float(lr),'final_by_seed':finals.tolist(),'auc_by_seed':aucs.tolist(),
      'final_mean':float(finals.mean()),'final_sd':float(finals.std(ddof=1)),
      'auc_mean':float(aucs.mean()),'auc_sd':float(aucs.std(ddof=1))}


def paired(a, b):
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = differences.size
    if n < 2:
        raise ValueError("paired comparison requires at least two paired seeds")
    critical = float(student_t.ppf(0.975, df=n - 1))
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    half = float(critical * sd / np.sqrt(n))
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def run_learning_rate_ablation():
    out={'grid':GRID,'tuning_seeds':TUNING_SEEDS,'heldout_seeds':HELDOUT_SEEDS,
         'ridge_relative_trace_scale':RIDGE_REL,'ridge':float(RIDGE),
         'selection_metric':'mean AUC over 100 iterations, 4 episodes; larger is better',
         'common_random_numbers':'same seed used for each arm'}
    for kind in ['direct','redundant','redundant_ng']:
        print('\n##',kind,flush=True); tuning=[]
        for lr in GRID:
            r=run_model(kind,lr,TUNING_SEEDS,100,4); tuning.append(r)
            print(f"lr={lr:g} final={r['final_mean']:.3f}+/-{r['final_sd']:.3f} auc={r['auc_mean']:.3f}+/-{r['auc_sd']:.3f}",flush=True)
        best=max(tuning,key=lambda x:x['auc_mean'])
        held=run_model(kind,best['lr'],HELDOUT_SEEDS,400,8)
        out[kind]={'tuning':tuning,'selected_lr':best['lr'],'heldout':held}
        print('selected',best['lr'],flush=True)
        print(f"held final={held['final_mean']:.3f}+/-{held['final_sd']:.3f} auc={held['auc_mean']:.3f}+/-{held['auc_sd']:.3f}",flush=True)
    d=out['direct']['heldout']; r=out['redundant']['heldout']; ng=out['redundant_ng']['heldout']
    out['comparison']={
      'direct_minus_redundant_final':paired(d['final_by_seed'],r['final_by_seed']),
      'direct_minus_ng_final':paired(d['final_by_seed'],ng['final_by_seed']),
      'ng_minus_redundant_final':paired(ng['final_by_seed'],r['final_by_seed']),
      'auc_ratio_direct_over_redundant':float(abs(d['auc_mean'])/abs(r['auc_mean'])),
      'auc_ratio_ng_over_redundant':float(abs(ng['auc_mean'])/abs(r['auc_mean'])),
      'auc_ratio_direct_over_ng':float(abs(d['auc_mean'])/abs(ng['auc_mean']))}
    with (OUTPUT_DIR / 'learning_rate_ablation.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'WROTE {OUTPUT_DIR / "learning_rate_ablation.json"}', flush=True)
    print('\n',json.dumps(out['comparison'],indent=2),flush=True)

CONDITIONING_SEEDS = [30, 31, 32, 33, 34]
CONDITIONING_RATE = 0.25
CONDITIONING_ITERATIONS = 400
CONDITIONING_EPISODES = 8


def _conditioning_worker(arguments):
    return train_one(*arguments)


def _run_conditioning_arm(kind):
    arguments = [
        (kind, CONDITIONING_RATE, seed, CONDITIONING_ITERATIONS,
         CONDITIONING_EPISODES)
        for seed in CONDITIONING_SEEDS
    ]
    workers = min(
        len(arguments),
        max(1, int(os.environ.get("LPG_WORKERS", "8"))),
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        curves = np.asarray(list(executor.map(_conditioning_worker, arguments)))
    final_returns = curves[:, -40:].mean(axis=1)
    auc_values = curves.sum(axis=1)
    summary = {
        "final_by_seed": final_returns.tolist(),
        "auc_by_seed": auc_values.tolist(),
        "final_mean": float(final_returns.mean()),
        "final_sd": float(final_returns.std(ddof=1)),
        "auc_mean": float(auc_values.mean()),
        "auc_sd": float(auc_values.std(ddof=1)),
    }
    return curves, summary


def run_conditioning_stress_test():
    output = {
        "seeds": CONDITIONING_SEEDS,
        "learning_rate": CONDITIONING_RATE,
        "iterations": CONDITIONING_ITERATIONS,
        "episodes_per_iteration": CONDITIONING_EPISODES,
        "ridge_relative_trace_scale": RIDGE_REL,
        "ridge": float(RIDGE),
        "arms": {},
    }
    curves = {}
    for kind in ("direct", "redundant", "redundant_ng"):
        arm_curves, summary = _run_conditioning_arm(kind)
        curves[kind] = arm_curves
        output["arms"][kind] = summary
        print(kind, summary, flush=True)

    output["comparisons"] = {
        "direct_minus_redundant_final": paired(
            output["arms"]["direct"]["final_by_seed"],
            output["arms"]["redundant"]["final_by_seed"],
        ),
        "direct_minus_ng_final": paired(
            output["arms"]["direct"]["final_by_seed"],
            output["arms"]["redundant_ng"]["final_by_seed"],
        ),
        "ng_minus_redundant_final": paired(
            output["arms"]["redundant_ng"]["final_by_seed"],
            output["arms"]["redundant"]["final_by_seed"],
        ),
        "auc_ratio_direct_over_redundant":
            abs(output["arms"]["direct"]["auc_mean"])
            / abs(output["arms"]["redundant"]["auc_mean"]),
        "auc_ratio_ng_over_redundant":
            abs(output["arms"]["redundant_ng"]["auc_mean"])
            / abs(output["arms"]["redundant"]["auc_mean"]),
        "auc_ratio_direct_over_ng":
            abs(output["arms"]["direct"]["auc_mean"])
            / abs(output["arms"]["redundant_ng"]["auc_mean"]),
    }

    json_path = OUTPUT_DIR / "conditioning_stress_test.json"
    npz_path = OUTPUT_DIR / "conditioning_stress_test.npz"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    np.savez(npz_path, **curves)

    labels = {
        "direct": "Direct ordinary gradient",
        "redundant": "Redundant ordinary gradient",
        "redundant_ng": "Redundant ridge metric correction",
    }
    x = np.arange(1, CONDITIONING_ITERATIONS + 1)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for kind in ("direct", "redundant", "redundant_ng"):
        mean = curves[kind].mean(axis=0)
        sd = curves[kind].std(axis=0, ddof=1)
        axis.plot(x, mean, label=labels[kind])
        axis.fill_between(x, mean - sd, mean + sd, alpha=0.18)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Mean episodic return")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure_path = FIGURE_DIR / "conditioning_stress_test.png"
    figure.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(figure)

    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {npz_path}", flush=True)
    print(f"WROTE {figure_path}", flush=True)
    print(json.dumps(output["comparisons"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("all", "learning-rate", "conditioning"),
        default="all",
    )
    arguments = parser.parse_args()
    if arguments.experiment in ("all", "learning-rate"):
        run_learning_rate_ablation()
    if arguments.experiment in ("all", "conditioning"):
        run_conditioning_stress_test()


if __name__ == "__main__":
    main()
