#!/usr/bin/env python3
#Run the state-dependent radius-projected policy experiment
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(max(1, int(os.environ.get("LPG_TORCH_THREADS", "1"))))
K=4
P=10  # 9 entries of Q_*^T Q plus bias
H=20
T=100
BATCH=int(math.ceil(math.sqrt(T)))
GAMMA=0.98
STD=0.35
ACTION_SCALE=0.12
ACTION_PENALTY=0.01
PILOT_SEEDS=[100,101]
EVAL_SEEDS=[110,111,112,113,114]
TAU_GRID=[0.0025,0.005,0.01,0.02,0.04]
R_GRID=[0.5,0.75,1.0,1.5,2.0]

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("LPG_RESULT_DIR", ROOT / "results")).resolve()
FIGURE_DIR = Path(os.environ.get("LPG_FIGURE_DIR", ROOT / "figures")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def hat(w):
    z=torch.zeros_like(w[...,0])
    return torch.stack([
      torch.stack([z,-w[...,2],w[...,1]],-1),
      torch.stack([w[...,2],z,-w[...,0]],-1),
      torch.stack([-w[...,1],w[...,0],z],-1)],-2)


def exp_so3(w):
    """Batched Rodrigues map with finite values and derivatives at zero."""
    th = torch.linalg.vector_norm(w, dim=-1)
    Kmat = hat(w)
    th2 = th * th
    nonzero = th > 1e-7
    safe_th = torch.where(nonzero, th, torch.ones_like(th))
    safe_th2 = safe_th * safe_th
    a_exact = torch.sin(th) / safe_th
    b_exact = (1.0 - torch.cos(th)) / safe_th2
    a_series = 1.0 - th2 / 6.0 + th2 * th2 / 120.0
    b_series = 0.5 - th2 / 24.0 + th2 * th2 / 720.0
    a = torch.where(nonzero, a_exact, a_series)
    b = torch.where(nonzero, b_exact, b_series)
    identity = torch.eye(3, dtype=w.dtype, device=w.device).expand(Kmat.shape)
    return identity + a[..., None, None] * Kmat + b[..., None, None] * (Kmat @ Kmat)


def sample_reset(gen,E):
    # Initial errors and targets remain away from singular feature maps because
    # features use the rotation matrix entries directly, not log/vee.
    tgt=exp_so3(torch.randn((E,K,3), generator=gen, device=gen.device)*0.8)
    rel=exp_so3(torch.randn((E,K,3), generator=gen, device=gen.device)*0.7)
    state=tgt@rel
    return state,tgt


def features(state,target):
    R=torch.transpose(target,-1,-2)@state
    flat=R.reshape(R.shape[0],K,9)
    one=torch.ones((R.shape[0],K,1), dtype=R.dtype, device=R.device)
    return torch.cat([flat,one],dim=-1)


def rollout_objective(theta,gen,E):
    state,target=sample_reset(gen,E)
    total=torch.zeros(E, dtype=theta.dtype, device=theta.device)
    disc=1.0
    for _ in range(H):
        phi=features(state,target)
        mu=torch.einsum('ekp,kpc->ekc',phi,theta)
        eps=torch.randn((E,K,3), generator=gen, dtype=theta.dtype, device=theta.device)
        action=mu+STD*eps
        state=state@exp_so3(ACTION_SCALE*action)
        R=torch.transpose(target,-1,-2)@state
        trace=torch.diagonal(R,dim1=-2,dim2=-1).sum(-1)
        reward=(trace-3.0).sum(-1)-ACTION_PENALTY*(action*action).sum(dim=(-1,-2))
        total=total+disc*reward
        disc*=GAMMA
    return total.mean()


def run_one(seed,tau,R0,constrained=True,n_iters=T):
    torch.manual_seed(seed)
    theta=torch.zeros((K,P,3), requires_grad=True)
    gen=torch.Generator().manual_seed(seed+100000)
    returns=[]; norms=[]; mapping=[]; active=0
    for _ in range(n_iters):
        J=rollout_objective(theta,gen,BATCH)
        grad,=torch.autograd.grad(J,theta)
        with torch.no_grad():
            cand=theta+tau*grad
            if constrained:
                cn=torch.linalg.vector_norm(cand)
                if cn>R0:
                    cand=cand*(R0/cn); active+=1
            Gmap=(cand-theta)/tau
            theta.copy_(cand)
            returns.append(float(J))
            norms.append(float(torch.linalg.vector_norm(theta)))
            mapping.append(float(torch.linalg.vector_norm(Gmap)))
    return {'seed':seed,'return_curve':returns,'norm_curve':norms,'mapping_curve':mapping,
            'active_count':active,'active_fraction':active/n_iters,
            'final_return_mean':float(np.mean(returns[-20:])),
            'final_norm':norms[-1],'final_mapping_mean':float(np.mean(mapping[-20:]))}


def _worker(args):
    return run_one(*args)


def summarize_rows(rows):
    final=np.array([r['final_return_mean'] for r in rows]); active=np.array([r['active_fraction'] for r in rows])
    norm=np.array([r['final_norm'] for r in rows]); mapping=np.array([r['final_mapping_mean'] for r in rows])
    return {'rows':rows,'final_mean':float(final.mean()),'final_sd':float(final.std(ddof=1)) if len(final)>1 else 0,
            'active_mean':float(active.mean()),'active_sd':float(active.std(ddof=1)) if len(active)>1 else 0,
            'norm_mean':float(norm.mean()),'norm_sd':float(norm.std(ddof=1)) if len(norm)>1 else 0,
            'mapping_mean':float(mapping.mean()),'mapping_sd':float(mapping.std(ddof=1)) if len(mapping)>1 else 0}


def run_group(seeds,tau,R0,constrained,n_iters=T):
    args=[(s,tau,R0,constrained,n_iters) for s in seeds]
    workers=max(1, min(len(args), int(os.environ.get('LPG_WORKERS', '8'))))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows=list(ex.map(_worker,args))
    return summarize_rows(rows)


def pilot():
    configurations=[(tau,R) for tau in TAU_GRID for R in R_GRID]
    args=[(seed,tau,R,True,60) for tau,R in configurations for seed in PILOT_SEEDS]
    workers=max(1, min(len(args), int(os.environ.get('LPG_WORKERS', '8'))))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        completed=list(ex.map(_worker,args))
    out=[]
    offset=0
    for tau,R in configurations:
        rows=completed[offset:offset+len(PILOT_SEEDS)]
        offset+=len(PILOT_SEEDS)
        r=summarize_rows(rows)
        out.append({'tau':tau,'R0':R,**{k:v for k,v in r.items() if k!='rows'}})
        print('pilot',tau,R,r['final_mean'],r['active_mean'],r['norm_mean'],flush=True)
    feasible=[x for x in out if 0.1<=x['active_mean']<=0.8]
    if not feasible:
        feasible=out
    feasible.sort(key=lambda x:(abs(x['active_mean']-0.35),-x['final_mean']))
    return out,feasible[0]


def paired(a,b):
    d=np.array(a)-np.array(b); t=2.7764451051977987
    m=float(d.mean()); sd=float(d.std(ddof=1)); h=float(t*sd/math.sqrt(len(d)))
    return {'mean':m,'sd':sd,'ci95':[m-h,m+h]}



def _curve_matrix(group, key):
    return np.asarray([row[key] for row in group["rows"]], dtype=float)


def _plot_band(axis, x, values, label, linestyle="-"):
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    axis.plot(x, mean, linestyle=linestyle, label=label)
    axis.fill_between(x, mean - sd, mean + sd, alpha=0.18)


def write_figure(data):
    constrained = data["constrained"]
    unconstrained = data["unconstrained"]
    return_c = _curve_matrix(constrained, "return_curve")
    return_u = _curve_matrix(unconstrained, "return_curve")
    norm_c = _curve_matrix(constrained, "norm_curve")
    norm_u = _curve_matrix(unconstrained, "norm_curve")
    mapping_c = _curve_matrix(constrained, "mapping_curve")
    mapping_u = _curve_matrix(unconstrained, "mapping_curve")
    x = np.arange(1, return_c.shape[1] + 1)
    radius = float(data["design"]["R0"])

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    _plot_band(axes[0], x, return_c, rf"LPG, $R_0={radius:g}$")
    _plot_band(axes[0], x, return_u, "Unconstrained", "--")
    axes[0].set(title="(a) Smooth policy return", xlabel="Iteration",
                ylabel="Mean episodic return")
    axes[0].legend(frameon=False)

    _plot_band(axes[1], x, norm_c, rf"LPG, $R_0={radius:g}$")
    _plot_band(axes[1], x, norm_u, "Unconstrained", "--")
    axes[1].axhline(radius, linestyle=":", label=r"$R_0$")
    axes[1].set(title="(b) Active radius", xlabel="Iteration",
                ylabel="Coefficient norm")
    axes[1].legend(frameon=False)

    _plot_band(axes[2], x, mapping_c, rf"LPG, $R_0={radius:g}$")
    _plot_band(axes[2], x, mapping_u, "Unconstrained", "--")
    axes[2].set(title="(c) Sampled mapping norm", xlabel="Iteration",
                ylabel=r"$\Vert G_\tau\Vert$")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output = FIGURE_DIR / "state_dependent_policy.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"WROTE {output}", flush=True)

def main():
    pilot_rows,sel=pilot(); tau=sel['tau']; R=sel['R0']
    print('SELECTED',tau,R,flush=True)
    evaluation_args=(
        [(seed,tau,R,True,T) for seed in EVAL_SEEDS]
        + [(seed,tau,R,False,T) for seed in EVAL_SEEDS]
    )
    workers=max(1, min(len(evaluation_args), int(os.environ.get('LPG_WORKERS', '8'))))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        evaluation_rows=list(ex.map(_worker,evaluation_args))
    con=summarize_rows(evaluation_rows[:len(EVAL_SEEDS)])
    unc=summarize_rows(evaluation_rows[len(EVAL_SEEDS):])
    cf=[r['final_return_mean'] for r in con['rows']]; uf=[r['final_return_mean'] for r in unc['rows']]
    out={'design':{'K':K,'features':'nine entries of Q_target^T Q plus bias','H':H,'T':T,'batch':BATCH,
                   'gamma':GAMMA,'std':STD,'action_scale':ACTION_SCALE,'action_penalty':ACTION_PENALTY,
                   'tau':tau,'nominal_L':1/(2*tau),'R0':R,'pilot_seeds':PILOT_SEEDS,'eval_seeds':EVAL_SEEDS,
                   'no_action_clipping':True,'no_return_standardization':True,'gradient':'Monte Carlo pathwise'},
         'pilot':pilot_rows,'constrained':con,'unconstrained':unc,
         'paired_constrained_minus_unconstrained':paired(cf,uf)}
    with (OUTPUT_DIR / 'state_dependent_policy.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    np.savez(OUTPUT_DIR / 'state_dependent_policy.npz',
      constrained_returns=np.array([r['return_curve'] for r in con['rows']]),
      unconstrained_returns=np.array([r['return_curve'] for r in unc['rows']]),
      constrained_norms=np.array([r['norm_curve'] for r in con['rows']]),
      unconstrained_norms=np.array([r['norm_curve'] for r in unc['rows']]),
      constrained_mapping=np.array([r['mapping_curve'] for r in con['rows']]),
      unconstrained_mapping=np.array([r['mapping_curve'] for r in unc['rows']]))
    write_figure(out)
    print(f'WROTE {OUTPUT_DIR / "state_dependent_policy.json"}', flush=True)
    print(f'WROTE {OUTPUT_DIR / "state_dependent_policy.npz"}', flush=True)
    print(json.dumps({k:v for k,v in out.items() if k not in ['pilot']},indent=2),flush=True)

if __name__=='__main__': main()
