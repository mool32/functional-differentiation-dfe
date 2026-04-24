"""Schiebinger binary-outcome replacement for fate_proxy_NN.

Replaces pluripotency_score-based fate proxy with direct 2i-arm membership label.
Two tests:

B1 — NN-propagated binary at day 2-3 (analog of pre-reg H1):
    For each day-2-3 cell, find K nearest neighbors in day-18 PCA space.
    Compute 2i-fraction of those K day-18 neighbors.
    Correlate PR with 2i-fraction. Directly uses 2i/serum label, no
    marker-gene choice.

B2 — Direct binary at day 8.5 (first post-split day):
    Compute PR at day 8.5. Correlate with binary 2i/serum label
    (point-biserial / Spearman). NO NN methodology. Most direct test
    of "does PR predict fate label". But earliest timepoint post-split
    not day 2-3, so different temporal window.

Note: day 2-3 itself has no binary label available (both arms pre-split
until day 8.5). B1 is the closest direct-outcome analog at day 2-3
without propagation.
"""
import json, os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')

SCHIEBINGER = '/Users/teo/Desktop/research/perceptual_modules/paper6/results/schiebinger_scored.h5ad'
ID_2I = '/Users/teo/Desktop/research/perceptual_modules/paper6/data/schiebinger2019/data/2i_cell_ids.txt'
ID_SERUM = '/Users/teo/Desktop/research/perceptual_modules/paper6/data/schiebinger2019/data/serum_cell_ids.txt'

N_HVG = 2000
N_PCS = 50
K_NN = 10
RNG_SEED = 20260424
N_BOOT = 10_000

DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def pr_vec(X):
    X = X.astype(np.float64)
    s1 = X.sum(axis=1); s2 = (X ** 2).sum(axis=1)
    n = X.shape[1]
    d = n * s2
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def spearman_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4: return None
    rho, p = sp.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ii = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(ii)) < 4:
            boots[i] = np.nan; continue
        boots[i] = sp.spearmanr(x[ii], y[ii]).statistic
    boots = boots[np.isfinite(boots)]
    return {'rho': float(rho), 'p': float(p), 'n': int(len(x)),
            'ci_lo': float(np.percentile(boots, 2.5)),
            'ci_hi': float(np.percentile(boots, 97.5))}


def main():
    print('[binary] loading...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)
    with open(ID_2I) as f: ids_2i = set(l.strip() for l in f if l.strip())
    with open(ID_SERUM) as f: ids_s = set(l.strip() for l in f if l.strip())
    # Exclusive labels: cells that appear ONLY in 2i (post-split iPSC arm) vs ONLY in serum
    # Cells in both = pre-split, no label information
    a.obs['only_2i'] = a.obs_names.isin(ids_2i) & ~a.obs_names.isin(ids_s)
    a.obs['only_serum'] = a.obs_names.isin(ids_s) & ~a.obs_names.isin(ids_2i)

    # HVG (sparse variance — memory safe)
    X_sp = a.X
    if hasattr(X_sp, 'multiply'):
        mean = np.asarray(X_sp.mean(axis=0)).flatten()
        Xsq = X_sp.multiply(X_sp)
        meansq = np.asarray(Xsq.mean(axis=0)).flatten()
        varp = meansq - mean ** 2
        del Xsq
    else:
        varp = np.asarray(X_sp).var(axis=0)
    if 'highly_variable' in a.var.columns:
        hv = a.var['highly_variable'].values
        hv_idx = np.where(hv)[0]
        if len(hv_idx) >= N_HVG:
            top = hv_idx[np.argsort(varp[hv_idx])[::-1][:N_HVG]]
        else:
            top = np.argsort(varp)[::-1][:N_HVG]
    else:
        top = np.argsort(varp)[::-1][:N_HVG]
    hvg = np.sort(top)

    X_hvg = _dense(a.X[:, hvg])
    pr = pr_vec(X_hvg)
    a.obs['PR'] = pr
    print(f'  {X_hvg.shape} HVG matrix, PR computed', flush=True)

    # ================================================================
    # B1 — NN-propagated binary at day 2-3
    # ================================================================
    # Day 18 cells with clean binary label
    mask_d18_labeled = (a.obs.day == DAY_LATE).values & (
        a.obs.only_2i.values | a.obs.only_serum.values)
    X_d18 = X_hvg[mask_d18_labeled]
    is_2i_d18 = a.obs.only_2i.values[mask_d18_labeled].astype(float)
    print(f'\n[B1] day 18 labeled cells: {X_d18.shape[0]} ({is_2i_d18.sum():.0f} 2i, {len(is_2i_d18)-is_2i_d18.sum():.0f} serum)', flush=True)

    pca = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_d18)
    E_d18 = pca.transform(X_d18)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)

    # Day 2-3 cells
    mask_e = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    X_e = X_hvg[mask_e]
    pr_e = pr[mask_e]
    E_e = pca.transform(X_e)
    _, ii = nn.kneighbors(E_e)
    fate_2i_frac = is_2i_d18[ii].mean(axis=1)  # fraction of K NN that are 2i
    print(f'  day 2-3 n={X_e.shape[0]}', flush=True)
    print(f'  2i-fraction: mean={fate_2i_frac.mean():.3f} std={fate_2i_frac.std():.3f} '
          f'min={fate_2i_frac.min():.3f} max={fate_2i_frac.max():.3f}', flush=True)

    r_b1 = spearman_ci(pr_e, fate_2i_frac)
    print(f'  B1 rho(PR_day2-3, 2i_fraction_from_day18_NN) = {r_b1["rho"]:+.4f}', flush=True)
    print(f'       CI [{r_b1["ci_lo"]:+.4f}, {r_b1["ci_hi"]:+.4f}]  p={r_b1["p"]:.2e}  n={r_b1["n"]}', flush=True)

    # Compare to Original pluripotency-based H1: -0.337
    print(f'  Compare to pluripotency-based H1: -0.337', flush=True)
    delta_rho = r_b1['rho'] - (-0.337)
    print(f'  Delta rho (B1 - pluripotency): {delta_rho:+.4f}', flush=True)

    # ================================================================
    # B2 — Direct binary at day 8.5 (first post-split day)
    # ================================================================
    d85 = 8.5
    mask_85_labeled = (a.obs.day == d85).values & (
        a.obs.only_2i.values | a.obs.only_serum.values)
    X_85 = X_hvg[mask_85_labeled]
    pr_85 = pr[mask_85_labeled]
    is_2i_85 = a.obs.only_2i.values[mask_85_labeled].astype(float)
    print(f'\n[B2] day 8.5 labeled cells: {X_85.shape[0]} ({is_2i_85.sum():.0f} 2i, {len(is_2i_85)-is_2i_85.sum():.0f} serum)', flush=True)

    r_b2 = spearman_ci(pr_85, is_2i_85)
    print(f'  B2 rho(PR_day8.5, binary_2i_label) = {r_b2["rho"]:+.4f}', flush=True)
    print(f'       CI [{r_b2["ci_lo"]:+.4f}, {r_b2["ci_hi"]:+.4f}]  p={r_b2["p"]:.2e}  n={r_b2["n"]}', flush=True)

    # Also: per-group means
    mu_2i = pr_85[is_2i_85 == 1].mean()
    mu_s  = pr_85[is_2i_85 == 0].mean()
    print(f'  mean PR(2i) = {mu_2i:.4f}, mean PR(serum) = {mu_s:.4f}, diff = {mu_2i - mu_s:+.4f}', flush=True)

    # ================================================================
    # B3 — Trajectory: rho per post-split day
    # ================================================================
    print('\n[B3] per-day post-split trajectory (PR vs binary 2i label)', flush=True)
    traj = {}
    for d in sorted(a.obs.day.unique()):
        if d < 8.5: continue
        mask_d = (a.obs.day == d).values & (a.obs.only_2i.values | a.obs.only_serum.values)
        n_d = mask_d.sum()
        if n_d < 50: continue
        pr_d = pr[mask_d]
        is_2i_d = a.obs.only_2i.values[mask_d].astype(float)
        rho_d, p_d = sp.spearmanr(pr_d, is_2i_d)
        traj[float(d)] = {'n': int(n_d), 'rho': float(rho_d), 'p': float(p_d),
                          'n_2i': int(is_2i_d.sum())}
        print(f'  day {d:>5}  n={n_d:>5}  n_2i={int(is_2i_d.sum()):>4}  rho={rho_d:+.4f}  p={p_d:.2e}', flush=True)

    # ================================================================
    # Plot B1 + B3 trajectory
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(pr_e, fate_2i_frac, s=3, alpha=0.15, color='#d62728')
    ax.set_xlabel('PR at day 2-3')
    ax.set_ylabel('2i-fraction of 10 day-18 NN')
    ax.set_title(f'B1: NN-propagated binary  ρ={r_b1["rho"]:+.3f}, n={r_b1["n"]}')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    days = sorted(traj.keys())
    rhos = [traj[d]['rho'] for d in days]
    ns = [traj[d]['n'] for d in days]
    ax.plot(days, rhos, 'o-', linewidth=1.6, markersize=7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('day (post-split)')
    ax.set_ylabel(r'Spearman ρ(PR_t, binary 2i label)')
    ax.set_title('B3: direct-outcome trajectory')
    ax.grid(True, alpha=0.3)
    for d, r, n in zip(days, rhos, ns):
        ax.annotate(f'n={n}', (d, r), fontsize=7, xytext=(3, 3), textcoords='offset points')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_binary_outcome.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'note': ('B1 is NN-propagated binary at day 2-3; B2 is direct binary at'
                 ' day 8.5 (first post-split day); B3 is trajectory post-split.'
                 ' Day 2-3 has no direct binary label because experimental arms'
                 ' had not yet been split (all cells in both 2i_cell_ids.txt and'
                 ' serum_cell_ids.txt at that time).'),
        'B1_nn_binary_day2_3': r_b1,
        'B1_compare_to_pluripotency_H1': -0.337,
        'B1_delta_rho_vs_pluripotency': float(r_b1['rho'] - (-0.337)),
        'B2_direct_binary_day8_5': r_b2,
        'B2_per_group_means': {'PR_2i': float(mu_2i), 'PR_serum': float(mu_s),
                               'diff': float(mu_2i - mu_s)},
        'B3_trajectory_postsplit': traj,
    }
    out = os.path.join(HERE, 'biology_schiebinger_binary_outcome_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')

    # Verdict
    print('\n=== VERDICT ===')
    print(f'B1 NN-binary ρ = {r_b1["rho"]:+.3f}')
    print(f'   vs pluripotency-based H1 ρ = -0.337')
    print(f'   delta = {delta_rho:+.3f}')
    if abs(delta_rho) < 0.10:
        print(f'   >>> SIGNALS CONSISTENT — pluripotency-based choice does not drive H1')
    elif abs(delta_rho) < 0.20:
        print(f'   >>> SIGNALS SIMILAR — pluripotency adds moderate but not dominant effect')
    else:
        print(f'   >>> SIGNALS DIVERGE — pluripotency-based choice drove the H1 magnitude')


if __name__ == '__main__':
    main()
