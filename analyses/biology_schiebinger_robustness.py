"""Schiebinger robustness — pre-registered sensitivity checks (Step A).

Varies each parameter independently around the pre-registered default:
  N_HVG: 500, 1000, 2000 (default), 4000
  K_NN:  5, 10 (default), 20, 50
  N_PCs: 20, 50 (default), 100

For each configuration, recomputes H1 primary: rho(PR_day2-3, fate_proxy).
Verifies stability of the -0.337 finding.

Pre-reg threshold: |rho| shouldn't jump by > 0.10 across choices.
"""
import json, os, itertools
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
RNG_SEED = 20260424

PLURIPOTENCY_MARKERS = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4',
                        'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0

# Default config
DEFAULTS = {'N_HVG': 2000, 'K_NN': 10, 'N_PCS': 50}

# Sensitivity variations (one-at-a-time)
SWEEPS = {
    'N_HVG': [500, 1000, 2000, 4000],
    'K_NN': [5, 10, 20, 50],
    'N_PCS': [20, 50, 100],
}


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def participation_ratio_vec(X):
    X = X.astype(np.float64)
    s1 = X.sum(axis=1)
    s2 = (X ** 2).sum(axis=1)
    n = X.shape[1]
    denom = n * s2
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    mask = denom > 0
    out[mask] = (s1[mask] ** 2) / denom[mask]
    return out


def sparse_variance(X):
    """Per-column variance of sparse or dense X (memory-safe)."""
    if hasattr(X, 'multiply'):
        mean = np.asarray(X.mean(axis=0)).flatten()
        Xsq = X.multiply(X)
        mean_sq = np.asarray(Xsq.mean(axis=0)).flatten()
        return mean_sq - mean ** 2
    X = np.asarray(X)
    return X.var(axis=0)


def select_hvg(a, n_hvg):
    var_per = sparse_variance(a.X)
    if 'highly_variable' in a.var.columns:
        hv_mask = a.var['highly_variable'].values
        hv_idx_all = np.where(hv_mask)[0]
        if len(hv_idx_all) >= n_hvg:
            top_k = hv_idx_all[np.argsort(var_per[hv_idx_all])[::-1][:n_hvg]]
            return np.sort(top_k)
    # Fallback: top n_hvg by variance globally
    return np.sort(np.argsort(var_per)[::-1][:n_hvg])


def run_config(a, n_hvg, k_nn, n_pcs, cache_hvg=None):
    # HVG
    if cache_hvg is not None and cache_hvg['n_hvg'] == n_hvg:
        hvg_idx = cache_hvg['idx']
        X_hvg = cache_hvg['X']
    else:
        hvg_idx = select_hvg(a, n_hvg)
        X_hvg = _dense(a.X[:, hvg_idx])

    # PR
    pr = participation_ratio_vec(X_hvg)

    # markers
    markers_found = [g for g in PLURIPOTENCY_MARKERS if g in a.var_names]
    marker_idx = [list(a.var_names).index(g) for g in markers_found]
    X_mk = _dense(a.X[:, marker_idx])
    mu = X_mk.mean(axis=0)
    sd = X_mk.std(axis=0) + 1e-9
    plur = ((X_mk - mu) / sd).mean(axis=1)

    # masks
    mask_d18 = (a.obs.day == DAY_LATE).values
    mask_early = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    X_d18 = X_hvg[mask_d18]
    X_early = X_hvg[mask_early]
    plur_d18 = plur[mask_d18]
    pr_early = pr[mask_early]

    # PCA + NN
    pca = PCA(n_components=n_pcs, random_state=RNG_SEED).fit(X_d18)
    E_d18 = pca.transform(X_d18)
    E_early = pca.transform(X_early)
    nn = NearestNeighbors(n_neighbors=k_nn).fit(E_d18)
    _, ii = nn.kneighbors(E_early)
    fate = plur_d18[ii].mean(axis=1)

    rho, p = sp.spearmanr(pr_early, fate)
    return {'N_HVG': int(n_hvg), 'K_NN': int(k_nn), 'N_PCS': int(n_pcs),
            'rho': float(rho), 'p': float(p),
            'n_day_2_3': int(X_early.shape[0]),
            'n_day18': int(X_d18.shape[0])}, hvg_idx, X_hvg


def main():
    print('[robustness] loading Schiebinger...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)

    results = []
    # Default (for reference)
    print('\n=== default config (2000 HVG, k=10, 50 PCs) ===', flush=True)
    res_def, hvg_idx_2000, X_hvg_2000 = run_config(a, 2000, 10, 50)
    print(f'  rho = {res_def["rho"]:+.4f}  p = {res_def["p"]:.2e}', flush=True)
    results.append({**res_def, 'sweep_param': 'default'})

    # Cache 2000 HVG for reuse
    cache = {'n_hvg': 2000, 'idx': hvg_idx_2000, 'X': X_hvg_2000}

    # Sweep N_HVG
    print('\n=== N_HVG sweep ===', flush=True)
    for n_hvg in SWEEPS['N_HVG']:
        if n_hvg == 2000:
            r = res_def
        else:
            r, _, _ = run_config(a, n_hvg, 10, 50)
        print(f'  N_HVG={n_hvg:>5}  rho = {r["rho"]:+.4f}  p = {r["p"]:.2e}', flush=True)
        results.append({**r, 'sweep_param': 'N_HVG'})

    # Sweep K_NN (reuse 2000 HVG cache)
    print('\n=== K_NN sweep ===', flush=True)
    for k in SWEEPS['K_NN']:
        if k == 10:
            r = res_def
        else:
            r, _, _ = run_config(a, 2000, k, 50, cache_hvg=cache)
        print(f'  K_NN={k:>3}  rho = {r["rho"]:+.4f}  p = {r["p"]:.2e}', flush=True)
        results.append({**r, 'sweep_param': 'K_NN'})

    # Sweep N_PCS
    print('\n=== N_PCS sweep ===', flush=True)
    for n_pcs in SWEEPS['N_PCS']:
        if n_pcs == 50:
            r = res_def
        else:
            r, _, _ = run_config(a, 2000, 10, n_pcs, cache_hvg=cache)
        print(f'  N_PCS={n_pcs:>4}  rho = {r["rho"]:+.4f}  p = {r["p"]:.2e}', flush=True)
        results.append({**r, 'sweep_param': 'N_PCS'})

    # Save
    df = pd.DataFrame(results)
    out_csv = os.path.join(TABLES, 'biology_schiebinger_robustness.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')

    # Stability check
    all_rhos = [r['rho'] for r in results if r['sweep_param'] != 'default']
    max_rho = max(all_rhos)
    min_rho = min(all_rhos)
    range_rho = max_rho - min_rho
    print(f'\n=== STABILITY ===')
    print(f'  rho range across all configs: [{min_rho:+.3f}, {max_rho:+.3f}]')
    print(f'  range magnitude: {range_rho:.3f}')
    print(f'  pre-reg threshold: range < 0.10 for stable')
    stable = range_rho < 0.10
    print(f'  VERDICT: {"STABLE" if stable else "UNSTABLE"}')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ['N_HVG', 'K_NN', 'N_PCS']):
        sub = df[df['sweep_param'] == key].sort_values(key)
        ax.plot(sub[key], sub['rho'], 'o-', linewidth=1.6, markersize=8)
        ax.axhline(res_def['rho'], color='#d62728', linestyle='--', alpha=0.7,
                   label=f'default ρ={res_def["rho"]:+.3f}')
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.set_xscale('log' if key != 'N_PCS' else 'linear')
        ax.set_xlabel(key)
        ax.set_ylabel('Spearman ρ(PR_day2-3, fate_proxy)')
        ax.set_title(f'sensitivity to {key}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_robustness.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'default': res_def,
        'sweeps': results,
        'rho_range': [min_rho, max_rho],
        'range_magnitude': range_rho,
        'pre_registered_threshold_range': 0.10,
        'verdict_stable': bool(stable),
    }
    out_json = os.path.join(HERE, 'biology_schiebinger_robustness_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'wrote {out_json}')


if __name__ == '__main__':
    main()
