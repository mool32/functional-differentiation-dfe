"""Schiebinger MEF→iPSC primary test — biology pre-registration v1.

Pre-registration: biology_preregistration_v1_schiebinger.md (locked before run).

Operations:
1. Load schiebinger_scored.h5ad
2. Select top-2000 HVGs
3. For each cell: compute participation ratio PR = (sum x)^2 / (n * sum x^2)
4. Compute pluripotency score per cell (canonical markers)
5. For day-2-3 cells, compute fate proxy via KNN in day-18 PCA space
6. Primary test: Spearman rho(PR_day_2_3, fate_proxy), bootstrap 10000
7. Null H0: repeat on day 0 cells
8. Phase transition H3: per-timepoint rho across trajectory

Outputs:
  tables/biology_schiebinger_primary.csv
  analyses/biology_schiebinger_summary.json
  figures/biology_schiebinger_{primary_scatter,phase_transition}.pdf
"""
import json, os, hashlib
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
N_HVG = 2000
N_PCS = 50
K_NN = 10
RNG_SEED = 20260424
N_BOOT = 10_000

# Pre-registered markers (locked in pre-registration doc)
PLURIPOTENCY_MARKERS = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4',
                        'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0  # inclusive
DAY_NULL = 0.0
DAY_LATE = 18.0
PHASE_TIMEPOINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def participation_ratio_vec(X):
    """Row-wise PR on (n_cells, n_genes). Vectorized."""
    X = X.astype(np.float64)
    s1 = X.sum(axis=1)
    s2 = (X ** 2).sum(axis=1)
    n = X.shape[1]
    denom = n * s2
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    mask = denom > 0
    out[mask] = (s1[mask] ** 2) / denom[mask]
    return out


def ipsc_score_vec(X_subset):
    """Z-scored mean across marker genes (cells, n_markers)."""
    X = _dense(X_subset).astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    return Z.mean(axis=1)


def fit_spearman_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None
    rho, p = sp.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ii = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(ii)) < 4:
            boots[i] = np.nan
            continue
        boots[i] = sp.spearmanr(x[ii], y[ii]).statistic
    boots = boots[np.isfinite(boots)]
    return {
        'rho': float(rho), 'p': float(p), 'n': int(len(x)),
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
    }


def decision(rho_result):
    """Four-tier pre-registered decision."""
    if rho_result is None:
        return 'NO_DATA'
    rho, p = rho_result['rho'], rho_result['p']
    if rho >= 0:
        return 'FAIL_WRONG_SIGN'
    mag = abs(rho)
    ci_lo, ci_hi = rho_result['ci_lo'], rho_result['ci_hi']
    ci_excl_zero = (ci_lo < 0 and ci_hi < 0)
    if mag >= 0.20 and p < 0.05 and ci_excl_zero:
        return 'PASS'
    if mag >= 0.10:
        return 'PARTIAL'
    if mag >= 0.05:
        return 'WEAK'
    return 'FAIL'


def main():
    print('[biology] loading Schiebinger h5ad...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)
    print(f'  {a.shape} cells x genes', flush=True)
    print(f'  day range: {a.obs.day.min()} - {a.obs.day.max()}', flush=True)

    # ------ HVG slice ------
    if 'highly_variable' in a.var.columns:
        hv_mask = a.var['highly_variable'].values
        n_hv = hv_mask.sum()
        print(f'  {n_hv} genes flagged highly_variable', flush=True)
        if n_hv >= N_HVG:
            # Pick top-N_HVG by variance among HV genes
            X_full = _dense(a.X)
            var_per_gene = X_full.var(axis=0)
            hv_indices = np.where(hv_mask)[0]
            hv_vars = var_per_gene[hv_indices]
            top_k = hv_indices[np.argsort(hv_vars)[::-1][:N_HVG]]
            hvg_idx = np.sort(top_k)
        else:
            # Use all HV flag + top up with most-variable non-HV
            X_full = _dense(a.X)
            var_per_gene = X_full.var(axis=0)
            top_indices = np.argsort(var_per_gene)[::-1][:N_HVG]
            hvg_idx = np.sort(top_indices)
    else:
        print('  no highly_variable column; selecting top-2000 by variance', flush=True)
        X_full = _dense(a.X)
        var_per_gene = X_full.var(axis=0)
        hvg_idx = np.sort(np.argsort(var_per_gene)[::-1][:N_HVG])
    print(f'  using {len(hvg_idx)} HVG for PR', flush=True)

    # ------ PR per cell ------
    print('[biology] computing PR per cell...', flush=True)
    X_hvg = _dense(a.X[:, hvg_idx])
    pr = participation_ratio_vec(X_hvg)
    a.obs['PR'] = pr
    print(f'  PR: mean={np.nanmean(pr):.4f}  std={np.nanstd(pr):.4f}  '
          f'min={np.nanmin(pr):.4f}  max={np.nanmax(pr):.4f}', flush=True)

    # ------ pluripotency score per cell ------
    print('[biology] computing pluripotency score...', flush=True)
    markers_found = [g for g in PLURIPOTENCY_MARKERS if g in a.var_names]
    print(f'  markers found: {markers_found}', flush=True)
    if len(markers_found) < 3:
        raise RuntimeError(f'Only {len(markers_found)} pluripotency markers present; pre-reg requires at least 3')
    marker_idx = [list(a.var_names).index(g) for g in markers_found]
    X_markers = _dense(a.X[:, marker_idx])
    # z-score per marker across ALL cells (not just day 18)
    mu = X_markers.mean(axis=0)
    sd = X_markers.std(axis=0) + 1e-9
    a.obs['pluripotency_score'] = ((X_markers - mu) / sd).mean(axis=1)
    print(f'  pluripotency mean={a.obs.pluripotency_score.mean():.4f}', flush=True)

    # ------ fate proxy via NN in day-18 PCA space ------
    print('[biology] fitting PCA on day 18 cells...', flush=True)
    mask_d18 = a.obs.day == DAY_LATE
    a_d18 = a[mask_d18]
    print(f'  day 18: {a_d18.shape[0]} cells', flush=True)
    X_d18 = _dense(a_d18.X[:, hvg_idx])
    pca = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_d18)
    E_d18 = pca.transform(X_d18)
    ipsc_d18 = a_d18.obs['pluripotency_score'].values
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)

    def fate_for_cells(indices_in_full):
        X_q = _dense(a.X[indices_in_full, :][:, hvg_idx])
        E_q = pca.transform(X_q)
        _, ii = nn.kneighbors(E_q)
        return ipsc_d18[ii].mean(axis=1)

    # ------ H1 PRIMARY: day 2-3 ------
    print('[biology] H1 primary: day 2-3...', flush=True)
    mask_early = (a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)
    idx_early = np.where(mask_early.values)[0]
    print(f'  day 2-3: {len(idx_early)} cells', flush=True)
    fate_early = fate_for_cells(idx_early)
    pr_early = a.obs.PR.values[idx_early]
    r_primary = fit_spearman_ci(pr_early, fate_early)
    v_primary = decision(r_primary)
    print(f'  rho={r_primary["rho"]:+.4f}  CI [{r_primary["ci_lo"]:+.4f}, {r_primary["ci_hi"]:+.4f}]  '
          f'p={r_primary["p"]:.2e}  n={r_primary["n"]}  =>  {v_primary}', flush=True)

    # ------ H0 NULL: day 0 ------
    print('[biology] H0 null: day 0...', flush=True)
    mask_d0 = a.obs.day == DAY_NULL
    idx_d0 = np.where(mask_d0.values)[0]
    print(f'  day 0: {len(idx_d0)} cells', flush=True)
    if len(idx_d0) > 0:
        fate_d0 = fate_for_cells(idx_d0)
        pr_d0 = a.obs.PR.values[idx_d0]
        r_null = fit_spearman_ci(pr_d0, fate_d0)
        h0_pass = (r_null is not None and abs(r_null['rho']) < 0.15)
        print(f'  rho={r_null["rho"]:+.4f}  CI [{r_null["ci_lo"]:+.4f}, {r_null["ci_hi"]:+.4f}]  '
              f'p={r_null["p"]:.2e}  n={r_null["n"]}', flush=True)
        print(f'  H0 pass (|rho|<0.15)?  {h0_pass}', flush=True)
    else:
        r_null = None; h0_pass = False
        print('  no day-0 cells', flush=True)

    # ------ H3 PHASE TRANSITION ------
    print('[biology] H3 phase transition (exploratory)...', flush=True)
    phase_results = {}
    for t in PHASE_TIMEPOINTS:
        mask_t = a.obs.day == t
        idx_t = np.where(mask_t.values)[0]
        if len(idx_t) < 50:
            phase_results[t] = {'n': int(len(idx_t)), 'rho': None}
            continue
        fate_t = fate_for_cells(idx_t)
        pr_t = a.obs.PR.values[idx_t]
        r_t = fit_spearman_ci(pr_t, fate_t, n_boot=1000)  # cheaper bootstrap for trajectory
        phase_results[t] = r_t
        print(f'    day {t:>5.1f}  n={r_t["n"]:>5}  rho={r_t["rho"]:+.3f}  p={r_t["p"]:.2e}', flush=True)

    # ------ per-cell table ------
    print('[biology] saving per-cell table...', flush=True)
    mask_to_save = (a.obs.day == 0.0) | mask_early | mask_d18
    idx_save = np.where(mask_to_save.values)[0]
    fate_save = fate_for_cells(idx_save)
    save_df = pd.DataFrame({
        'cell_id': a.obs_names[idx_save],
        'day': a.obs.day.values[idx_save],
        'PR': a.obs.PR.values[idx_save],
        'pluripotency_score': a.obs.pluripotency_score.values[idx_save],
        'fate_proxy_nn': fate_save,
    })
    out_csv = os.path.join(TABLES, 'biology_schiebinger_primary.csv')
    save_df.to_csv(out_csv, index=False)
    print(f'  wrote {out_csv}', flush=True)

    # ------ plots ------
    print('[biology] plotting...', flush=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(pr_early, fate_early, s=4, alpha=0.25, color='#d62728')
    ax.set_xlabel('participation ratio at day 2-3')
    ax.set_ylabel('fate proxy (mean pluripotency of 10 day-18 NN)')
    ax.set_title(f'Schiebinger primary H1: rho={r_primary["rho"]:+.3f}, p={r_primary["p"]:.2e}, n={r_primary["n"]}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_primary_scatter.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    days = sorted(phase_results.keys())
    rhos = [phase_results[t]['rho'] if phase_results[t].get('rho') is not None else np.nan for t in days]
    ns = [phase_results[t]['n'] for t in days]
    ax.plot(days, rhos, 'o-', linewidth=1.6)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(DAY_EARLY_LO, color='grey', linestyle=':', alpha=0.4)
    ax.axvline(DAY_EARLY_HI, color='grey', linestyle=':', alpha=0.4)
    ax.set_xlabel('training/developmental day')
    ax.set_ylabel(r'Spearman $\rho$(PR_t, fate_proxy_NN)')
    ax.set_title('Schiebinger H3 phase transition (exploratory)')
    ax.grid(True, alpha=0.3)
    for d, r, n in zip(days, rhos, ns):
        if not np.isnan(r):
            ax.annotate(f'n={n}', (d, r), fontsize=7, xytext=(3, 3), textcoords='offset points')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_phase_transition.pdf'))
    plt.close(fig)

    # ------ verdict ------
    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'pre_registration_commit': 'tbd',  # filled post-commit
        'config': {
            'n_hvg': N_HVG, 'n_pcs': N_PCS, 'k_nn': K_NN,
            'day_early_range': [DAY_EARLY_LO, DAY_EARLY_HI],
            'markers_found': markers_found,
        },
        'h1_primary': {'rho': r_primary, 'verdict': v_primary},
        'h0_null': {'rho': r_null, 'h0_pass': bool(h0_pass)},
        'h3_phase_transition': phase_results,
    }
    out_json = os.path.join(HERE, 'biology_schiebinger_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n[biology] wrote {out_json}')
    print(f'\n=== VERDICT ===')
    print(f'H1 primary: {v_primary}  (rho={r_primary["rho"]:+.3f}, pre-registered direction negative)')
    if r_null is not None:
        print(f'H0 null:    {"PASS" if h0_pass else "FAIL"}  (rho={r_null["rho"]:+.3f}, threshold |rho|<0.15)')
    print(f'H3 exploratory: trajectory saved, see phase transition plot')


if __name__ == '__main__':
    main()
