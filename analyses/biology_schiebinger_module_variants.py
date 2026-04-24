"""Schiebinger module-variants test — extension of Step 1.

Three alternative operationalizations of per-cell module concentration,
motivated by Paper 3's Δ_PC1 24/24 sign-consistency which validated
modules as carrying cell-type-level signal.

  V2: raw PR on module scores -- (Σm)²/(n·Σm²) on 43 modules per cell.
      Same formula as gene-PR but on module activity vector.
  V3: top-3 concentration ratio -- sum(top3)/sum(all 43).
  V4a: PC1 score (signed) -- project cell's 43-module vector onto PC1
       learned from module-activity PCA fit on all 236k cells.
  V4b: |PC1 score| -- magnitude along dominant module axis.

For each variant:
- rho(variant_day_2_3, fate_proxy_pluripotency) + CI
- rho(variant_day_2_3, fate_proxy_binary_2i) + CI
- permutation null (1000 shuffles of variant among day-2-3 cells)
- trajectory across Phase timepoints (both fate types)
- cross-correlation with gene-PR and module entropy (diagnostic)

Not pre-registered as new test — extension of v1 Step 1 per user.
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
N_PERM = 1000

PLURIPOTENCY = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4', 'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0
DAY_NULL = 0.0
PHASE_TPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def sparse_variance(X):
    if hasattr(X, 'multiply'):
        mean = np.asarray(X.mean(axis=0)).flatten()
        Xsq = X.multiply(X)
        meansq = np.asarray(Xsq.mean(axis=0)).flatten()
        return meansq - mean ** 2
    return np.asarray(X).var(axis=0)


def raw_pr(M):
    """Per-row raw participation ratio on module scores.
    M: (n_cells, 43) module activity matrix, typically non-negative."""
    M = np.asarray(M, dtype=np.float64)
    s1 = M.sum(axis=1); s2 = (M ** 2).sum(axis=1)
    n = M.shape[1]
    d = n * s2
    out = np.full(M.shape[0], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def top3_ratio(M):
    """Fraction of total module activity in top-3 modules per cell."""
    M = np.asarray(M, dtype=np.float64)
    M_pos = np.maximum(M, 0)  # only positive activity counts
    sums = M_pos.sum(axis=1)
    sorted_ = np.sort(M_pos, axis=1)[:, ::-1]
    top3 = sorted_[:, :3].sum(axis=1)
    out = np.full(M.shape[0], np.nan, dtype=np.float64)
    m = sums > 0
    out[m] = top3[m] / sums[m]
    return out


def module_pca_pc1(M):
    """Project each cell onto PC1 of global module-activity PCA.
    Returns signed PC1 score per cell. |PC1| is the magnitude variant."""
    pca = PCA(n_components=1, random_state=RNG_SEED).fit(M)
    return pca.transform(M).flatten(), pca


def gene_pr(X):
    X = X.astype(np.float64)
    s1 = X.sum(axis=1); s2 = (X ** 2).sum(axis=1)
    n = X.shape[1]
    d = n * s2
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def softmax_entropy(M, beta=1.0):
    s = np.asarray(M, dtype=np.float64) * beta
    s_stable = s - s.max(axis=1, keepdims=True)
    p = np.exp(s_stable); p /= p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-20, 1.0)
    return -(p * np.log(p)).sum(axis=1)


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


def permutation_null(x, y, n_perm=N_PERM, seed=RNG_SEED):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho_obs = sp.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        xs = rng.permutation(x)
        nulls[i] = sp.spearmanr(xs, y).statistic
    p = 2 * min((nulls >= rho_obs).mean(), (nulls <= rho_obs).mean())
    p = max(p, 1 / (n_perm + 1))
    return {'observed_rho': float(rho_obs),
            'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std()),
            'null_ci_95': (float(np.percentile(nulls, 2.5)),
                           float(np.percentile(nulls, 97.5))),
            'empirical_p_two_sided': float(p)}


def main():
    print('[variants] loading...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)
    ms = np.asarray(a.obsm['module_scores'])
    print(f'  modules: {ms.shape}  range [{ms.min():.3f}, {ms.max():.3f}]', flush=True)

    # ===== Compute all 4 variants per cell =====
    print('[variants] computing V2 (raw PR on modules)...', flush=True)
    v2 = raw_pr(ms)
    print('[variants] computing V3 (top-3 ratio)...', flush=True)
    v3 = top3_ratio(ms)
    print('[variants] computing V4 (module PCA PC1)...', flush=True)
    v4a, mod_pca = module_pca_pc1(ms)
    v4b = np.abs(v4a)
    print(f'  V2: mean={v2.mean():.4f} std={v2.std():.4f}', flush=True)
    print(f'  V3: mean={v3.mean():.4f} std={v3.std():.4f}', flush=True)
    print(f'  V4a (PC1): mean={v4a.mean():.4f} std={v4a.std():.4f}  var_explained={mod_pca.explained_variance_ratio_[0]:.3f}', flush=True)

    # ===== Fate proxy (same as before) =====
    print('[variants] building fate proxies...', flush=True)
    varp = sparse_variance(a.X)
    if 'highly_variable' in a.var.columns:
        hv_mask = a.var['highly_variable'].values
        hv_idx = np.where(hv_mask)[0]
        if len(hv_idx) >= N_HVG:
            top = hv_idx[np.argsort(varp[hv_idx])[::-1][:N_HVG]]
        else:
            top = np.argsort(varp)[::-1][:N_HVG]
    else:
        top = np.argsort(varp)[::-1][:N_HVG]
    hvg = np.sort(top)

    markers_found = [g for g in PLURIPOTENCY if g in a.var_names]
    marker_idx = [list(a.var_names).index(g) for g in markers_found]
    X_mk = _dense(a.X[:, marker_idx])
    mu = X_mk.mean(axis=0); sd = X_mk.std(axis=0) + 1e-9
    plur = ((X_mk - mu) / sd).mean(axis=1)

    with open(ID_2I) as f: ids_2i = set(l.strip() for l in f if l.strip())
    with open(ID_SERUM) as f: ids_s = set(l.strip() for l in f if l.strip())
    only_2i = np.asarray([c in ids_2i and c not in ids_s for c in a.obs_names])
    only_serum = np.asarray([c in ids_s and c not in ids_2i for c in a.obs_names])

    mask_d18 = (a.obs.day == DAY_LATE).values
    mask_d18_lab = mask_d18 & (only_2i | only_serum)
    X_d18_hvg = _dense(a.X[:, hvg])[mask_d18_lab]
    plur_d18_lab = plur[mask_d18_lab]
    is_2i_d18_lab = only_2i[mask_d18_lab].astype(float)

    pca_fate = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_d18_hvg)
    E_d18 = pca_fate.transform(X_d18_hvg)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)

    def fate_for(idx):
        X_q = _dense(a.X[idx, :][:, hvg])
        E_q = pca_fate.transform(X_q)
        _, ii = nn.kneighbors(E_q)
        return plur_d18_lab[ii].mean(axis=1), is_2i_d18_lab[ii].mean(axis=1)

    # ===== H1-equivalent at day 2-3 for each variant =====
    mask_e = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    idx_e = np.where(mask_e)[0]
    fate_plur_e, fate_bin_e = fate_for(idx_e)
    v2_e, v3_e, v4a_e, v4b_e = v2[idx_e], v3[idx_e], v4a[idx_e], v4b[idx_e]

    results = {}
    print('\n=== DAY 2-3 PRIMARY TESTS ===')
    print(f'{"variant":<20} {"fate":<12} {"rho":>8} {"CI":>22} {"perm p":>10}')
    print('-' * 75)
    for name, vals in [('V2 raw_PR_module', v2_e),
                       ('V3 top3_ratio', v3_e),
                       ('V4a PC1_signed', v4a_e),
                       ('V4b |PC1|', v4b_e)]:
        for fname, fvec in [('pluripotency', fate_plur_e), ('binary_2i', fate_bin_e)]:
            r = spearman_ci(vals, fvec, n_boot=2000)
            pnull = permutation_null(vals, fvec, n_perm=500)
            key = f'{name}__{fname}'
            results[key] = {'corr': r, 'null': pnull}
            print(f'{name:<20} {fname:<12} {r["rho"]:+8.4f} '
                  f'[{r["ci_lo"]:+6.3f},{r["ci_hi"]:+6.3f}]  {pnull["empirical_p_two_sided"]:.2e}')

    # ===== Trajectory per variant =====
    print('\n=== TRAJECTORY PER VARIANT ===')
    print(f'{"day":>5}  ' + '  '.join([f'{n:>12}' for n in ['V2 plur', 'V2 bin', 'V3 plur', 'V3 bin',
                                                             'V4a plur', 'V4b plur']]))
    traj = {}
    for t in PHASE_TPS:
        mask_t = (a.obs.day == t).values
        idx_t = np.where(mask_t)[0]
        if len(idx_t) < 50: continue
        f_plur_t, f_bin_t = fate_for(idx_t)
        row = {'n': len(idx_t)}
        for name, vec in [('V2', v2), ('V3', v3), ('V4a', v4a), ('V4b', v4b)]:
            vec_t = vec[idx_t]
            row[f'{name}_plur'] = float(sp.spearmanr(vec_t, f_plur_t).statistic)
            row[f'{name}_bin'] = float(sp.spearmanr(vec_t, f_bin_t).statistic)
        traj[float(t)] = row
        print(f'{t:>5.1f}  ' + '  '.join([f'{row["V2_plur"]:+12.3f}', f'{row["V2_bin"]:+12.3f}',
                                           f'{row["V3_plur"]:+12.3f}', f'{row["V3_bin"]:+12.3f}',
                                           f'{row["V4a_plur"]:+12.3f}', f'{row["V4b_plur"]:+12.3f}']))

    # ===== Diagnostic: cross-correlations among variants + gene-PR + entropy =====
    print('\n=== DIAGNOSTIC: variant-variant correlations on day 2-3 ===')
    # Also need gene_PR on day 2-3 for comparison
    X_hvg_e = _dense(a.X[idx_e, :][:, hvg])
    gpr_e = gene_pr(X_hvg_e)
    ent_e = softmax_entropy(ms, beta=1.0)[idx_e]

    candidates = {'V2_modPR': v2_e, 'V3_top3': v3_e, 'V4a_PC1': v4a_e,
                  'V4b_|PC1|': v4b_e, 'gene_PR': gpr_e, 'softmax_ent': ent_e}
    cross = {}
    names = list(candidates.keys())
    print(f'{"":>12} ' + ' '.join([f'{n:>10}' for n in names]))
    for n1 in names:
        row_str = f'{n1:>12} '
        row_vals = {}
        for n2 in names:
            r = sp.spearmanr(candidates[n1], candidates[n2]).statistic
            row_vals[n2] = float(r)
            row_str += f' {r:>+9.3f}'
        cross[n1] = row_vals
        print(row_str)

    # ===== Plots =====
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    variants_plot = [('V2 raw_PR_module', v2_e, 'V2'),
                     ('V3 top3_ratio', v3_e, 'V3'),
                     ('V4a PC1_signed', v4a_e, 'V4a'),
                     ('V4b |PC1|', v4b_e, 'V4b')]
    for ax, (title, vals, short) in zip(axes.flat, variants_plot):
        ax.scatter(vals, fate_bin_e, s=3, alpha=0.15, color='#1f77b4', label='binary')
        ax2 = ax.twinx()
        ax2.scatter(vals, fate_plur_e, s=3, alpha=0.15, color='#d62728', label='pluripotency')
        r_plur = results[f'{title}__pluripotency']['corr']['rho']
        r_bin = results[f'{title}__binary_2i']['corr']['rho']
        ax.set_title(f'{title}  ρ_plur={r_plur:+.3f}  ρ_bin={r_bin:+.3f}')
        ax.set_xlabel(short); ax.set_ylabel('2i fraction'); ax2.set_ylabel('pluripotency fate')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_module_variants_scatter.pdf'))
    plt.close(fig)

    # Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    days_list = sorted(traj.keys())
    colors = {'V2': '#1f77b4', 'V3': '#ff7f0e', 'V4a': '#2ca02c', 'V4b': '#d62728'}
    for ax, ftype, label in [(axes[0], 'plur', 'pluripotency'),
                              (axes[1], 'bin', 'binary 2i')]:
        for v in ['V2', 'V3', 'V4a', 'V4b']:
            ys = [traj[d][f'{v}_{ftype}'] for d in days_list]
            ax.plot(days_list, ys, 'o-', label=v, color=colors[v], linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvspan(DAY_EARLY_LO, DAY_EARLY_HI, alpha=0.1, color='grey')
        ax.set_xlabel('day'); ax.set_ylabel(f'ρ vs {label} fate')
        ax.set_title(f'Trajectory — {label} fate')
        ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_module_variants_trajectory.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'note': ('Extension of Step 1 per user directive 2026-04-24: test four'
                 ' module-activity operationalizations closer to Paper 3 metrics.'),
        'module_stats': {
            'shape': list(ms.shape),
            'V2_mean': float(v2.mean()), 'V2_std': float(v2.std()),
            'V3_mean': float(v3.mean()), 'V3_std': float(v3.std()),
            'V4a_var_explained': float(mod_pca.explained_variance_ratio_[0]),
        },
        'day2_3_results': results,
        'trajectory': traj,
        'cross_correlations': cross,
    }
    out = os.path.join(HERE, 'biology_schiebinger_module_variants_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')

    # Verdict
    print('\n=== VERDICT ===')
    for v in ['V2 raw_PR_module', 'V3 top3_ratio', 'V4a PC1_signed', 'V4b |PC1|']:
        r_plur = results[f'{v}__pluripotency']['corr']['rho']
        r_bin = results[f'{v}__binary_2i']['corr']['rho']
        mag = max(abs(r_plur), abs(r_bin))
        if mag < 0.05: tier = 'NULL'
        elif mag < 0.10: tier = 'WEAK'
        elif mag < 0.20: tier = 'PARTIAL'
        else: tier = 'PASS'
        print(f'  {v:<22}  ρ_plur={r_plur:+.3f}  ρ_bin={r_bin:+.3f}  max|ρ|={mag:.3f}  {tier}')


if __name__ == '__main__':
    main()
