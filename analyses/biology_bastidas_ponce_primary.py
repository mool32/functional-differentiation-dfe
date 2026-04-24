"""Bastidas-Ponce pancreatic endocrinogenesis — one-shot pre-registered test.

Pre-registration: biology_preregistration_bastidas_ponce.md (locked in same commit).

NO post-hoc tweaks. NO alternative methodology if this fails.

Executes:
- Methodology-null gate (§3.2 of pre-reg)
- H1 primary (§3.1)
- S1 scrambled-identity control (§3.3)
- S2 trajectory across E12.5-E15.5 (§3.3)

Outputs fixed per pre-reg §7.
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

BP = '/Users/teo/Desktop/research/perceptual_modules/paper6/results/bastidas_ponce_scored.h5ad'

# LOCKED PARAMETERS
N_HVG = 2000
N_PCS = 50
K_NN = 10
RNG_SEED = 20260424
N_SHUFFLES_GATE = 200

ENDOCRINE_CLUSTERS = {'Alpha', 'Beta', 'Delta', 'Epsilon', 'Fev+',
                      'Ngn3 high EP', 'Ngn3 low EP',
                      'Ngn3 High late', 'Ngn3 High early',
                      'Fev+ Beta', 'Fev+ Alpha', 'Fev+ Epsilon', 'Fev+ Pyy'}
# Check this against actual cluster names; any endocrine-subtype name counts.

DAY_EARLY = 12.5
DAY_LATE = 15.5


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def sparse_variance(X):
    if hasattr(X, 'multiply'):
        mean = np.asarray(X.mean(axis=0)).flatten()
        Xsq = X.multiply(X)
        meansq = np.asarray(Xsq.mean(axis=0)).flatten()
        return meansq - mean ** 2
    return np.asarray(X).var(axis=0)


def compute_V4a(module_scores, orientation_anchor_mask):
    M = np.asarray(module_scores, dtype=np.float64)
    pca = PCA(n_components=1, random_state=RNG_SEED).fit(M)
    pc1 = pca.transform(M).flatten()
    # Orientation: committed (anchor = True) should have LOW V4a
    if pc1[orientation_anchor_mask].mean() > 0:
        pc1 = -pc1
    return pc1, pca


def main():
    print('[BP] loading Bastidas-Ponce...', flush=True)
    a = ad.read_h5ad(BP)
    print(f'  shape: {a.shape}', flush=True)
    days = pd.to_numeric(a.obs['day'].astype(str), errors='coerce').values
    print(f'  day counts: {dict(pd.Series(days).value_counts().sort_index())}', flush=True)

    # ------------------- module scores + V4a -------------------
    ms = np.asarray(a.obsm['module_scores'])
    n_modules = ms.shape[1]
    print(f'  modules: {ms.shape}', flush=True)

    # Cluster column — identify endocrine cells at E15.5
    cluster_col = 'clusters_fig6_broad_final'
    clusters = a.obs[cluster_col].astype(str).values
    unique_clusters = pd.Series(clusters).unique()
    print(f'  clusters in {cluster_col}: {len(unique_clusters)}')

    # For orientation anchor: endocrine cells at E15.5
    mask_e15 = (days == DAY_LATE)
    is_endo = np.isin(clusters, list(ENDOCRINE_CLUSTERS))
    orient_mask = mask_e15 & is_endo
    n_orient = orient_mask.sum()
    print(f'  orientation anchor (E15.5 endocrine): {n_orient} cells', flush=True)
    if n_orient < 50:
        raise RuntimeError(f'too few E15.5 endocrine cells for orientation: {n_orient}')

    v4a, pca_mod = compute_V4a(ms, orient_mask)
    print(f'  V4a: mean={v4a.mean():.4f} std={v4a.std():.4f}  '
          f'var_exp_PC1={pca_mod.explained_variance_ratio_[0]:.3f}', flush=True)
    print(f'  V4a mean in orient anchor (E15.5 endo): {v4a[orient_mask].mean():+.3f} '
          f'(should be negative after orientation)', flush=True)

    # ------------------- HVG + PCA for fate proxy -------------------
    print('[BP] selecting HVGs...', flush=True)
    varp = sparse_variance(a.X)
    hv_flag = a.var.get('highly_variable_genes', a.var.get('highly_variable'))
    if hv_flag is not None:
        hv_flag = hv_flag.values
        hv_idx = np.where(hv_flag)[0]
        if len(hv_idx) >= N_HVG:
            top = hv_idx[np.argsort(varp[hv_idx])[::-1][:N_HVG]]
        else:
            top = np.argsort(varp)[::-1][:N_HVG]
    else:
        top = np.argsort(varp)[::-1][:N_HVG]
    hvg = np.sort(top)
    print(f'  using {len(hvg)} HVGs', flush=True)

    X_e15_hvg = _dense(a.X[:, hvg])[mask_e15]
    is_endo_e15 = is_endo[mask_e15].astype(float)

    pca_fate = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_e15_hvg)
    E_15 = pca_fate.transform(X_e15_hvg)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_15)

    mask_e12 = (days == DAY_EARLY)
    idx_e12 = np.where(mask_e12)[0]
    X_e12_hvg = _dense(a.X[idx_e12, :][:, hvg])
    E_12 = pca_fate.transform(X_e12_hvg)
    _, ii = nn.kneighbors(E_12)
    fate_e12 = is_endo_e15[ii].mean(axis=1)
    print(f'  E12.5 cells: {len(idx_e12)}, E15.5 endocrine in NN pool: {is_endo_e15.sum():.0f}/{len(is_endo_e15)}')

    v4a_e12 = v4a[idx_e12]

    # ------------------- METHODOLOGY NULL GATE -------------------
    print(f'\n[BP] methodology null gate (n={N_SHUFFLES_GATE} shuffles of E15 endocrine labels)...', flush=True)
    rng = np.random.default_rng(RNG_SEED)
    null_rhos = np.empty(N_SHUFFLES_GATE)
    for i in range(N_SHUFFLES_GATE):
        shuf = rng.permutation(is_endo_e15)
        fate_null = shuf[ii].mean(axis=1)
        null_rhos[i] = sp.spearmanr(v4a_e12, fate_null).statistic
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{N_SHUFFLES_GATE}', flush=True)
    null_mean = float(null_rhos.mean())
    null_std = float(null_rhos.std())
    null_ci_95 = (float(np.percentile(null_rhos, 2.5)),
                  float(np.percentile(null_rhos, 97.5)))
    null_span = max(abs(null_ci_95[0]), abs(null_ci_95[1]))
    print(f'  null mean={null_mean:+.4f}  std={null_std:.4f}')
    print(f'  null 95% CI: [{null_ci_95[0]:+.4f}, {null_ci_95[1]:+.4f}]')
    print(f'  null span (max |ρ|): {null_span:.4f}')

    # Gate per §3.2
    if null_span <= 0.10:
        gate = 'PASS'
        magnitude_adjusted_threshold = {'partial': 0.15, 'strong': 0.25}
    elif null_span <= 0.15:
        gate = 'CAUTION'
        magnitude_adjusted_threshold = {'partial': 0.20, 'strong': 0.30}
    else:
        gate = 'FAIL'
        magnitude_adjusted_threshold = None
    print(f'  >>> methodology gate: {gate} <<<')

    # ------------------- H1 PRIMARY -------------------
    print('\n[BP] H1 primary test: ρ(V4a_E12.5, endocrine_fraction_E15_NN)...', flush=True)
    r_h1, p_h1 = sp.spearmanr(v4a_e12, fate_e12)
    # Bootstrap CI
    n_boot = 10000
    rng2 = np.random.default_rng(RNG_SEED + 1)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng2.choice(len(v4a_e12), size=len(v4a_e12), replace=True)
        if len(np.unique(idx)) < 4:
            boots[i] = np.nan; continue
        boots[i] = sp.spearmanr(v4a_e12[idx], fate_e12[idx]).statistic
    boots = boots[np.isfinite(boots)]
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    print(f'  ρ = {r_h1:+.4f}  CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p = {p_h1:.2e}  n = {len(v4a_e12)}')

    # Decision
    if gate == 'FAIL':
        verdict = 'HOLD (methodology gate failed — cannot distinguish from noise)'
    else:
        mag = abs(r_h1)
        if magnitude_adjusted_threshold is None:
            verdict = 'HOLD'
        elif r_h1 > 0 and mag >= 0.10:
            verdict = 'FAIL_WRONG_DIRECTION'
        elif mag >= magnitude_adjusted_threshold['strong'] and r_h1 < 0 and p_h1 < 0.01:
            verdict = 'PASS'
        elif mag >= magnitude_adjusted_threshold['partial'] and r_h1 < 0:
            verdict = 'PARTIAL'
        elif mag >= 0.10 and r_h1 < 0:
            verdict = 'WEAK'
        else:
            verdict = 'NULL'
    print(f'  >>> H1 verdict: {verdict} <<<')

    # ------------------- S1 scrambled identity control -------------------
    print('\n[BP] S1 scrambled identity control...', flush=True)
    rng3 = np.random.default_rng(RNG_SEED + 2)
    # Shuffle V4a values among E12.5 cells (breaks cell-level assoc)
    v4a_scr = rng3.permutation(v4a_e12)
    r_s1, p_s1 = sp.spearmanr(v4a_scr, fate_e12)
    print(f'  ρ = {r_s1:+.4f}  p = {p_s1:.2e}  (expect ≈ 0)')

    # ------------------- S2 trajectory -------------------
    print('\n[BP] S2 trajectory across timepoints...', flush=True)
    traj = {}
    for t in [12.5, 13.5, 14.5, 15.5]:
        mask_t = (days == t)
        idx_t = np.where(mask_t)[0]
        if len(idx_t) < 50: continue
        X_t = _dense(a.X[idx_t, :][:, hvg])
        E_t = pca_fate.transform(X_t)
        _, ii_t = nn.kneighbors(E_t)
        fate_t = is_endo_e15[ii_t].mean(axis=1)
        r_t, p_t = sp.spearmanr(v4a[idx_t], fate_t)
        traj[float(t)] = {'n': int(len(idx_t)), 'rho': float(r_t), 'p': float(p_t)}
        print(f'  E{t:.1f}  n={len(idx_t):>5}  rho={r_t:+.4f}  p={p_t:.2e}')

    # ------------------- save -------------------
    save_df = pd.DataFrame({
        'cell_id': a.obs_names[idx_e12],
        'V4a_E12_5': v4a_e12,
        'endocrine_fraction_NN': fate_e12,
    })
    out_csv = os.path.join(TABLES, 'biology_bastidas_ponce_primary.csv')
    save_df.to_csv(out_csv, index=False)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.scatter(v4a_e12, fate_e12, s=4, alpha=0.25, color='#2ca02c')
    ax.set_xlabel('V4a (module PC1 signed) at E12.5')
    ax.set_ylabel('Endocrine fraction of 10 NN at E15.5')
    ax.set_title(f'H1 primary: ρ={r_h1:+.3f} CI[{ci_lo:+.3f},{ci_hi:+.3f}] n={len(v4a_e12)}')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(null_rhos, bins=30, color='#999999', alpha=0.7, label='methodology null')
    ax.axvline(r_h1, color='#d62728', linewidth=2, label=f'observed={r_h1:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('ρ'); ax.set_ylabel('count')
    ax.set_title(f'Methodology null (span={null_span:.3f}, gate={gate})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    days_list = sorted(traj.keys())
    ax.plot(days_list, [traj[d]['rho'] for d in days_list], 'o-', linewidth=1.6, markersize=8)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    for d in days_list:
        ax.annotate(f'n={traj[d]["n"]}\np={traj[d]["p"]:.1e}',
                    (d, traj[d]['rho']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('embryonic day')
    ax.set_ylabel('ρ(V4a_t, endocrine_fraction_NN)')
    ax.set_title('S2 trajectory')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_bastidas_ponce_all.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'pre_registration': 'biology_preregistration_bastidas_ponce.md',
        'config': {'N_HVG': N_HVG, 'N_PCS': N_PCS, 'K_NN': K_NN,
                   'N_SHUFFLES_GATE': N_SHUFFLES_GATE},
        'methodology_null': {
            'mean': null_mean, 'std': null_std,
            'ci_95': null_ci_95, 'span': float(null_span),
            'gate': gate,
            'adjusted_thresholds': magnitude_adjusted_threshold,
        },
        'H1_primary': {
            'rho': float(r_h1), 'p': float(p_h1),
            'ci_95': [ci_lo, ci_hi],
            'n': int(len(v4a_e12)),
            'verdict': verdict,
        },
        'S1_scrambled': {'rho': float(r_s1), 'p': float(p_s1)},
        'S2_trajectory': traj,
    }
    out_json = os.path.join(HERE, 'biology_bastidas_ponce_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n>>> FINAL VERDICT <<<')
    print(f'Methodology gate: {gate}  (null span = {null_span:.4f})')
    print(f'H1 ρ = {r_h1:+.4f}  verdict: {verdict}')
    print(f'S1 scrambled ρ = {r_s1:+.4f} (sanity)')
    print(f'\nwrote {out_json}')
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
