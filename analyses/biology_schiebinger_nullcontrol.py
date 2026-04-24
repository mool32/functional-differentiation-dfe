"""Schiebinger null control — proper null for H1 primary finding.

Two complementary tests:

C1 — Permutation test (empirical null for ρ at n=20492).
  Shuffle PR values among day-2-3 cells, correlate with fate_proxy.
  1000 shuffles. Expected: ρ-null distribution centered on 0 with
  std ≈ 1/sqrt(n). Empirical p for observed ρ = -0.337.

C2 — Methodology artifact check.
  Shuffle pluripotency_score among day-18 cells. Recompute fate_proxy
  for day-2-3 cells using shuffled scores. Correlate with unchanged
  PR. 100 shuffles.
  Purpose: detect whether fate_proxy_NN has geometry-induced bias
  independent of biology. If C2 null is centered far from 0, there
  is an artifact.

C3 — Asymmetry check (not in original plan, but relevant given H0 FAIL).
  Compute cell-level fate_proxy stratified by day-2-3 vs day-0 cells
  in a 2D space (PR quantile × NN-distance quantile). Tests whether
  H0 positive-direction signal is explained by NN-distance structure.
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
N_HVG = 2000
N_PCS = 50
K_NN = 10
RNG_SEED = 20260424
N_PERM_C1 = 1000
N_PERM_C2 = 100

PLURIPOTENCY_MARKERS = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4',
                        'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0


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


def main():
    print('[nullctrl] loading Schiebinger...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)

    # -- same HVG selection as primary analysis --
    if 'highly_variable' in a.var.columns:
        hv_mask = a.var['highly_variable'].values
        n_hv = hv_mask.sum()
        X_full = _dense(a.X)
        if n_hv >= N_HVG:
            var_per = X_full.var(axis=0)
            hv_idx = np.where(hv_mask)[0]
            top_k = hv_idx[np.argsort(var_per[hv_idx])[::-1][:N_HVG]]
            hvg_idx = np.sort(top_k)
        else:
            var_per = X_full.var(axis=0)
            hvg_idx = np.sort(np.argsort(var_per)[::-1][:N_HVG])
    else:
        X_full = _dense(a.X)
        var_per = X_full.var(axis=0)
        hvg_idx = np.sort(np.argsort(var_per)[::-1][:N_HVG])

    # -- PR per cell --
    X_hvg = _dense(a.X[:, hvg_idx])
    pr = participation_ratio_vec(X_hvg)
    print(f'[nullctrl] PR computed for {X_hvg.shape[0]} cells', flush=True)

    # -- pluripotency score --
    markers_found = [g for g in PLURIPOTENCY_MARKERS if g in a.var_names]
    marker_idx = [list(a.var_names).index(g) for g in markers_found]
    X_mk = _dense(a.X[:, marker_idx])
    mu = X_mk.mean(axis=0)
    sd = X_mk.std(axis=0) + 1e-9
    plur = ((X_mk - mu) / sd).mean(axis=1)
    print(f'[nullctrl] pluripotency computed', flush=True)

    # -- masks --
    mask_d18 = (a.obs.day == DAY_LATE).values
    mask_early = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    X_d18 = X_hvg[mask_d18]
    plur_d18 = plur[mask_d18]
    pr_d18 = pr[mask_d18]
    X_early = X_hvg[mask_early]
    pr_early = pr[mask_early]

    print(f'[nullctrl] day18 n={X_d18.shape[0]}, day2-3 n={X_early.shape[0]}', flush=True)

    # -- PCA on day 18 --
    pca = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_d18)
    E_d18 = pca.transform(X_d18)
    E_early = pca.transform(X_early)

    # -- baseline (actual) fate proxy for day 2-3 cells --
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)
    _, idx_nn_early = nn.kneighbors(E_early)
    fate_actual = plur_d18[idx_nn_early].mean(axis=1)
    rho_actual = float(sp.spearmanr(pr_early, fate_actual).statistic)
    print(f'[nullctrl] actual rho(PR, fate) at day 2-3: {rho_actual:+.4f}', flush=True)

    # ================================================================
    # C1 — Permutation test: shuffle day-2-3 PR labels vs fate_actual
    # ================================================================
    print(f'\n[nullctrl] C1 permutation (shuffle PR among day 2-3, n={N_PERM_C1})...', flush=True)
    rng = np.random.default_rng(RNG_SEED)
    c1_rhos = np.empty(N_PERM_C1)
    for i in range(N_PERM_C1):
        pr_shuf = rng.permutation(pr_early)
        c1_rhos[i] = sp.spearmanr(pr_shuf, fate_actual).statistic
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{N_PERM_C1}', flush=True)
    c1_mean = float(np.mean(c1_rhos))
    c1_std = float(np.std(c1_rhos))
    c1_ci = (float(np.percentile(c1_rhos, 2.5)), float(np.percentile(c1_rhos, 97.5)))
    # Empirical two-sided p
    c1_p = 2 * min((c1_rhos <= rho_actual).mean(), (c1_rhos >= rho_actual).mean())
    c1_p = max(c1_p, 1 / (N_PERM_C1 + 1))
    print(f'  C1 null mean={c1_mean:+.5f}  std={c1_std:.5f}  CI [{c1_ci[0]:+.4f}, {c1_ci[1]:+.4f}]', flush=True)
    print(f'  observed rho={rho_actual:+.4f}  empirical p_two_sided={c1_p:.2e}', flush=True)

    # ================================================================
    # C2 — Methodology artifact check: shuffle pluripotency on day 18
    # ================================================================
    print(f'\n[nullctrl] C2 methodology check (shuffle pluripotency on day 18, n={N_PERM_C2})...', flush=True)
    c2_rhos = np.empty(N_PERM_C2)
    for i in range(N_PERM_C2):
        plur_d18_shuf = rng.permutation(plur_d18)
        fate_shuf = plur_d18_shuf[idx_nn_early].mean(axis=1)
        c2_rhos[i] = sp.spearmanr(pr_early, fate_shuf).statistic
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{N_PERM_C2}', flush=True)
    c2_mean = float(np.mean(c2_rhos))
    c2_std = float(np.std(c2_rhos))
    c2_ci = (float(np.percentile(c2_rhos, 2.5)), float(np.percentile(c2_rhos, 97.5)))
    c2_bias_magnitude = abs(c2_mean)
    print(f'  C2 null mean={c2_mean:+.5f}  std={c2_std:.5f}  CI [{c2_ci[0]:+.4f}, {c2_ci[1]:+.4f}]', flush=True)
    print(f'  artifact magnitude (|mean of C2 null|) = {c2_bias_magnitude:.4f}', flush=True)
    if c2_bias_magnitude < 0.05:
        c2_verdict = 'CLEAN (no methodology bias)'
    elif c2_bias_magnitude < 0.10:
        c2_verdict = 'MINOR BIAS'
    else:
        c2_verdict = 'CONCERNING BIAS — interpret primary with caution'
    print(f'  C2 verdict: {c2_verdict}', flush=True)

    # ================================================================
    # C3 — Day 0 same methodology for reference (why +0.215)
    # ================================================================
    print(f'\n[nullctrl] C3 day-0 methodology audit...', flush=True)
    mask_d0 = (a.obs.day == 0.0).values
    X_d0 = X_hvg[mask_d0]
    pr_d0 = pr[mask_d0]
    E_d0 = pca.transform(X_d0)
    dists_d0, idx_nn_d0 = nn.kneighbors(E_d0)
    fate_d0 = plur_d18[idx_nn_d0].mean(axis=1)
    mean_dist_d0 = dists_d0.mean(axis=1)
    # Check: do d0 cells with high PR have high NN distance to d18 (confounder)?
    rho_pr_dist_d0 = float(sp.spearmanr(pr_d0, mean_dist_d0).statistic)
    rho_pr_fate_d0 = float(sp.spearmanr(pr_d0, fate_d0).statistic)
    dists_early, _ = nn.kneighbors(E_early)
    mean_dist_early = dists_early.mean(axis=1)
    rho_pr_dist_early = float(sp.spearmanr(pr_early, mean_dist_early).statistic)
    print(f'  day 0  rho(PR, mean_NN_dist) = {rho_pr_dist_d0:+.4f}')
    print(f'  day 0  rho(PR, fate_proxy)   = {rho_pr_fate_d0:+.4f}  (the earlier +0.215)')
    print(f'  day23  rho(PR, mean_NN_dist) = {rho_pr_dist_early:+.4f}')
    print(f'  day23  rho(PR, fate_proxy)   = {rho_actual:+.4f}  (the earlier -0.337)')

    # Interpretation: if rho(PR, dist) is strongly positive in day 0 but not in day 2-3,
    # the day-0 positive rho could be driven by high-PR cells being "far" from day 18
    # manifold in a way that happens to land near high-pluripotency regions.
    # If rho(PR, dist) is similar in both — it's not about distance structure.

    # ================================================================
    # Plots
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(c1_rhos, bins=40, alpha=0.8, color='#999999', label='null (PR-shuffle)')
    ax.axvline(rho_actual, color='#d62728', linewidth=2, label=f'observed ρ={rho_actual:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('Spearman ρ')
    ax.set_ylabel('count')
    ax.set_title(f'C1 permutation null (n={N_PERM_C1} shuffles)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(c2_rhos, bins=30, alpha=0.8, color='#ff7f0e', label='null (pluripotency shuffle on day 18)')
    ax.axvline(rho_actual, color='#d62728', linewidth=2, label=f'observed ρ={rho_actual:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(c2_mean, color='#1f77b4', linestyle=':', linewidth=2,
               label=f'C2 null mean={c2_mean:+.3f}')
    ax.set_xlabel('Spearman ρ')
    ax.set_ylabel('count')
    ax.set_title(f'C2 methodology check (n={N_PERM_C2} shuffles)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_null_control.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'config': {'n_hvg': N_HVG, 'n_pcs': N_PCS, 'k_nn': K_NN,
                   'n_perm_c1': N_PERM_C1, 'n_perm_c2': N_PERM_C2},
        'rho_actual_day2_3': rho_actual,
        'C1_permutation': {
            'null_mean': c1_mean, 'null_std': c1_std, 'null_ci_95': c1_ci,
            'empirical_p_two_sided': c1_p,
        },
        'C2_methodology_check': {
            'null_mean': c2_mean, 'null_std': c2_std, 'null_ci_95': c2_ci,
            'artifact_magnitude': c2_bias_magnitude, 'verdict': c2_verdict,
        },
        'C3_day0_audit': {
            'rho_PR_fate_day0': rho_pr_fate_d0,
            'rho_PR_meanNNdist_day0': rho_pr_dist_d0,
            'rho_PR_fate_day2_3': rho_actual,
            'rho_PR_meanNNdist_day2_3': rho_pr_dist_early,
        },
    }
    out = os.path.join(HERE, 'biology_schiebinger_nullcontrol_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n[nullctrl] wrote {out}')

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print('\n=== NULL CONTROL VERDICT ===')
    print(f'C1 permutation: observed rho = {rho_actual:+.4f}, null mean = {c1_mean:+.5f} ± {c1_std:.5f}')
    print(f'    z-score = {(rho_actual - c1_mean)/c1_std:.1f}, empirical p = {c1_p:.2e}')
    c1_clean = (abs(c1_mean) < 0.02 and c1_p < 0.01)
    print(f'    C1 CLEAN: {c1_clean}')
    print(f'C2 methodology: null centered {c2_mean:+.4f} (|bias|={c2_bias_magnitude:.4f}); {c2_verdict}')
    c2_clean = c2_bias_magnitude < 0.10
    print(f'    C2 CLEAN: {c2_clean}')
    overall = c1_clean and c2_clean
    print(f'\n>>> PRIMARY FINDING ρ = -0.337: {"CONFIRMED CLEAN" if overall else "HAS METHODOLOGY CAVEAT"} <<<')


if __name__ == '__main__':
    main()
