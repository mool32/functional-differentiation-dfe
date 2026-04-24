"""V4a (module PC1 signed) — robustness + methodology variance check.

Step 1: Robustness to module-subset / z-scoring / PC count choices.
Step 2: Methodology variance check — shuffle pluripotency on day 18,
        see what ρ methodology produces under null.

Goal: establish confidence interval on V4a ρ ≈ -0.15 finding before
moving to Bastidas-Ponce.
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

N_HVG = 2000; N_PCS_FATE = 50; K_NN = 10
RNG_SEED = 20260424
N_PERM_C2 = 100

PLURIPOTENCY = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4', 'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def sparse_variance(X):
    if hasattr(X, 'multiply'):
        mean = np.asarray(X.mean(axis=0)).flatten()
        Xsq = X.multiply(X)
        meansq = np.asarray(Xsq.mean(axis=0)).flatten()
        return meansq - mean ** 2
    return np.asarray(X).var(axis=0)


def compute_v4a(M_scores, pre_zscore=False, n_pcs=1, return_loadings=False):
    """V4a-like measure: PCA on modules, sum of top-n PC scores (signed) per cell.
    If n_pcs=1 returns signed PC1 score per cell.
    If n_pcs>1 returns signed sum of top PC scores per cell (weighted by var explained).
    """
    M = np.asarray(M_scores, dtype=np.float64)
    if pre_zscore:
        mu = M.mean(axis=0); sd = M.std(axis=0) + 1e-9
        M = (M - mu) / sd
    pca = PCA(n_components=max(n_pcs, 3), random_state=RNG_SEED).fit(M)
    scores = pca.transform(M)
    if n_pcs == 1:
        out = scores[:, 0]
    else:
        # Weighted sum of top-n PC scores by variance explained
        w = pca.explained_variance_ratio_[:n_pcs]
        out = scores[:, :n_pcs] @ w
    if return_loadings:
        return out, pca
    return out


def main():
    print('[v4a_checks] loading...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)
    ms = np.asarray(a.obsm['module_scores'])
    n_modules = ms.shape[1]
    print(f'  modules: {ms.shape}', flush=True)

    # ===== Build fate proxies (same infrastructure) =====
    varp = sparse_variance(a.X)
    hv_mask = a.var.get('highly_variable')
    if hv_mask is not None:
        hv_mask = hv_mask.values
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
    plur_all = ((X_mk - mu) / sd).mean(axis=1)

    with open(ID_2I) as f: ids_2i = set(l.strip() for l in f if l.strip())
    with open(ID_SERUM) as f: ids_s = set(l.strip() for l in f if l.strip())
    only_2i = np.asarray([c in ids_2i and c not in ids_s for c in a.obs_names])
    only_serum = np.asarray([c in ids_s and c not in ids_2i for c in a.obs_names])

    mask_d18 = (a.obs.day == DAY_LATE).values
    mask_d18_lab = mask_d18 & (only_2i | only_serum)
    X_d18_hvg = _dense(a.X[:, hvg])[mask_d18_lab]
    plur_d18_lab = plur_all[mask_d18_lab]
    is_2i_d18_lab = only_2i[mask_d18_lab].astype(float)

    pca_fate = PCA(n_components=N_PCS_FATE, random_state=RNG_SEED).fit(X_d18_hvg)
    E_d18 = pca_fate.transform(X_d18_hvg)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)

    mask_e = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    idx_e = np.where(mask_e)[0]
    X_e_hvg = _dense(a.X[idx_e, :][:, hvg])
    E_e = pca_fate.transform(X_e_hvg)
    _, ii_e = nn.kneighbors(E_e)
    fate_plur_e = plur_d18_lab[ii_e].mean(axis=1)
    fate_bin_e = is_2i_d18_lab[ii_e].mean(axis=1)

    # ===== Baseline: default V4a config =====
    v4a_default = compute_v4a(ms, pre_zscore=False, n_pcs=1)
    rho_default_plur = float(sp.spearmanr(v4a_default[idx_e], fate_plur_e).statistic)
    rho_default_bin = float(sp.spearmanr(v4a_default[idx_e], fate_bin_e).statistic)
    print(f'\nBaseline V4a (43 modules, no z-score, PC1 signed):')
    print(f'  ρ_plur={rho_default_plur:+.4f}  ρ_bin={rho_default_bin:+.4f}')

    # ============================================================
    # STEP 1: ROBUSTNESS
    # ============================================================
    print('\n===== STEP 1: ROBUSTNESS =====')
    robust_results = []

    # A) Module subset sweep: 43 (full), top-22 by PC1 loading, random-22
    print('\n(A) Module subset:')
    _, pca_default = compute_v4a(ms, n_pcs=1, return_loadings=True)
    pc1_loadings = np.abs(pca_default.components_[0])
    top22_idx = np.argsort(pc1_loadings)[::-1][:22]
    rng = np.random.default_rng(RNG_SEED)
    rnd22_idx = rng.choice(n_modules, size=22, replace=False)
    for name, idx in [('all_43', None),
                       ('top22_by_PC1_loading', top22_idx),
                       ('random_22', rnd22_idx)]:
        sub = ms if idx is None else ms[:, idx]
        v = compute_v4a(sub, pre_zscore=False, n_pcs=1)
        rp = float(sp.spearmanr(v[idx_e], fate_plur_e).statistic)
        rb = float(sp.spearmanr(v[idx_e], fate_bin_e).statistic)
        robust_results.append({'axis': 'module_subset', 'config': name,
                               'rho_plur': rp, 'rho_bin': rb})
        print(f'  {name:<25}  ρ_plur={rp:+.4f}  ρ_bin={rb:+.4f}')

    # B) Pre-z-score yes/no
    print('\n(B) Pre-z-score modules before PCA:')
    for name, pz in [('no_zscore', False), ('z_scored', True)]:
        v = compute_v4a(ms, pre_zscore=pz, n_pcs=1)
        rp = float(sp.spearmanr(v[idx_e], fate_plur_e).statistic)
        rb = float(sp.spearmanr(v[idx_e], fate_bin_e).statistic)
        robust_results.append({'axis': 'z_score', 'config': name,
                               'rho_plur': rp, 'rho_bin': rb})
        print(f'  {name:<15}  ρ_plur={rp:+.4f}  ρ_bin={rb:+.4f}')

    # C) Number of PCs combined
    print('\n(C) Number of PCs (weighted sum of top-n):')
    for n_pcs in [1, 2, 3]:
        v = compute_v4a(ms, pre_zscore=False, n_pcs=n_pcs)
        rp = float(sp.spearmanr(v[idx_e], fate_plur_e).statistic)
        rb = float(sp.spearmanr(v[idx_e], fate_bin_e).statistic)
        robust_results.append({'axis': 'n_pcs', 'config': f'top_{n_pcs}',
                               'rho_plur': rp, 'rho_bin': rb})
        print(f'  top_{n_pcs}  ρ_plur={rp:+.4f}  ρ_bin={rb:+.4f}')

    # D) K_NN for fate proxy (inherited from v1 — reuse)
    print('\n(D) Fate proxy K_NN:')
    for k in [5, 10, 20, 50]:
        nn_k = NearestNeighbors(n_neighbors=k).fit(E_d18)
        _, ii_k = nn_k.kneighbors(E_e)
        fp_k = plur_d18_lab[ii_k].mean(axis=1)
        fb_k = is_2i_d18_lab[ii_k].mean(axis=1)
        rp = float(sp.spearmanr(v4a_default[idx_e], fp_k).statistic)
        rb = float(sp.spearmanr(v4a_default[idx_e], fb_k).statistic)
        robust_results.append({'axis': 'K_NN', 'config': f'k={k}',
                               'rho_plur': rp, 'rho_bin': rb})
        print(f'  k={k}  ρ_plur={rp:+.4f}  ρ_bin={rb:+.4f}')

    all_rho_plur = [r['rho_plur'] for r in robust_results]
    all_rho_bin = [r['rho_bin'] for r in robust_results]
    range_plur = max(all_rho_plur) - min(all_rho_plur)
    range_bin = max(all_rho_bin) - min(all_rho_bin)
    print(f'\nρ_plur range: [{min(all_rho_plur):+.3f}, {max(all_rho_plur):+.3f}]  = {range_plur:.3f}')
    print(f'ρ_bin  range: [{min(all_rho_bin):+.3f}, {max(all_rho_bin):+.3f}]  = {range_bin:.3f}')
    robust_plur_pass = range_plur < 0.15
    robust_bin_pass = range_bin < 0.15
    print(f'Robustness pass (<0.15 for both pluripotency and binary)?  '
          f'plur={robust_plur_pass}  bin={robust_bin_pass}')

    # ============================================================
    # STEP 2: METHODOLOGY VARIANCE (C2-analog for V4a)
    # ============================================================
    print('\n===== STEP 2: METHODOLOGY VARIANCE CHECK =====')
    print(f'  shuffle day-18 pluripotency scores, n={N_PERM_C2} shuffles')
    rng = np.random.default_rng(RNG_SEED + 42)
    plur_null_rhos = np.empty(N_PERM_C2)
    # binary label shuffle for bin fate
    bin_null_rhos = np.empty(N_PERM_C2)
    for i in range(N_PERM_C2):
        plur_shuf = rng.permutation(plur_d18_lab)
        is2i_shuf = rng.permutation(is_2i_d18_lab)
        fp_null = plur_shuf[ii_e].mean(axis=1)
        fb_null = is2i_shuf[ii_e].mean(axis=1)
        plur_null_rhos[i] = sp.spearmanr(v4a_default[idx_e], fp_null).statistic
        bin_null_rhos[i] = sp.spearmanr(v4a_default[idx_e], fb_null).statistic
        if (i + 1) % 20 == 0:
            print(f'    {i+1}/{N_PERM_C2}', flush=True)

    mean_null_plur = float(plur_null_rhos.mean())
    std_null_plur = float(plur_null_rhos.std())
    ci_plur = (float(np.percentile(plur_null_rhos, 2.5)),
               float(np.percentile(plur_null_rhos, 97.5)))
    mean_null_bin = float(bin_null_rhos.mean())
    std_null_bin = float(bin_null_rhos.std())
    ci_bin = (float(np.percentile(bin_null_rhos, 2.5)),
              float(np.percentile(bin_null_rhos, 97.5)))

    # Z-score of observed vs null
    z_plur = (rho_default_plur - mean_null_plur) / (std_null_plur + 1e-12)
    z_bin = (rho_default_bin - mean_null_bin) / (std_null_bin + 1e-12)
    p_one_plur = float((plur_null_rhos <= rho_default_plur).mean())  # one-sided left tail
    p_one_bin = float((bin_null_rhos <= rho_default_bin).mean())

    print(f'\nObserved V4a ρ_plur = {rho_default_plur:+.4f}')
    print(f'  null mean={mean_null_plur:+.4f}  std={std_null_plur:.4f}  '
          f'CI [{ci_plur[0]:+.4f}, {ci_plur[1]:+.4f}]')
    print(f'  z={z_plur:.2f}  one-sided p (ρ ≤ obs) = {p_one_plur:.3f}')

    print(f'\nObserved V4a ρ_bin  = {rho_default_bin:+.4f}')
    print(f'  null mean={mean_null_bin:+.4f}  std={std_null_bin:.4f}  '
          f'CI [{ci_bin[0]:+.4f}, {ci_bin[1]:+.4f}]')
    print(f'  z={z_bin:.2f}  one-sided p (ρ ≤ obs) = {p_one_bin:.3f}')

    method_noise_plur = std_null_plur
    method_noise_bin = std_null_bin

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    df = pd.DataFrame(robust_results)
    for axis, style in [('module_subset', 'o'), ('z_score', 's'),
                        ('n_pcs', '^'), ('K_NN', 'd')]:
        sub = df[df['axis'] == axis]
        ax.scatter(sub['rho_plur'], sub['rho_bin'], marker=style,
                   s=60, alpha=0.8, label=axis)
    ax.axhline(rho_default_bin, color='grey', linestyle=':', alpha=0.5)
    ax.axvline(rho_default_plur, color='grey', linestyle=':', alpha=0.5)
    ax.scatter(rho_default_plur, rho_default_bin, marker='*',
               s=200, color='#d62728', label='default', zorder=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('ρ vs pluripotency'); ax.set_ylabel('ρ vs binary 2i')
    ax.set_title('Step 1: V4a robustness')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(plur_null_rhos, bins=30, color='#999999', alpha=0.7, label='null')
    ax.axvline(rho_default_plur, color='#d62728', linewidth=2, label=f'obs={rho_default_plur:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('ρ (plur)'); ax.set_ylabel('count')
    ax.set_title(f'Step 2: methodology null (plur)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.hist(bin_null_rhos, bins=30, color='#ff7f0e', alpha=0.7, label='null')
    ax.axvline(rho_default_bin, color='#d62728', linewidth=2, label=f'obs={rho_default_bin:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('ρ (bin)'); ax.set_ylabel('count')
    ax.set_title('Step 2: methodology null (bin)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_v4a_checks.pdf'))
    plt.close(fig)

    # ============================================================
    # Summary + decision
    # ============================================================
    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'baseline': {'rho_plur': rho_default_plur, 'rho_bin': rho_default_bin},
        'step1_robustness': {
            'results': robust_results,
            'range_plur': float(range_plur),
            'range_bin': float(range_bin),
            'threshold': 0.15,
            'plur_pass': bool(robust_plur_pass),
            'bin_pass': bool(robust_bin_pass),
        },
        'step2_methodology_null': {
            'plur': {
                'null_mean': mean_null_plur, 'null_std': std_null_plur,
                'null_ci_95': ci_plur, 'z_observed': float(z_plur),
                'one_sided_p': p_one_plur,
            },
            'bin': {
                'null_mean': mean_null_bin, 'null_std': std_null_bin,
                'null_ci_95': ci_bin, 'z_observed': float(z_bin),
                'one_sided_p': p_one_bin,
            },
        },
    }
    out = os.path.join(HERE, 'biology_schiebinger_v4a_checks_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')

    # ============================================================
    # Gate decision for Bastidas-Ponce
    # ============================================================
    print('\n===== GATE DECISION =====')
    print(f'Step 1 robustness: plur range={range_plur:.3f} ({"PASS" if robust_plur_pass else "FAIL"}), '
          f'bin range={range_bin:.3f} ({"PASS" if robust_bin_pass else "FAIL"})')
    print(f'Step 2 methodology: plur z={z_plur:.2f}, p={p_one_plur:.3f}; '
          f'bin z={z_bin:.2f}, p={p_one_bin:.3f}')

    step1_ok = robust_plur_pass and robust_bin_pass
    step2_ok_plur = p_one_plur < 0.10  # observed in lower 10% of null
    step2_ok_bin = p_one_bin < 0.10
    step2_ok = step2_ok_plur and step2_ok_bin
    gate = step1_ok and step2_ok

    print(f'\n>>> Gate to Bastidas-Ponce: {"OPEN" if gate else "HOLD"} <<<')
    if not gate:
        if not step1_ok: print('    Step 1 failed: V4a parameter-sensitive, need reformulation')
        if not step2_ok: print('    Step 2 failed: observed ρ within methodology null range')
    else:
        print('    Both checks passed. V4a methodology robust enough for replication test.')


if __name__ == '__main__':
    main()
