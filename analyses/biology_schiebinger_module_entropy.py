"""Schiebinger module-entropy test — biology pre-registration v2.

Pre-registration: biology_preregistration_v2_module_entropy.md (locked before run).

Replaces (Σx)²/(n·Σx²) HVG-based PR with softmax Shannon entropy of
perceptome 43-module activity. Principled (all modules used, no HVG choice),
tests whether the v1 robustness failure was due to operational-definition
fragility rather than absence of a biology analog.

Outputs:
  tables/biology_schiebinger_module_entropy.csv
  analyses/biology_schiebinger_module_entropy_summary.json
  figures/biology_schiebinger_module_entropy_{primary,phase,robust}.pdf
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

N_HVG_FOR_PCA = 2000
N_PCS = 50
K_NN = 10
RNG_SEED = 20260424

PLURIPOTENCY_MARKERS = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4',
                        'Esrrb', 'Tfcp2l1', 'Tbx3']
DAY_LATE = 18.0
DAY_EARLY_LO, DAY_EARLY_HI = 2.0, 3.0
DAY_NULL = 0.0
PHASE_TIMEPOINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def sparse_variance(X):
    if hasattr(X, 'multiply'):
        mean = np.asarray(X.mean(axis=0)).flatten()
        Xsq = X.multiply(X)
        meansq = np.asarray(Xsq.mean(axis=0)).flatten()
        return meansq - mean ** 2
    return np.asarray(X).var(axis=0)


def module_entropy(module_scores, beta=1.0):
    """Softmax entropy per cell (nats). beta = inverse temperature."""
    s = np.asarray(module_scores, dtype=np.float64) * beta
    s_stable = s - s.max(axis=1, keepdims=True)
    p = np.exp(s_stable)
    p /= p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-20, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def pr_gene_vec(X_hvg):
    X = X_hvg.astype(np.float64)
    s1 = X.sum(axis=1); s2 = (X ** 2).sum(axis=1)
    n = X.shape[1]
    d = n * s2
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def spearman_ci(x, y, n_boot=10_000, seed=RNG_SEED):
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


def decision_tier(r):
    if r is None: return 'NO_DATA'
    rho, p = r['rho'], r['p']
    if rho >= 0: return 'FAIL_WRONG_SIGN'
    mag = abs(rho)
    ci_excl0 = r['ci_lo'] < 0 and r['ci_hi'] < 0
    if mag >= 0.20 and p < 0.05 and ci_excl0: return 'PASS'
    if mag >= 0.10: return 'PARTIAL'
    if mag >= 0.05: return 'WEAK'
    return 'FAIL'


def main():
    print('[module_ent] loading Schiebinger...', flush=True)
    a = ad.read_h5ad(SCHIEBINGER)

    # ---- Module scores check ----
    assert 'module_scores' in a.obsm.keys(), 'missing obsm[module_scores]'
    ms = a.obsm['module_scores']
    print(f'  module_scores shape: {ms.shape}', flush=True)
    print(f'  module_scores stats: min={ms.min():.3f}  max={ms.max():.3f}  '
          f'mean={ms.mean():.3f}  std={ms.std():.3f}', flush=True)
    n_modules = ms.shape[1]
    mnames = a.uns.get('module_names', [f'mod{i}' for i in range(n_modules)])
    print(f'  {n_modules} modules', flush=True)

    # ---- Compute module entropy per cell ----
    ent = module_entropy(ms, beta=1.0)
    a.obs['module_entropy'] = ent
    print(f'  module_entropy: mean={ent.mean():.4f}  std={ent.std():.4f}  '
          f'min={ent.min():.4f}  max={ent.max():.4f}  (log(43)={np.log(43):.4f})', flush=True)

    # ---- Build pluripotency fate proxy (same methodology as v1) ----
    varp = sparse_variance(a.X)
    if 'highly_variable' in a.var.columns:
        hv_mask = a.var['highly_variable'].values
        hv_idx = np.where(hv_mask)[0]
        if len(hv_idx) >= N_HVG_FOR_PCA:
            top = hv_idx[np.argsort(varp[hv_idx])[::-1][:N_HVG_FOR_PCA]]
        else:
            top = np.argsort(varp)[::-1][:N_HVG_FOR_PCA]
    else:
        top = np.argsort(varp)[::-1][:N_HVG_FOR_PCA]
    hvg = np.sort(top)

    markers_found = [g for g in PLURIPOTENCY_MARKERS if g in a.var_names]
    marker_idx = [list(a.var_names).index(g) for g in markers_found]
    X_mk = _dense(a.X[:, marker_idx])
    mu = X_mk.mean(axis=0); sd = X_mk.std(axis=0) + 1e-9
    plur = ((X_mk - mu) / sd).mean(axis=1)

    # Binary labels (B1 outcome)
    with open(ID_2I) as f: ids_2i = set(l.strip() for l in f if l.strip())
    with open(ID_SERUM) as f: ids_s = set(l.strip() for l in f if l.strip())
    only_2i = np.asarray([c in ids_2i and c not in ids_s for c in a.obs_names])
    only_serum = np.asarray([c in ids_s and c not in ids_2i for c in a.obs_names])

    mask_d18 = (a.obs.day == DAY_LATE).values
    mask_d18_labeled = mask_d18 & (only_2i | only_serum)
    X_d18_hvg = _dense(a.X[:, hvg])[mask_d18_labeled]
    plur_d18 = plur[mask_d18]  # all day 18 (for pluripotency proxy)
    plur_d18_lab = plur[mask_d18_labeled]  # labeled subset
    is_2i_d18_lab = only_2i[mask_d18_labeled].astype(float)

    # PCA + NN on HVG day 18 labeled subset (same as v1-binary methodology)
    pca = PCA(n_components=N_PCS, random_state=RNG_SEED).fit(X_d18_hvg)
    E_d18 = pca.transform(X_d18_hvg)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(E_d18)

    def fate_for_indices(indices):
        X_q = _dense(a.X[indices, :][:, hvg])
        E_q = pca.transform(X_q)
        _, ii = nn.kneighbors(E_q)
        return plur_d18_lab[ii].mean(axis=1), is_2i_d18_lab[ii].mean(axis=1)

    # ---- H1 PRIMARY day 2-3 ----
    mask_e = ((a.obs.day >= DAY_EARLY_LO) & (a.obs.day <= DAY_EARLY_HI)).values
    idx_e = np.where(mask_e)[0]
    ent_e = ent[idx_e]
    fate_plur_e, fate_bin_e = fate_for_indices(idx_e)

    print('\n=== H1 PRIMARY: module_entropy at day 2-3 vs pluripotency fate ===')
    r_h1 = spearman_ci(ent_e, fate_plur_e)
    print(f'  rho = {r_h1["rho"]:+.4f}  CI [{r_h1["ci_lo"]:+.4f}, {r_h1["ci_hi"]:+.4f}]  p={r_h1["p"]:.2e}  n={r_h1["n"]}')
    v_h1 = decision_tier(r_h1)
    print(f'  verdict: {v_h1}')

    print('\n=== H1\' PRIMARY: module_entropy at day 2-3 vs binary 2i fraction ===')
    r_h1p = spearman_ci(ent_e, fate_bin_e)
    print(f'  rho = {r_h1p["rho"]:+.4f}  CI [{r_h1p["ci_lo"]:+.4f}, {r_h1p["ci_hi"]:+.4f}]  p={r_h1p["p"]:.2e}')
    v_h1p = decision_tier(r_h1p)
    print(f'  verdict: {v_h1p}')

    # ---- H0 NULL day 0 ----
    print('\n=== H0 NULL: day 0 ===')
    mask_d0 = (a.obs.day == DAY_NULL).values
    idx_d0 = np.where(mask_d0)[0]
    ent_d0 = ent[idx_d0]
    fate_plur_d0, fate_bin_d0 = fate_for_indices(idx_d0)
    r_h0 = spearman_ci(ent_d0, fate_plur_d0)
    r_h0b = spearman_ci(ent_d0, fate_bin_d0)
    print(f'  pluripotency-based: rho={r_h0["rho"]:+.4f}  n={r_h0["n"]}')
    print(f'  binary-based:       rho={r_h0b["rho"]:+.4f}  n={r_h0b["n"]}')
    h0_pass = abs(r_h0['rho']) < 0.15 and abs(r_h0b['rho']) < 0.15
    print(f'  H0 pass (both |rho|<0.15): {h0_pass}')

    # ---- H3 TRAJECTORY ----
    print('\n=== H3 TRAJECTORY ===')
    traj = {}
    for t in PHASE_TIMEPOINTS:
        mask_t = (a.obs.day == t).values
        idx_t = np.where(mask_t)[0]
        if len(idx_t) < 50: continue
        ent_t = ent[idx_t]
        fate_plur_t, fate_bin_t = fate_for_indices(idx_t)
        r_plur = sp.spearmanr(ent_t, fate_plur_t)
        r_bin = sp.spearmanr(ent_t, fate_bin_t)
        traj[float(t)] = {
            'n': int(len(idx_t)),
            'rho_plur': float(r_plur.statistic), 'p_plur': float(r_plur.pvalue),
            'rho_bin': float(r_bin.statistic), 'p_bin': float(r_bin.pvalue),
        }
        print(f'  day {t:>5.1f}  n={len(idx_t):>5}  '
              f'rho_plur={r_plur.statistic:+.3f}  rho_bin={r_bin.statistic:+.3f}')

    # ---- Robustness: softmax beta + K_NN ----
    print('\n=== ROBUSTNESS: softmax beta ===')
    beta_results = {}
    for beta in [0.5, 1.0, 2.0, 4.0]:
        ent_b = module_entropy(ms, beta=beta)
        rho_b = sp.spearmanr(ent_b[idx_e], fate_plur_e).statistic
        rho_b_bin = sp.spearmanr(ent_b[idx_e], fate_bin_e).statistic
        beta_results[beta] = {'rho_plur': float(rho_b), 'rho_bin': float(rho_b_bin)}
        print(f'  beta={beta}:  rho_plur={rho_b:+.4f}  rho_bin={rho_b_bin:+.4f}')

    print('\n=== ROBUSTNESS: K_NN ===')
    knn_results = {}
    for k in [5, 10, 20, 50]:
        nn_k = NearestNeighbors(n_neighbors=k).fit(E_d18)
        X_q = _dense(a.X[idx_e, :][:, hvg])
        E_q = pca.transform(X_q)
        _, iik = nn_k.kneighbors(E_q)
        fate_p_k = plur_d18_lab[iik].mean(axis=1)
        fate_b_k = is_2i_d18_lab[iik].mean(axis=1)
        rho_kp = sp.spearmanr(ent_e, fate_p_k).statistic
        rho_kb = sp.spearmanr(ent_e, fate_b_k).statistic
        knn_results[k] = {'rho_plur': float(rho_kp), 'rho_bin': float(rho_kb)}
        print(f'  k={k}:  rho_plur={rho_kp:+.4f}  rho_bin={rho_kb:+.4f}')

    all_plur = [r['rho_plur'] for r in beta_results.values()] + [r['rho_plur'] for r in knn_results.values()]
    all_bin = [r['rho_bin'] for r in beta_results.values()] + [r['rho_bin'] for r in knn_results.values()]
    range_plur = max(all_plur) - min(all_plur)
    range_bin = max(all_bin) - min(all_bin)
    print(f'\nrho_plur range across robustness: [{min(all_plur):+.3f}, {max(all_plur):+.3f}] = {range_plur:.3f}')
    print(f'rho_bin  range across robustness: [{min(all_bin):+.3f}, {max(all_bin):+.3f}] = {range_bin:.3f}')
    robust_pass = range_plur < 0.15 and range_bin < 0.15
    print(f'Robustness pass (range < 0.15 for both): {robust_pass}')

    # ---- Comparison: entropy vs gene-PR (v1) on same cells ----
    print('\n=== DIAGNOSTIC: entropy vs gene-PR on day 2-3 ===')
    X_hvg_e = _dense(a.X[idx_e, :][:, hvg])
    pr_gene_e = pr_gene_vec(X_hvg_e)
    rho_ent_pr = sp.spearmanr(ent_e, pr_gene_e).statistic
    print(f'  rho(module_entropy, gene_PR) on same day 2-3 cells: {rho_ent_pr:+.4f}')
    if abs(rho_ent_pr) > 0.8:
        diag = 'Nearly same measurement (entropy and PR capture same structure)'
    elif abs(rho_ent_pr) > 0.3:
        diag = 'Moderately related measurements'
    else:
        diag = 'Largely independent measurements'
    print(f'  diagnosis: {diag}')

    # ---- Plots ----
    # Primary scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(ent_e, fate_plur_e, s=3, alpha=0.2, color='#d62728')
    ax.set_xlabel('module entropy at day 2-3')
    ax.set_ylabel('fate proxy (pluripotency, NN)')
    ax.set_title(f'H1 pluripotency: ρ={r_h1["rho"]:+.3f}')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(ent_e, fate_bin_e, s=3, alpha=0.2, color='#1f77b4')
    ax.set_xlabel('module entropy at day 2-3')
    ax.set_ylabel('2i fraction among 10 day-18 NN')
    ax.set_title(f'H1\' binary: ρ={r_h1p["rho"]:+.3f}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_module_entropy_primary.pdf'))
    plt.close(fig)

    # Phase transition
    fig, ax = plt.subplots(figsize=(8, 4.5))
    days = sorted(traj.keys())
    ax.plot(days, [traj[d]['rho_plur'] for d in days], 'o-', label='pluripotency', linewidth=1.6)
    ax.plot(days, [traj[d]['rho_bin'] for d in days], 's-', label='binary 2i-fraction', linewidth=1.6)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('day')
    ax.set_ylabel(r'Spearman ρ(module_entropy, fate_proxy)')
    ax.set_title('H3 trajectory with module entropy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_module_entropy_phase.pdf'))
    plt.close(fig)

    # Robustness
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    betas = sorted(beta_results.keys())
    ax.plot(betas, [beta_results[b]['rho_plur'] for b in betas], 'o-', label='pluripotency')
    ax.plot(betas, [beta_results[b]['rho_bin'] for b in betas], 's-', label='binary')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xscale('log'); ax.set_xlabel('softmax β')
    ax.set_ylabel('ρ'); ax.set_title('β sweep'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ks = sorted(knn_results.keys())
    ax.plot(ks, [knn_results[k]['rho_plur'] for k in ks], 'o-', label='pluripotency')
    ax.plot(ks, [knn_results[k]['rho_bin'] for k in ks], 's-', label='binary')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xscale('log'); ax.set_xlabel('K_NN')
    ax.set_ylabel('ρ'); ax.set_title('K_NN sweep'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_schiebinger_module_entropy_robust.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'pre_registration': 'biology_preregistration_v2_module_entropy.md',
        'n_modules': int(n_modules),
        'H1_primary_pluripotency': {'rho': r_h1, 'verdict': v_h1},
        'H1p_primary_binary': {'rho': r_h1p, 'verdict': v_h1p},
        'H0_null': {'rho_plur': r_h0, 'rho_bin': r_h0b, 'pass': bool(h0_pass)},
        'H3_trajectory': traj,
        'robustness_beta': beta_results,
        'robustness_knn': knn_results,
        'rho_range_plur': float(range_plur),
        'rho_range_bin': float(range_bin),
        'robust_pass': bool(robust_pass),
        'diagnostic_entropy_vs_genePR_day2_3': float(rho_ent_pr),
    }
    out_json = os.path.join(HERE, 'biology_schiebinger_module_entropy_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    out_csv = os.path.join(TABLES, 'biology_schiebinger_module_entropy.csv')
    save_df = pd.DataFrame({
        'cell_id': a.obs_names[idx_e],
        'day': a.obs.day.values[idx_e],
        'module_entropy': ent_e,
        'gene_PR': pr_gene_e,
        'fate_pluripotency_NN': fate_plur_e,
        'fate_binary_2i_NN': fate_bin_e,
    })
    save_df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_json}')
    print(f'wrote {out_csv}')

    # Final verdict
    print('\n=== OVERALL VERDICT ===')
    print(f'H1 (pluripotency fate):  {v_h1}   ρ={r_h1["rho"]:+.3f}')
    print(f'H1\' (binary fate):       {v_h1p}  ρ={r_h1p["rho"]:+.3f}')
    print(f'H0 null (day 0):        {"PASS" if h0_pass else "FAIL"}   |ρ|={max(abs(r_h0["rho"]),abs(r_h0b["rho"])):.3f}')
    print(f'Robustness:             {"PASS" if robust_pass else "FAIL"}   '
          f'range_plur={range_plur:.3f} range_bin={range_bin:.3f}')
    print(f'Entropy vs gene-PR:     ρ={rho_ent_pr:+.3f}  ({diag})')


if __name__ == '__main__':
    main()
