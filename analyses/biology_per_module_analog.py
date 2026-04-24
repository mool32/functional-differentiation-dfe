"""Per-module biology test — correct ML analog unit of analysis.

Per user's reframing: ML model ≈ cell, head ≈ module.
Unit of analysis is MODULE (n=43), not cell (n=20k+).

Data (all from Paper 6 existing output):
- schiebinger_module_change_order.csv: per-module time_of_max_change, magnitude
- bastidas_ponce_module_change_order.csv: same
- Dixit2016_perturbability.csv: per-module CRISPRi perturbability (K562)
- Replogle2022_perturbability_corrected.csv: same (larger dataset)

ML analog:
  Per-head (n=144): OV_PR @step1000 → |Δ| @step143000.  Predicted: ρ < 0.

Biology analog (per-module, n=43):
  time_of_max_change (early dynamics marker) → perturbability (importance).
  Hypothesis A: modules that change EARLY during reprogramming → high perturbability.
    → NEGATIVE correlation between time_of_max_change and perturbability.
  Hypothesis B: modules with LARGER magnitude of change → more perturbable.
    → POSITIVE correlation between magnitude and perturbability.

Tests:
1. Schiebinger (reprogramming) × Dixit
2. Schiebinger × Replogle
3. Bastidas-Ponce (pancreas dev) × Dixit
4. Bastidas-Ponce × Replogle

Also: per-module PR of activity distribution across cells at early timepoint
(within-timepoint module concentration), correlate with perturbability.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats as sp
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')

P6 = '/Users/teo/Desktop/research/perceptual_modules/paper6/results'

# Per-module dynamics (from trajectory analysis)
SCHIEBINGER_CHANGE = f'{P6}/schiebinger_module_change_order.csv'
BP_CHANGE = f'{P6}/bastidas_ponce_module_change_order.csv'
# Per-module perturbability (from CRISPRi screens)
DIXIT = f'{P6}/Dixit2016_perturbability.csv'
REPLOGLE = f'{P6}/Replogle2022_perturbability_corrected.csv'

# For complementary within-timepoint test
SCHIEB_H5 = f'{P6}/schiebinger_scored.h5ad'
BP_H5 = f'{P6}/bastidas_ponce_scored.h5ad'

DAY_EARLY_SCHIEB = 2.5  # early specification window
DAY_EARLY_BP = 12.5
RNG_SEED = 20260424


def spearman_ci(x, y, n_boot=5000, seed=RNG_SEED):
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


def participation_ratio_across_cells(M_2d):
    """PR formula on activity distribution across cells, per module.
    M_2d: (n_cells, 43) matrix. Returns (43,) array of per-module PR values.
    PR_m = (Σ_cells m)² / (n_cells * Σ_cells m²)
    Low PR = module activity concentrated in few cells (specialized).
    """
    M = np.asarray(M_2d, dtype=np.float64)
    s1 = M.sum(axis=0)
    s2 = (M ** 2).sum(axis=0)
    n = M.shape[0]
    d = n * s2
    out = np.full(M.shape[1], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def main():
    print('[per-module] loading data...', flush=True)
    sc = pd.read_csv(SCHIEBINGER_CHANGE, index_col=0)
    bp = pd.read_csv(BP_CHANGE, index_col=0)
    dx = pd.read_csv(DIXIT, index_col=0)
    rp = pd.read_csv(REPLOGLE, index_col=0)
    print(f'  Schiebinger change: {len(sc)} modules')
    print(f'  Bastidas-Ponce change: {len(bp)} modules')
    print(f'  Dixit perturbability: {len(dx)} modules')
    print(f'  Replogle perturbability: {len(rp)} modules')

    print('\n[per-module] overlap:')
    print(f'  Schieb ∩ Dixit: {len(set(sc.index) & set(dx.index))}')
    print(f'  Schieb ∩ Replogle: {len(set(sc.index) & set(rp.index))}')
    print(f'  BP ∩ Dixit: {len(set(bp.index) & set(dx.index))}')
    print(f'  BP ∩ Replogle: {len(set(bp.index) & set(rp.index))}')

    # ---- Tests 1-4: per-module correlation between dynamics and perturbability ----
    results = []
    print('\n=== PER-MODULE CORRELATION TESTS (n ≈ 43) ===')
    print(f'{"dataset":<12} {"perturb":<10} {"metric":<20} {"rho":>8} {"CI":>22} {"p":>10} {"n":>4}')
    print('-' * 100)
    for dev_name, dev_df in [('Schiebinger', sc), ('Bastidas', bp)]:
        for pert_name, pert_df in [('Dixit', dx), ('Replogle', rp)]:
            common = dev_df.index.intersection(pert_df.index)
            for dev_metric in ['time_of_max_change', 'magnitude']:
                if dev_metric not in dev_df.columns: continue
                x = dev_df.loc[common, dev_metric].values
                y = pert_df.loc[common, 'perturbability'].values
                r = spearman_ci(x, y, n_boot=2000)
                rows = {'dev_dataset': dev_name, 'pert_dataset': pert_name,
                        'dev_metric': dev_metric, **r}
                results.append(rows)
                print(f'{dev_name:<12} {pert_name:<10} {dev_metric:<20} '
                      f'{r["rho"]:+8.4f} [{r["ci_lo"]:+6.3f},{r["ci_hi"]:+6.3f}] '
                      f'{r["p"]:>10.3e} {r["n"]:>4}')

    # ---- Additional: within-timepoint PR per module across cells ----
    print('\n=== WITHIN-TIMEPOINT PR PER MODULE × PERTURBABILITY ===')
    print('For each module m, PR = concentration of its activity across cells at early timepoint.')
    print('Low PR = activity concentrated in few cells (specialized expression).')
    print('Hypothesis: specialized (low PR) early → more perturbable (important at maturity).')
    within_results = []

    # Schiebinger early
    print('\n  Loading Schiebinger h5ad...', flush=True)
    a_sc = ad.read_h5ad(SCHIEB_H5)
    ms_sc = np.asarray(a_sc.obsm['module_scores'])
    days_sc = a_sc.obs['day'].astype(float).values
    # Module names from uns
    mnames_sc = list(a_sc.uns.get('module_names', []))
    if not mnames_sc:
        mnames_sc = [f'mod{i}' for i in range(ms_sc.shape[1])]
    mask_early_sc = days_sc == DAY_EARLY_SCHIEB
    print(f'  Schiebinger day {DAY_EARLY_SCHIEB}: {mask_early_sc.sum()} cells')
    if mask_early_sc.sum() >= 50:
        pr_sc = participation_ratio_across_cells(ms_sc[mask_early_sc])
        pr_series_sc = pd.Series(pr_sc, index=mnames_sc, name='pr_early_schieb')
        for pert_name, pert_df in [('Dixit', dx), ('Replogle', rp)]:
            common = pr_series_sc.index.intersection(pert_df.index)
            x = pr_series_sc.loc[common].values
            y = pert_df.loc[common, 'perturbability'].values
            r = spearman_ci(x, y, n_boot=2000)
            print(f'  Schiebinger-{DAY_EARLY_SCHIEB} × {pert_name}: rho={r["rho"]:+.4f} '
                  f'CI [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}] p={r["p"]:.2e} n={r["n"]}')
            within_results.append({'dataset': f'Schieb_d{DAY_EARLY_SCHIEB}',
                                    'pert_dataset': pert_name, **r})

    # BP early
    print(f'\n  Loading Bastidas-Ponce h5ad...', flush=True)
    a_bp = ad.read_h5ad(BP_H5)
    ms_bp = np.asarray(a_bp.obsm['module_scores'])
    days_bp = pd.to_numeric(a_bp.obs['day'].astype(str), errors='coerce').values
    mnames_bp = list(a_bp.uns.get('module_names', []))
    if not mnames_bp:
        mnames_bp = [f'mod{i}' for i in range(ms_bp.shape[1])]
    mask_early_bp = days_bp == DAY_EARLY_BP
    print(f'  BP day {DAY_EARLY_BP}: {mask_early_bp.sum()} cells')
    if mask_early_bp.sum() >= 50:
        pr_bp = participation_ratio_across_cells(ms_bp[mask_early_bp])
        pr_series_bp = pd.Series(pr_bp, index=mnames_bp, name='pr_early_bp')
        for pert_name, pert_df in [('Dixit', dx), ('Replogle', rp)]:
            common = pr_series_bp.index.intersection(pert_df.index)
            x = pr_series_bp.loc[common].values
            y = pert_df.loc[common, 'perturbability'].values
            r = spearman_ci(x, y, n_boot=2000)
            print(f'  BP-E{DAY_EARLY_BP} × {pert_name}: rho={r["rho"]:+.4f} '
                  f'CI [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}] p={r["p"]:.2e} n={r["n"]}')
            within_results.append({'dataset': f'BP_E{DAY_EARLY_BP}',
                                   'pert_dataset': pert_name, **r})

    # ---- Plot ----
    # Main figure: 4 panels (2 datasets × 2 pert datasets) with time_of_max_change
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (dev_name, dev_df) in zip(axes[0], [('Schiebinger', sc), ('Bastidas', bp)]):
        for pert_name, pert_df, color in [('Dixit', dx, '#1f77b4'),
                                           ('Replogle', rp, '#ff7f0e')]:
            common = dev_df.index.intersection(pert_df.index)
            x = dev_df.loc[common, 'time_of_max_change'].values
            y = pert_df.loc[common, 'perturbability'].values
            r = sp.spearmanr(x, y).statistic
            ax.scatter(x, y, s=40, alpha=0.7, color=color,
                       label=f'{pert_name} ρ={r:+.2f}')
        ax.set_xlabel('time_of_max_change (developmental)')
        ax.set_ylabel('perturbability (K562 CRISPRi)')
        ax.set_title(f'{dev_name} × perturbability (time_of_max_change)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    for ax, (dev_name, dev_df) in zip(axes[1], [('Schiebinger', sc), ('Bastidas', bp)]):
        for pert_name, pert_df, color in [('Dixit', dx, '#1f77b4'),
                                           ('Replogle', rp, '#ff7f0e')]:
            common = dev_df.index.intersection(pert_df.index)
            x = dev_df.loc[common, 'magnitude'].values
            y = pert_df.loc[common, 'perturbability'].values
            r = sp.spearmanr(x, y).statistic
            ax.scatter(x, y, s=40, alpha=0.7, color=color,
                       label=f'{pert_name} ρ={r:+.2f}')
        ax.set_xlabel('magnitude_of_max_change')
        ax.set_ylabel('perturbability')
        ax.set_title(f'{dev_name} × perturbability (magnitude)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_per_module_analog.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'note': ('Per-module test (n=43) using existing Paper 6 output. ML analog: '
                 'head=module, unit of analysis=module across modules, not cells.'),
        'n_modules': int(max(len(sc), len(bp))),
        'tests_1_4_dynamics_vs_perturbability': results,
        'tests_within_timepoint_PR_vs_perturbability': within_results,
    }
    out = os.path.join(HERE, 'biology_per_module_analog_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
