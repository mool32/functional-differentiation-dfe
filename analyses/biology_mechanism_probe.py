"""Path II — mechanism probe.

Does the developmental ρ ≈ +0.6 come from:
  (a) Housekeeping/infrastructure modules (tautology-adjacent)
  (b) Specific developmental signaling modules (substantive finding)
  (c) Mix

Approach:
1. Top-module overlap across four rankings (dev PR × 2, perturbability × 2)
2. Module functional categories (nuclear receptors, stress, growth,
   dev signaling, immune, second messenger, sensors, other)
3. Leave-category-out: remove each category's modules, recompute
   correlation, see which category's removal kills signal
4. Correlation between biology measurement and Paper 3 MII
   (Module Importance Index) — if high, our finding just recapitulates MII
"""
import json, os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')

P6 = '/Users/teo/Desktop/research/perceptual_modules/paper6/results'
SCHIEB_H5 = f'{P6}/schiebinger_scored.h5ad'
BP_H5 = f'{P6}/bastidas_ponce_scored.h5ad'
DIXIT = f'{P6}/Dixit2016_perturbability.csv'
REPLOGLE = f'{P6}/Replogle2022_perturbability_corrected.csv'

DAY_EARLY_SCHIEB = 2.5
DAY_EARLY_BP = 12.5

# Module functional categories (from Paper 3 manuscript)
CATEGORIES = {
    'nuclear_receptor': ['AR', 'ER', 'GR', 'MR', 'PR', 'VDR', 'TR',
                          'PPARα', 'PPARγ', 'FXR', 'LXR', 'PXR/CAR', 'RAR', 'AhR'],
    'stress_infrastructure': ['HSF1', 'UPR-ATF6', 'UPR-PERK', 'UPR-IRE1',
                               'Autophagy', 'NRF2', 'p53'],
    'growth_proliferation': ['Cell Cycle', 'mTOR', 'ERK/MAPK', 'PI3K/PTEN', 'AMPK'],
    'dev_signaling': ['BMP', 'Wnt', 'Notch', 'Hedgehog', 'TGF-β', 'Hippo'],
    'immune_cytokine': ['NF-κB', 'JAK-STAT', 'Type I IFN'],
    'second_messenger': ['Calcium', 'NFAT', 'cAMP/CREB'],
    'sensors': ['HIF', 'cGAS-STING', 'Insulin/FOXO'],
    'other': ['Circadian', 'SREBP'],
}

# Paper 3 MII values (from paper3_manuscript.md). Approximate, published values.
# Top: NF-kB=0.89, ERK/MAPK=0.86, JAK-STAT=0.76, Cell Cycle=0.76, PI3K/PTEN=0.72
# Bottom: FXR=0.16, PXR/CAR=0.19, TR=0.20, MR=0.22
# These are the only published values; interpolate plausibly for others not listed
# Actually — skip MII recreation; too risky without full data. Do category analysis instead.


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def pr_across_cells(M_2d):
    M = np.asarray(M_2d, dtype=np.float64)
    s1 = M.sum(axis=0); s2 = (M ** 2).sum(axis=0)
    n = M.shape[0]
    d = n * s2
    out = np.full(M.shape[1], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def main():
    print('[mechanism] loading...', flush=True)
    dx = pd.read_csv(DIXIT, index_col=0)
    rp = pd.read_csv(REPLOGLE, index_col=0)

    a_sc = ad.read_h5ad(SCHIEB_H5)
    ms_sc = np.asarray(a_sc.obsm['module_scores'])
    days_sc = a_sc.obs['day'].astype(float).values
    mnames_sc = list(a_sc.uns.get('module_names', []))
    mask_sc = days_sc == DAY_EARLY_SCHIEB
    pr_sc = pr_across_cells(ms_sc[mask_sc])
    pr_sc_series = pd.Series(pr_sc, index=mnames_sc)

    a_bp = ad.read_h5ad(BP_H5)
    ms_bp = np.asarray(a_bp.obsm['module_scores'])
    days_bp = pd.to_numeric(a_bp.obs['day'].astype(str), errors='coerce').values
    mnames_bp = list(a_bp.uns.get('module_names', []))
    mask_bp = days_bp == DAY_EARLY_BP
    pr_bp = pr_across_cells(ms_bp[mask_bp])
    pr_bp_series = pd.Series(pr_bp, index=mnames_bp)

    # Common modules across all four measurements
    common = pr_sc_series.index.intersection(pr_bp_series.index)
    common = common.intersection(dx.index)
    common = common.intersection(rp.index)
    common = sorted(common)
    print(f'  common modules across 4 measurements: {len(common)}')

    # Build joint table
    tbl = pd.DataFrame({
        'PR_schieb_d2.5': pr_sc_series.loc[common],
        'PR_bp_E12.5': pr_bp_series.loc[common],
        'pert_dixit': dx.loc[common, 'perturbability'],
        'pert_replogle': rp.loc[common, 'perturbability'],
    })
    # Category assignment
    module_to_cat = {}
    for cat, mods in CATEGORIES.items():
        for m in mods:
            module_to_cat[m] = cat
    tbl['category'] = [module_to_cat.get(m, 'unassigned') for m in tbl.index]
    unassigned = tbl[tbl['category'] == 'unassigned'].index.tolist()
    if unassigned:
        print(f'  WARNING unassigned modules: {unassigned}')

    tbl.to_csv(os.path.join(TABLES, 'biology_mechanism_probe_table.csv'))

    # ---- Top-15 ranked by each measurement ----
    print('\n=== TOP-15 MODULES BY EACH MEASUREMENT ===')
    for col in ['PR_schieb_d2.5', 'PR_bp_E12.5', 'pert_dixit', 'pert_replogle']:
        top = tbl[col].sort_values(ascending=False).head(15)
        print(f'\n{col}:')
        for m, v in top.items():
            cat = tbl.loc[m, 'category']
            print(f'  {m:<18} {v:>9.4f}   [{cat}]')

    # ---- Consensus top modules (top-10 in any of the 4) ----
    top10_union = set()
    top10_sets = {}
    for col in ['PR_schieb_d2.5', 'PR_bp_E12.5', 'pert_dixit', 'pert_replogle']:
        s = set(tbl[col].sort_values(ascending=False).head(10).index)
        top10_sets[col] = s
        top10_union.update(s)

    # For each module in union, how many top-10 lists it's in
    top_counts = pd.Series({m: sum(m in s for s in top10_sets.values()) for m in top10_union})
    print(f'\n=== CONSENSUS MODULES (in top-10 of ≥3 of 4 rankings) ===')
    consensus = top_counts[top_counts >= 3].sort_values(ascending=False)
    for m in consensus.index:
        cat = module_to_cat.get(m, 'unassigned')
        print(f'  {m:<18}  in {consensus[m]}/4 top-10 lists   [{cat}]')

    # ---- Leave-category-out analysis ----
    print('\n=== LEAVE-CATEGORY-OUT: remove each category, recompute correlations ===')
    baseline_correlations = {}
    for dev_col in ['PR_schieb_d2.5', 'PR_bp_E12.5']:
        for pert_col in ['pert_dixit', 'pert_replogle']:
            rho_full = sp.spearmanr(tbl[dev_col], tbl[pert_col]).statistic
            baseline_correlations[f'{dev_col}__{pert_col}'] = float(rho_full)
    print('Baseline (all modules):')
    for k, v in baseline_correlations.items():
        print(f'  {k}: {v:+.4f}')

    cat_results = {}
    print('\nLeaving out each category:')
    print(f'{"category":<25} {"n_removed":>10} {"n_left":>7}   {"Schieb×Dixit":>12} {"Schieb×Repl":>12} {"BP×Dixit":>12} {"BP×Repl":>12}')
    for cat in list(CATEGORIES.keys()):
        mods_to_remove = [m for m in tbl.index if tbl.loc[m, 'category'] == cat]
        tbl_sub = tbl.drop(mods_to_remove)
        cr = {}
        for dev_col in ['PR_schieb_d2.5', 'PR_bp_E12.5']:
            for pert_col in ['pert_dixit', 'pert_replogle']:
                rho = sp.spearmanr(tbl_sub[dev_col], tbl_sub[pert_col]).statistic
                cr[f'{dev_col}__{pert_col}'] = float(rho)
        cat_results[cat] = {'n_removed': len(mods_to_remove),
                             'n_left': len(tbl_sub),
                             'correlations': cr}
        print(f'{cat:<25} {len(mods_to_remove):>10} {len(tbl_sub):>7}   '
              f'{cr["PR_schieb_d2.5__pert_dixit"]:>+12.3f} '
              f'{cr["PR_schieb_d2.5__pert_replogle"]:>+12.3f} '
              f'{cr["PR_bp_E12.5__pert_dixit"]:>+12.3f} '
              f'{cr["PR_bp_E12.5__pert_replogle"]:>+12.3f}')

    # ---- Keep-only-one-category ----
    print('\n=== KEEP-ONE-CATEGORY: retain only that category ===')
    print(f'{"category":<25} {"n":>5}   {"Schieb×Dixit":>12} {"Schieb×Repl":>12} {"BP×Dixit":>12} {"BP×Repl":>12}')
    keep_results = {}
    for cat in list(CATEGORIES.keys()):
        mods = [m for m in tbl.index if tbl.loc[m, 'category'] == cat]
        if len(mods) < 3:
            print(f'{cat:<25} {len(mods):>5}   (too few for correlation)')
            continue
        tbl_sub = tbl.loc[mods]
        cr = {}
        for dev_col in ['PR_schieb_d2.5', 'PR_bp_E12.5']:
            for pert_col in ['pert_dixit', 'pert_replogle']:
                rho = sp.spearmanr(tbl_sub[dev_col], tbl_sub[pert_col]).statistic
                cr[f'{dev_col}__{pert_col}'] = float(rho) if np.isfinite(rho) else None
        keep_results[cat] = {'n': len(mods), 'correlations': cr}
        fmt = lambda v: f'{v:>+12.3f}' if v is not None else f'{"n.d.":>12}'
        print(f'{cat:<25} {len(mods):>5}   '
              f'{fmt(cr["PR_schieb_d2.5__pert_dixit"])} '
              f'{fmt(cr["PR_schieb_d2.5__pert_replogle"])} '
              f'{fmt(cr["PR_bp_E12.5__pert_dixit"])} '
              f'{fmt(cr["PR_bp_E12.5__pert_replogle"])}')

    # ---- Mechanistic diagnosis ----
    # Sum |Δ rho| when each category removed — biggest drop = most responsible
    print('\n=== CATEGORY RESPONSIBILITY: mean |baseline_rho - leave_out_rho| ===')
    print('(larger value = this category drives more of the correlation)')
    resp = {}
    for cat in cat_results:
        dr = []
        for k, base_rho in baseline_correlations.items():
            left_rho = cat_results[cat]['correlations'][k]
            dr.append(abs(base_rho - left_rho))
        resp[cat] = {
            'mean_abs_drho': float(np.mean(dr)),
            'n_removed': cat_results[cat]['n_removed'],
        }
    resp_df = pd.DataFrame(resp).T.sort_values('mean_abs_drho', ascending=False)
    for cat, row in resp_df.iterrows():
        print(f'  {cat:<25}  n_removed={int(row["n_removed"]):>2}  mean |Δρ|={row["mean_abs_drho"]:.4f}')

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) Schieb d2.5 × Replogle
    ax = axes[0]
    cats_unique = sorted(tbl['category'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(cats_unique)))
    for cat, c in zip(cats_unique, colors):
        sub = tbl[tbl['category'] == cat]
        ax.scatter(sub['PR_schieb_d2.5'], sub['pert_replogle'], s=80, color=c,
                   label=cat, alpha=0.8, edgecolor='k', linewidth=0.5)
    for m in tbl.index:
        ax.annotate(m, (tbl.loc[m, 'PR_schieb_d2.5'], tbl.loc[m, 'pert_replogle']),
                    fontsize=6, xytext=(2, 2), textcoords='offset points', alpha=0.7)
    r0 = sp.spearmanr(tbl['PR_schieb_d2.5'], tbl['pert_replogle']).statistic
    ax.set_title(f'Schieb d2.5 × Replogle (ρ={r0:+.3f})')
    ax.set_xlabel('PR (activity across cells)'); ax.set_ylabel('perturbability')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc='lower right')

    # (b) responsibility bar chart
    ax = axes[1]
    resp_df_plot = resp_df.reset_index().rename(columns={'index': 'category'})
    ax.barh(resp_df_plot['category'], resp_df_plot['mean_abs_drho'], color='#1f77b4')
    ax.set_xlabel('mean |Δρ| when category removed')
    ax.set_title('Category responsibility for developmental×pert correlation')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_mechanism_probe.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'n_modules_common': len(common),
        'baseline_correlations': baseline_correlations,
        'consensus_top_modules': {m: int(top_counts[m]) for m in consensus.index},
        'top10_per_measurement': {k: list(v) for k, v in top10_sets.items()},
        'leave_category_out': cat_results,
        'keep_only_category': keep_results,
        'category_responsibility': {k: v for k, v in resp.items()},
    }
    out = os.path.join(HERE, 'biology_mechanism_probe_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
