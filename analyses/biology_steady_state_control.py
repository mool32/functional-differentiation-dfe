"""Steady-state control for per-module biology finding.

Critical test: does per-module PR in HEALTHY STEADY-STATE tissue correlate
with K562 perturbability as strongly as it did in developmental data?

Two mutually exclusive interpretations:

A) If steady-state ρ ≈ +0.6 (similar to developmental):
   Observed developmental correlation is NOT developmental-specific.
   It recapitulates general "modules universal in biology → modules
   universally essential" tautology-adjacent pattern.
   Cross-substrate claim fails.

B) If steady-state ρ ≈ 0 or substantially weaker (<0.3):
   Developmental signal is genuine, distinct from baseline module
   centrality. Finding has specific scientific content.

Datasets tested:
- HPA 154 cell types × 43 modules (primary healthy steady-state)
- GTEx tissue medians × 43 modules (bulk RNA-seq, complementary)

Comparison to previously-observed developmental correlations:
  Schiebinger day 2.5 × Dixit:    rho = +0.561
  Schiebinger day 2.5 × Replogle: rho = +0.679
  Bastidas E12.5 × Dixit:         rho = +0.600
  Bastidas E12.5 × Replogle:      rho = +0.688
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')

HPA_CSV = '/Users/teo/Desktop/research/perceptual_modules/power_law_test/results/phase0_baseline/activity_hpa_43.csv'
GTEX_NPZ = '/Users/teo/Desktop/research/perceptual_modules/power_law_test/data/processed/gtex.npz'
TCGA_NPZ = '/Users/teo/Desktop/research/perceptual_modules/power_law_test/data/processed/tcga.npz'
CCLE_NPZ = '/Users/teo/Desktop/research/perceptual_modules/power_law_test/data/processed/ccle.npz'

P6 = '/Users/teo/Desktop/research/perceptual_modules/paper6/results'
DIXIT = f'{P6}/Dixit2016_perturbability.csv'
REPLOGLE = f'{P6}/Replogle2022_perturbability_corrected.csv'

RNG_SEED = 20260424


def participation_ratio_per_column(M_2d):
    """PR per column (per module) over rows (cell types / samples)."""
    M = np.asarray(M_2d, dtype=np.float64)
    s1 = M.sum(axis=0)
    s2 = (M ** 2).sum(axis=0)
    n = M.shape[0]
    d = n * s2
    out = np.full(M.shape[1], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


def participation_ratio_abs(M_2d):
    """PR per column using |activity| (handles signed z-scores)."""
    M = np.abs(np.asarray(M_2d, dtype=np.float64))
    s1 = M.sum(axis=0)
    s2 = (M ** 2).sum(axis=0)
    n = M.shape[0]
    d = n * s2
    out = np.full(M.shape[1], np.nan, dtype=np.float64)
    m = d > 0
    out[m] = (s1[m] ** 2) / d[m]
    return out


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


def main():
    print('[control] loading data...', flush=True)
    dx = pd.read_csv(DIXIT, index_col=0)
    rp = pd.read_csv(REPLOGLE, index_col=0)

    # ---- HPA per-cell-type (154 × 43) ----
    hpa = pd.read_csv(HPA_CSV)
    meta_cols = ['cell_type', 'cell_class']
    mod_cols = [c for c in hpa.columns if c not in meta_cols]
    print(f'  HPA: {hpa.shape[0]} cell types × {len(mod_cols)} modules')
    print(f'  HPA modules: {mod_cols[:5]}... (first 5)', flush=True)
    hpa_mat = hpa[mod_cols].values  # (154, 43)

    # These are z-scored activities (signed). Use |activity| for PR.
    pr_hpa_signed = participation_ratio_per_column(hpa_mat)  # will be ill-defined for signed
    pr_hpa_abs = participation_ratio_abs(hpa_mat)
    pr_hpa_series_signed = pd.Series(pr_hpa_signed, index=mod_cols, name='hpa_pr_signed')
    pr_hpa_series_abs = pd.Series(pr_hpa_abs, index=mod_cols, name='hpa_pr_abs')

    # ---- GTEx / TCGA / CCLE from .npz (samples × modules) ----
    def load_npz_if_exists(path, label):
        if not os.path.exists(path):
            print(f'  {label}: file not found ({path})')
            return None
        d = np.load(path, allow_pickle=True)
        keys = list(d.keys())
        print(f'  {label} npz keys: {keys}')
        # Expected: "data" key or similar with (n_samples, 43)
        # Inspect and pick
        for k in ['data', 'module_activity', 'activity', 'X']:
            if k in keys:
                arr = d[k]
                # Find a module-names key
                for nk in ['module_names', 'modules', 'columns', 'feature_names']:
                    if nk in keys:
                        return arr, list(d[nk])
                return arr, None
        # Fallback: first 2D array
        for k in keys:
            arr = d[k]
            if arr.ndim == 2:
                print(f'  using key "{k}" shape {arr.shape}')
                return arr, None
        return None

    results_all = []

    # -------- HPA primary control --------
    print('\n=== STEADY-STATE CONTROL: HPA (154 cell types × 43 modules) ===')
    for label, pr_series in [('hpa_PR_abs', pr_hpa_series_abs),
                              ('hpa_PR_signed', pr_hpa_series_signed)]:
        for pert_name, pert_df in [('Dixit', dx), ('Replogle', rp)]:
            common = pr_series.index.intersection(pert_df.index)
            x = pr_series.loc[common].values
            y = pert_df.loc[common, 'perturbability'].values
            r = spearman_ci(x, y, n_boot=2000)
            if r is None:
                print(f'  {label} × {pert_name}: too few'); continue
            print(f'  {label:<15} × {pert_name:<10}: rho={r["rho"]:+.4f} '
                  f'CI [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}]  p={r["p"]:.2e}  n={r["n"]}')
            results_all.append({'dataset': label, 'pert': pert_name, **r})

    # -------- Try GTEx / TCGA / CCLE --------
    print('\n=== ADDITIONAL STEADY-STATE / DISEASE CONTROLS ===')
    for path, label in [(GTEX_NPZ, 'gtex'),
                         (TCGA_NPZ, 'tcga'),
                         (CCLE_NPZ, 'ccle')]:
        if not os.path.exists(path):
            print(f'  {label}: not found'); continue
        try:
            d = np.load(path, allow_pickle=True)
            keys = list(d.keys())
            print(f'  {label} keys: {keys}')
            # Guess: probably 'X' or 'data' or similar
            arr = None
            mnames = None
            for k in keys:
                v = d[k]
                if hasattr(v, 'shape') and v.ndim == 2 and v.shape[1] in (41, 43):
                    arr = v
                    print(f'    using key "{k}" shape {v.shape}')
                    break
            if arr is None:
                for k in keys:
                    v = d[k]
                    if hasattr(v, 'shape') and v.ndim == 2:
                        arr = v
                        print(f'    fallback key "{k}" shape {v.shape}')
                        break
            if arr is None:
                print(f'    no 2D array found')
                continue

            # Find column names
            for nk in ['module_names', 'modules', 'columns', 'feature_names']:
                if nk in keys:
                    mnames = [str(x) for x in d[nk]]
                    break
            if mnames is None and arr.shape[1] == len(mod_cols):
                mnames = mod_cols
                print(f'    using HPA mod_cols as column names')
            if mnames is None:
                print(f'    no column names, cannot match modules')
                continue

            pr_vec_abs = participation_ratio_abs(arr)
            pr_ser = pd.Series(pr_vec_abs, index=mnames)
            for pert_name, pert_df in [('Dixit', dx), ('Replogle', rp)]:
                common = pr_ser.index.intersection(pert_df.index)
                if len(common) < 10: continue
                x = pr_ser.loc[common].values
                y = pert_df.loc[common, 'perturbability'].values
                r = spearman_ci(x, y, n_boot=2000)
                if r is None: continue
                print(f'  {label}_PR_abs × {pert_name:<10}: rho={r["rho"]:+.4f} '
                      f'CI [{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}]  p={r["p"]:.2e}  n={r["n"]}')
                results_all.append({'dataset': f'{label}_PR_abs', 'pert': pert_name, **r})
        except Exception as e:
            print(f'  {label}: error {e}')

    # -------- Summary comparison --------
    print('\n=== COMPARISON TO DEVELOPMENTAL FINDING ===')
    print('Previously observed (developmental, per-cell PR):')
    print('  Schiebinger d2.5 × Dixit:    rho = +0.561')
    print('  Schiebinger d2.5 × Replogle: rho = +0.679')
    print('  Bastidas E12.5 × Dixit:      rho = +0.600')
    print('  Bastidas E12.5 × Replogle:   rho = +0.688')
    print('\nSteady-state (THIS RUN):')
    for r in results_all:
        print(f'  {r["dataset"]:<20} × {r["pert"]:<10}: rho={r["rho"]:+.4f}')

    # Decision
    print('\n=== VERDICT ===')
    hpa_rhos = [r['rho'] for r in results_all if 'hpa_PR_abs' in r['dataset']]
    if hpa_rhos:
        max_hpa = max(abs(x) for x in hpa_rhos)
        if max_hpa >= 0.45:
            verdict = ('STEADY-STATE MATCHES DEVELOPMENTAL — '
                       'developmental signal NOT developmental-specific. '
                       'Observed correlation recapitulates general module-centrality pattern. '
                       'Cross-substrate claim fails.')
        elif max_hpa >= 0.25:
            verdict = ('PARTIAL — steady-state shows weaker but real correlation; '
                       'developmental correlation has some developmental-specific component but also baseline.')
        else:
            verdict = ('STEADY-STATE NEAR NULL — developmental signal is genuinely specific '
                       'to developmental dynamics, not baseline module centrality.')
        print(verdict)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'steady_state_controls': results_all,
        'developmental_reference': {
            'Schiebinger_d2.5_Dixit': 0.561,
            'Schiebinger_d2.5_Replogle': 0.679,
            'Bastidas_E12.5_Dixit': 0.600,
            'Bastidas_E12.5_Replogle': 0.688,
        },
    }
    out = os.path.join(HERE, 'biology_steady_state_control_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
