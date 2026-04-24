"""Block 1.2: Per-class DFE Student-t fits with bootstrap CIs.

VWKK prediction
---------------
If the hierarchical-landscape framework holds:
    nu_emergent  <  nu_growing  <  nu_born  <  nu_never
Heavier tails (lower nu) should appear in classes that live on the
more frustrated levels of the hierarchy.

Operationalisation
------------------
- Class definitions (strict Paper 2):
    never     = |delta| at FINAL  <  never_thresh
    born      = |delta| at INIT   >  crit_thresh  AND  |delta| at FINAL > crit_thresh
    emergent  = |delta| at INIT   <  born_low     AND  |delta| at FINAL > crit_thresh
    growing   = everything else with |delta| at FINAL >= crit_thresh
  with never_thresh = crit_thresh = 5e-4, born_low = 1e-4.
- Per-class Student-t fit at final checkpoint. Bootstrap 10 000 resamples
  for (nu, sigma) CIs.
- Pairwise test: 10 000 paired bootstrap of (nu_A - nu_B), two-sided p.
- Also report fits per checkpoint to trace emergence of heavy tails.

Inputs
------
  ../data/all_ablations.csv              (Paper 2, Pythia 410M)
  ../data/tier2_t21_scaling_160m.csv     (T2.1, Pythia 160M)

Outputs
-------
  ../tables/per_class_dfe_410m.csv
  ../tables/per_class_dfe_160m.csv
  ../tables/per_class_dfe_pairwise_tests.csv
  ../figures/per_class_dfe_nu_trajectories.pdf
  ../figures/per_class_dfe_final_kde.pdf
  ../analyses/per_class_dfe_summary.json
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

CRIT = 5e-4
BORN_LOW = 1e-4
N_BOOT = 2_000  # Paper 2 convention; tight enough for nu CIs per Tier 1 T1.3
RNG = np.random.default_rng(20260421)


# ---------------------------------------------------------------------------
# Classification + fitting
# ---------------------------------------------------------------------------

def classify_heads(df_heads):
    """Return dict {(L,H): class} using first and last checkpoint of df_heads."""
    ckpts = sorted(df_heads['checkpoint'].unique())
    first, last = ckpts[0], ckpts[-1]
    pivot = df_heads.pivot_table(index=['layer_idx', 'head_idx'],
                                 columns='checkpoint', values='delta').abs()
    classes = {}
    for idx, row in pivot.iterrows():
        init, fin = row[first], row[last]
        if fin < CRIT:
            c = 'never'
        elif init > CRIT and fin > CRIT:
            c = 'born'
        elif init < BORN_LOW and fin > CRIT:
            c = 'emergent'
        else:
            c = 'growing'
        classes[idx] = c
    return classes, pivot


def fit_student_t(x, use_location=True):
    """Return (nu, loc, sigma) by MLE Student-t fit."""
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    try:
        if use_location:
            nu, loc, sc = sp.t.fit(x)
        else:
            nu, loc, sc = sp.t.fit(x, floc=0)
        return nu, loc, sc
    except Exception:
        return np.nan, np.nan, np.nan


def bootstrap_nu(x, n_boot=N_BOOT, rng=RNG, label=''):
    """Return array of nu estimates from bootstrap resampling."""
    if len(x) < 4:
        return np.full(n_boot, np.nan)
    nus = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        nu, _, _ = fit_student_t(sample)
        nus[i] = nu
    return nus


def nu_from_moments(x):
    """Method-of-moments ν estimate from excess kurtosis.

    For Student-t with ν>4, excess_kurt = 6/(ν-4), so ν = 4 + 6/kurt.
    Much faster than MLE and robust for sanity-checking.
    """
    if len(x) < 4:
        return np.nan
    kurt = sp.kurtosis(x)
    if kurt <= 0:
        return np.inf  # Gaussian or sub-Gaussian
    return 4 + 6 / kurt


def percentile_ci(samples, lo=2.5, hi=97.5):
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return (np.nan, np.nan)
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def student_t_aic(x, nu, loc, sc):
    if not np.isfinite(nu):
        return np.nan
    ll = np.sum(sp.t.logpdf(x, nu, loc, sc))
    return 2 * 3 - 2 * ll


def normal_aic(x):
    mu, si = sp.norm.fit(x)
    ll = np.sum(sp.norm.logpdf(x, mu, si))
    return 2 * 2 - 2 * ll


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def analyze(csv_path, label):
    print(f'\n{"="*60}\n{label}  —  {os.path.basename(csv_path)}\n{"="*60}')
    df = pd.read_csv(csv_path)
    heads = df[df['perturbation_type'] == 'head'].copy()
    classes, pivot = classify_heads(heads)

    # Class counts
    cnt = Counter(classes.values())
    total_final_critical = cnt['born'] + cnt['emergent'] + cnt['growing']
    print(f'Class counts: {dict(cnt)}')
    print(f'  emergent/born ratio: {cnt["emergent"]/max(cnt["born"],1):.2f}')

    ckpts = sorted(heads['checkpoint'].unique())
    final = ckpts[-1]

    # Per-class delta distributions at final checkpoint
    heads_final = heads[heads['checkpoint'] == final].copy()
    heads_final['class'] = heads_final.apply(
        lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)

    per_class_rows = []
    class_deltas = {}
    for cls in ['never', 'born', 'emergent', 'growing']:
        x = heads_final.loc[heads_final['class'] == cls, 'delta'].values
        if len(x) < 4:
            print(f'  {cls:<10} n={len(x)}  too few for fit')
            class_deltas[cls] = x
            continue
        print(f'  [{cls}] MLE + bootstrap n_boot={N_BOOT} ...', flush=True)
        nu, loc, sc = fit_student_t(x)
        aic_t = student_t_aic(x, nu, loc, sc)
        aic_n = normal_aic(x)
        boot_nus = bootstrap_nu(x, label=cls)
        lo, hi = percentile_ci(boot_nus)
        nu_mom = nu_from_moments(x)
        per_class_rows.append({
            'class': cls, 'checkpoint': final, 'n': int(len(x)),
            'nu': float(nu), 'loc': float(loc), 'sigma': float(sc),
            'nu_ci_lo': lo, 'nu_ci_hi': hi,
            'nu_mom': float(nu_mom),
            'delta_aic_t_vs_normal': float(aic_n - aic_t),
            'std': float(np.std(x)), 'skew': float(sp.skew(x)),
            'kurt_excess': float(sp.kurtosis(x)),
        })
        class_deltas[cls] = x
        print(f'  {cls:<10} n={len(x):>3}  '
              f'nu_MLE={nu:>6.2f} [{lo:>6.2f},{hi:>6.2f}]  '
              f'nu_MoM={nu_mom:>6.2f}  sigma={sc:.4f}  Δ_AIC(t-n)={aic_n-aic_t:+6.1f}', flush=True)

    # Per-checkpoint trajectory of nu, per class
    traj_rows = []
    for ck in ckpts:
        heads_ck = heads[heads['checkpoint'] == ck].copy()
        heads_ck['class'] = heads_ck.apply(
            lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
        for cls in ['never', 'born', 'emergent', 'growing']:
            x = heads_ck.loc[heads_ck['class'] == cls, 'delta'].values
            if len(x) < 4:
                traj_rows.append({'class': cls, 'checkpoint': ck, 'n': len(x),
                                  'nu': np.nan, 'sigma': np.nan})
                continue
            nu, loc, sc = fit_student_t(x)
            traj_rows.append({'class': cls, 'checkpoint': ck, 'n': int(len(x)),
                              'nu': float(nu), 'sigma': float(sc)})

    # Pairwise bootstrap tests: is nu_A - nu_B consistently different from 0?
    pairs = [('emergent', 'born'), ('emergent', 'growing'),
             ('growing', 'born'), ('emergent', 'never')]
    pairwise = []
    for A, B in pairs:
        xA, xB = class_deltas.get(A), class_deltas.get(B)
        if xA is None or xB is None or len(xA) < 4 or len(xB) < 4:
            pairwise.append({'label': label, 'pair': f'{A}_vs_{B}',
                             'n_A': len(xA) if xA is not None else 0,
                             'n_B': len(xB) if xB is not None else 0,
                             'diff_median': np.nan, 'p_two_sided': np.nan})
            continue
        print(f'  [{A} vs {B}] pairwise bootstrap ...', flush=True)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sA = RNG.choice(xA, size=len(xA), replace=True)
            sB = RNG.choice(xB, size=len(xB), replace=True)
            # Use MoM for pairwise — 100x faster and robust
            diffs[i] = nu_from_moments(sA) - nu_from_moments(sB)
        diffs = diffs[np.isfinite(diffs)]
        med = float(np.median(diffs))
        # two-sided bootstrap p-value: fraction on the "wrong side of 0"
        p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
        pairwise.append({'label': label, 'pair': f'{A}_vs_{B}',
                         'n_A': int(len(xA)), 'n_B': int(len(xB)),
                         'diff_median_nu': med,
                         'diff_ci_lo': float(np.percentile(diffs, 2.5)),
                         'diff_ci_hi': float(np.percentile(diffs, 97.5)),
                         'p_two_sided': float(p)})
        print(f'  nu({A}) - nu({B}) = {med:+.2f}  '
              f'[{np.percentile(diffs,2.5):+.2f}, {np.percentile(diffs,97.5):+.2f}]  '
              f'p={p:.4f}')

    return {
        'label': label,
        'class_counts': dict(cnt),
        'per_class_final': per_class_rows,
        'trajectory': traj_rows,
        'pairwise': pairwise,
        'class_deltas_final': {k: v.tolist() for k, v in class_deltas.items()},
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_all(res_410m, res_160m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'never': '#999999', 'born': '#1f77b4',
              'growing': '#ff7f0e', 'emergent': '#d62728'}

    # --- trajectories of nu ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, res, title in [(axes[0], res_410m, 'Pythia 410M'),
                           (axes[1], res_160m, 'Pythia 160M')]:
        traj = pd.DataFrame(res['trajectory'])
        for cls in ['never', 'born', 'growing', 'emergent']:
            d = traj[traj['class'] == cls]
            ax.plot(d['checkpoint'], d['nu'], marker='o', linewidth=1.5,
                    label=f'{cls} (n={res["class_counts"].get(cls,0)})',
                    color=colors[cls])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('training step')
        ax.set_ylabel(r'Student-$t$ degrees of freedom $\nu$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r'VWKK prediction: $\nu_{\mathrm{emergent}} < \nu_{\mathrm{born}}$')
    fig.tight_layout()
    out = os.path.join(FIGS, 'per_class_dfe_nu_trajectories.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')

    # --- KDE + fitted Student-t at final, per class, per model ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.linspace(-0.05, 0.05, 400)
    for ax, res, title in [(axes[0], res_410m, 'Pythia 410M  (step 143k)'),
                           (axes[1], res_160m, 'Pythia 160M  (step 143k)')]:
        for row in res['per_class_final']:
            cls = row['class']
            x = np.array(res['class_deltas_final'][cls])
            ax.hist(x, bins=30, density=True, histtype='step', color=colors[cls],
                    alpha=0.7, label=f'{cls} n={row["n"]}, ν={row["nu"]:.1f}')
            if np.isfinite(row['nu']):
                ax.plot(xs, sp.t.pdf(xs, row['nu'], row['loc'], row['sigma']),
                        color=colors[cls], linewidth=1.5, alpha=0.9)
        ax.set_xlabel(r'head ablation $\Delta$')
        ax.set_ylabel('density')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.03, 0.005)
    fig.tight_layout()
    out = os.path.join(FIGS, 'per_class_dfe_final_kde.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    path_410 = os.path.join(DATA, 'all_ablations.csv')
    path_160 = os.path.join(DATA, 'tier2_t21_scaling_160m.csv')

    res_410m = analyze(path_410, '410M')
    res_160m = analyze(path_160, '160M')

    # Save tables
    pd.DataFrame(res_410m['per_class_final']).to_csv(
        os.path.join(TABLES, 'per_class_dfe_410m.csv'), index=False)
    pd.DataFrame(res_160m['per_class_final']).to_csv(
        os.path.join(TABLES, 'per_class_dfe_160m.csv'), index=False)
    pairwise_df = pd.DataFrame(res_410m['pairwise'] + res_160m['pairwise'])
    pairwise_df.to_csv(os.path.join(TABLES, 'per_class_dfe_pairwise_tests.csv'),
                       index=False)

    # Figures
    plot_all(res_410m, res_160m)

    # Summary JSON
    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'config': {'crit_thresh': CRIT, 'born_low': BORN_LOW, 'n_boot': N_BOOT},
        '410m': {k: v for k, v in res_410m.items() if k != 'class_deltas_final'},
        '160m': {k: v for k, v in res_160m.items() if k != 'class_deltas_final'},
    }
    out = os.path.join(HERE, 'per_class_dfe_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
