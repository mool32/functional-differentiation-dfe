"""Block 1.2 robustness pipeline.

Runs in order:
  (A) threshold sensitivity: crit in {2e-4, 3e-4, 5e-4, 7e-4, 1e-3}
  (B) class-count matching (cross-check for classification-artefact story)
  (C) random-shuffle null test
  (D) cross-model AIC-inversion test (bootstrap)
  (E) pooled-checkpoints + time-resolved nu(t) per class

Emphasis: AIC structure, not pairwise nu. n=6 classes flagged as unreliable.

Inputs:  ../data/all_ablations.csv, ../data/tier2_t21_scaling_160m.csv
Outputs: ../tables/robust_*.csv, ../figures/robust_*.pdf, summary json
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

BORN_LOW_FRAC = 1 / 5  # init threshold for emergent is born_low = crit * BORN_LOW_FRAC
N_BOOT = 2_000
N_SHUFFLES = 1_000
RNG = np.random.default_rng(20260421)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def classify_heads(df_heads, crit, born_low=None):
    if born_low is None:
        born_low = crit * BORN_LOW_FRAC
    ckpts = sorted(df_heads['checkpoint'].unique())
    first, last = ckpts[0], ckpts[-1]
    pivot = df_heads.pivot_table(index=['layer_idx', 'head_idx'],
                                 columns='checkpoint', values='delta').abs()
    classes = {}
    for idx, row in pivot.iterrows():
        init, fin = row[first], row[last]
        if fin < crit:
            c = 'never'
        elif init > crit and fin > crit:
            c = 'born'
        elif init < born_low and fin > crit:
            c = 'emergent'
        else:
            c = 'growing'
        classes[idx] = c
    return classes


def fit_student_t(x):
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    try:
        nu, loc, sc = sp.t.fit(x)
        return nu, loc, sc
    except Exception:
        return np.nan, np.nan, np.nan


def nu_from_moments(x):
    if len(x) < 4:
        return np.nan
    k = sp.kurtosis(x)
    if k <= 0:
        return np.inf
    return 4 + 6 / k


def student_t_aic(x, nu, loc, sc):
    if not np.isfinite(nu) or nu <= 0:
        return np.nan
    return 2 * 3 - 2 * np.sum(sp.t.logpdf(x, nu, loc, sc))


def normal_aic(x):
    mu, si = sp.norm.fit(x)
    return 2 * 2 - 2 * np.sum(sp.norm.logpdf(x, mu, si))


def aic_diff_t_minus_normal(x):
    """Positive => Student-t is preferred."""
    if len(x) < 4:
        return np.nan
    nu, loc, sc = fit_student_t(x)
    return normal_aic(x) - student_t_aic(x, nu, loc, sc)


def per_class_summary(df_heads, classes, checkpoint=None):
    """At a given checkpoint (default last), fit per class."""
    if checkpoint is None:
        checkpoint = df_heads['checkpoint'].max()
    d = df_heads[df_heads['checkpoint'] == checkpoint].copy()
    d['cls'] = d.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
    out = {}
    for cls in ['never', 'born', 'emergent', 'growing']:
        x = d.loc[d['cls'] == cls, 'delta'].values
        nu, loc, sc = fit_student_t(x) if len(x) >= 4 else (np.nan, np.nan, np.nan)
        out[cls] = {
            'n': int(len(x)),
            'nu_MLE': float(nu), 'nu_MoM': float(nu_from_moments(x)),
            'sigma': float(sc),
            'delta_aic_t_minus_n': float(aic_diff_t_minus_normal(x)) if len(x) >= 4 else np.nan,
            'std': float(np.std(x)) if len(x) else np.nan,
        }
    return out


# ---------------------------------------------------------------------------
# (A) THRESHOLD SENSITIVITY
# ---------------------------------------------------------------------------

def threshold_sweep(path, label, thresholds):
    df = pd.read_csv(path)
    heads = df[df['perturbation_type'] == 'head'].copy()
    rows = []
    for crit in thresholds:
        classes = classify_heads(heads, crit)
        cnt = Counter(classes.values())
        summ = per_class_summary(heads, classes)
        for cls, s in summ.items():
            rows.append({'model': label, 'crit': crit, 'class': cls,
                         'n': s['n'], 'nu_MoM': s['nu_MoM'],
                         'delta_aic_t_minus_n': s['delta_aic_t_minus_n']})
    return pd.DataFrame(rows)


def plot_threshold_sweep(df_thr, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {'never': '#999999', 'born': '#1f77b4',
              'growing': '#ff7f0e', 'emergent': '#d62728'}
    for ax, (model, metric, title) in zip(
            axes.flat,
            [('410M', 'delta_aic_t_minus_n', '410M — Δ_AIC(t − n) per class'),
             ('160M', 'delta_aic_t_minus_n', '160M — Δ_AIC(t − n) per class'),
             ('410M', 'n', '410M — class counts'),
             ('160M', 'n', '160M — class counts')]):
        d = df_thr[df_thr['model'] == model]
        for cls in ['never', 'born', 'growing', 'emergent']:
            dd = d[d['class'] == cls]
            ax.plot(dd['crit'], dd[metric], 'o-', color=colors[cls],
                    label=cls, linewidth=1.6)
        if metric == 'delta_aic_t_minus_n':
            ax.axhline(0, color='k', linestyle='--', alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel(r'classification threshold $\theta_{\rm crit}$')
        ax.set_ylabel(metric)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (B) CLASS-COUNT MATCHING
# ---------------------------------------------------------------------------

def find_threshold_for_n_born(path, n_target, grid=None):
    df = pd.read_csv(path)
    heads = df[df['perturbation_type'] == 'head'].copy()
    if grid is None:
        grid = np.logspace(-5, -2.5, 60)
    best = None
    for crit in grid:
        classes = classify_heads(heads, crit)
        n_born = sum(1 for v in classes.values() if v == 'born')
        if best is None or abs(n_born - n_target) < abs(best[1] - n_target):
            best = (crit, n_born, classes)
    return best  # (crit, n_born, classes_dict)


def class_count_match(path_410, path_160):
    df410 = pd.read_csv(path_410)
    h410 = df410[df410['perturbation_type'] == 'head'].copy()
    df160 = pd.read_csv(path_160)
    h160 = df160[df160['perturbation_type'] == 'head'].copy()

    rows = []

    # (B1) 410M with a permissive threshold that yields ~40 born (matching 160M)
    crit, n, classes = find_threshold_for_n_born(path_410, n_target=40)
    summ = per_class_summary(h410, classes)
    print(f'\n(B1) 410M @ permissive crit={crit:.2e}  -> n_born={n}')
    for cls, s in summ.items():
        print(f'    {cls:<10} n={s["n"]:>3}  nu_MoM={s["nu_MoM"]:>7.2f}  Δ_AIC(t-n)={s["delta_aic_t_minus_n"]:+7.1f}')
        rows.append({'test': 'B1_410M_permissive', 'crit': crit, 'class': cls, **s})

    # (B2) 160M with a strict threshold that yields ~6 born (matching 410M)
    crit2, n2, classes2 = find_threshold_for_n_born(path_160, n_target=6)
    summ2 = per_class_summary(h160, classes2)
    print(f'\n(B2) 160M @ strict crit={crit2:.2e}  -> n_born={n2}')
    for cls, s in summ2.items():
        print(f'    {cls:<10} n={s["n"]:>3}  nu_MoM={s["nu_MoM"]:>7.2f}  Δ_AIC(t-n)={s["delta_aic_t_minus_n"]:+7.1f}')
        rows.append({'test': 'B2_160M_strict', 'crit': crit2, 'class': cls, **s})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (C) RANDOM-SHUFFLE NULL
# ---------------------------------------------------------------------------

def shuffle_null(path, label, n_shuffles=N_SHUFFLES, crit=5e-4):
    """Shuffle class labels within model; compute Δ_AIC(emergent) - Δ_AIC(born) per shuffle."""
    df = pd.read_csv(path)
    heads = df[df['perturbation_type'] == 'head'].copy()
    classes = classify_heads(heads, crit)
    last = heads['checkpoint'].max()
    d = heads[heads['checkpoint'] == last].copy()
    d['cls'] = d.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)

    # Real statistic
    real_emer_aic = aic_diff_t_minus_normal(d.loc[d['cls'] == 'emergent', 'delta'].values)
    real_born_aic = aic_diff_t_minus_normal(d.loc[d['cls'] == 'born', 'delta'].values)
    real_S = real_emer_aic - real_born_aic

    # Class sizes
    counts = Counter(d['cls'])
    labels = list(d['cls'].values)
    deltas = d['delta'].values

    null_S = np.empty(n_shuffles)
    rng = np.random.default_rng(RNG.integers(1e9))
    for i in range(n_shuffles):
        shuffled = np.array(labels); rng.shuffle(shuffled)
        emer_x = deltas[shuffled == 'emergent']
        born_x = deltas[shuffled == 'born']
        a_e = aic_diff_t_minus_normal(emer_x) if len(emer_x) >= 4 else np.nan
        a_b = aic_diff_t_minus_normal(born_x) if len(born_x) >= 4 else np.nan
        null_S[i] = a_e - a_b
    null_S = null_S[np.isfinite(null_S)]
    p_two = 2 * min((null_S >= real_S).mean(), (null_S <= real_S).mean())
    p_two = max(p_two, 1 / (len(null_S) + 1))
    return {
        'model': label, 'n_shuffles': int(len(null_S)),
        'real_S_aic_emer_minus_born': float(real_S),
        'null_mean': float(np.mean(null_S)), 'null_std': float(np.std(null_S)),
        'null_ci_95': (float(np.percentile(null_S, 2.5)), float(np.percentile(null_S, 97.5))),
        'p_two_sided': float(p_two),
    }


# ---------------------------------------------------------------------------
# (D) CROSS-MODEL AIC-INVERSION
# ---------------------------------------------------------------------------

def cross_model_inversion(path_410, path_160, crit=5e-4, n_boot=N_BOOT):
    """Bootstrap statistic  S = [ΔAIC(emer) - ΔAIC(born)]_410M  -  [same]_160M.
    If phase transition is real, S should be strongly positive.
    """
    def draw_stat(heads, classes, rng):
        last = heads['checkpoint'].max()
        d = heads[heads['checkpoint'] == last].copy()
        d['cls'] = d.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
        xe = d.loc[d['cls'] == 'emergent', 'delta'].values
        xb = d.loc[d['cls'] == 'born', 'delta'].values
        if len(xe) < 4 or len(xb) < 4:
            return np.nan
        se = rng.choice(xe, size=len(xe), replace=True)
        sb = rng.choice(xb, size=len(xb), replace=True)
        return aic_diff_t_minus_normal(se) - aic_diff_t_minus_normal(sb)

    df410 = pd.read_csv(path_410); h410 = df410[df410['perturbation_type'] == 'head']
    df160 = pd.read_csv(path_160); h160 = df160[df160['perturbation_type'] == 'head']
    c410 = classify_heads(h410, crit)
    c160 = classify_heads(h160, crit)

    rng = np.random.default_rng(RNG.integers(1e9))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s410 = draw_stat(h410, c410, rng)
        s160 = draw_stat(h160, c160, rng)
        diffs[i] = s410 - s160
    diffs = diffs[np.isfinite(diffs)]

    # Real (no resample) values
    def real_stat(heads, classes):
        last = heads['checkpoint'].max()
        d = heads[heads['checkpoint'] == last].copy()
        d['cls'] = d.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
        xe = d.loc[d['cls'] == 'emergent', 'delta'].values
        xb = d.loc[d['cls'] == 'born', 'delta'].values
        return aic_diff_t_minus_normal(xe) - aic_diff_t_minus_normal(xb)
    S410 = real_stat(h410, c410)
    S160 = real_stat(h160, c160)

    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_two = max(p_two, 1 / (len(diffs) + 1))
    return {
        'S_410M_real': float(S410), 'S_160M_real': float(S160),
        'S_410M_minus_S_160M_real': float(S410 - S160),
        'boot_mean': float(np.mean(diffs)),
        'boot_ci_95': (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        'p_two_sided': float(p_two),
    }


# ---------------------------------------------------------------------------
# (E) POOLED + TIME-RESOLVED
# ---------------------------------------------------------------------------

def time_resolved_and_pooled(path, label, crit=5e-4):
    df = pd.read_csv(path)
    heads = df[df['perturbation_type'] == 'head'].copy()
    classes = classify_heads(heads, crit)
    ckpts = sorted(heads['checkpoint'].unique())

    traj_rows = []
    for ck in ckpts:
        d = heads[heads['checkpoint'] == ck].copy()
        d['cls'] = d.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
        for cls in ['never', 'born', 'emergent', 'growing']:
            x = d.loc[d['cls'] == cls, 'delta'].values
            row = {'model': label, 'checkpoint': ck, 'class': cls, 'n': int(len(x))}
            if len(x) >= 4:
                nu, loc, sc = fit_student_t(x)
                row['nu_MoM'] = float(nu_from_moments(x))
                row['nu_MLE'] = float(nu)
                row['sigma'] = float(sc)
                row['delta_aic_t_minus_n'] = float(aic_diff_t_minus_normal(x))
            else:
                row.update(nu_MoM=np.nan, nu_MLE=np.nan, sigma=np.nan, delta_aic_t_minus_n=np.nan)
            traj_rows.append(row)

    # Pool over late checkpoints (last 4)
    late = ckpts[-4:]
    pooled = {}
    for cls in ['never', 'born', 'emergent', 'growing']:
        x = heads[(heads['checkpoint'].isin(late))].copy()
        x['cls'] = x.apply(lambda r: classes[(int(r['layer_idx']), int(r['head_idx']))], axis=1)
        vec = x.loc[x['cls'] == cls, 'delta'].values
        if len(vec) < 4:
            pooled[cls] = {'n': int(len(vec)), 'nu_MoM': np.nan, 'delta_aic_t_minus_n': np.nan}
        else:
            pooled[cls] = {
                'n': int(len(vec)),
                'nu_MoM': float(nu_from_moments(vec)),
                'nu_MLE': float(fit_student_t(vec)[0]),
                'delta_aic_t_minus_n': float(aic_diff_t_minus_normal(vec)),
            }
    return pd.DataFrame(traj_rows), pooled


def plot_time_resolved(df_traj_410, df_traj_160, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'never': '#999999', 'born': '#1f77b4',
              'growing': '#ff7f0e', 'emergent': '#d62728'}
    for ax, df, title in [(axes[0], df_traj_410, '410M — Δ_AIC(t − n)(t)'),
                          (axes[1], df_traj_160, '160M — Δ_AIC(t − n)(t)')]:
        for cls in ['never', 'born', 'growing', 'emergent']:
            d = df[df['class'] == cls]
            ax.plot(d['checkpoint'], d['delta_aic_t_minus_n'], 'o-',
                    color=colors[cls], label=cls, linewidth=1.6)
        ax.axhline(0, color='k', linestyle='--', alpha=0.4)
        ax.set_xscale('log')
        ax.set_xlabel('training step')
        ax.set_ylabel('Δ_AIC(t − n)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    path_410 = os.path.join(DATA, 'all_ablations.csv')
    path_160 = os.path.join(DATA, 'tier2_t21_scaling_160m.csv')

    print('\n# (A) THRESHOLD SENSITIVITY\n')
    thresholds = [2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    df_thr = pd.concat([
        threshold_sweep(path_410, '410M', thresholds),
        threshold_sweep(path_160, '160M', thresholds),
    ], ignore_index=True)
    df_thr.to_csv(os.path.join(TABLES, 'robust_threshold_sweep.csv'), index=False)
    plot_threshold_sweep(df_thr, os.path.join(FIGS, 'robust_threshold_sweep.pdf'))
    print(df_thr.to_string(index=False))

    print('\n# (B) CLASS-COUNT MATCHING\n')
    df_match = class_count_match(path_410, path_160)
    df_match.to_csv(os.path.join(TABLES, 'robust_class_count_matching.csv'), index=False)

    print('\n# (C) RANDOM-SHUFFLE NULL\n')
    null_410 = shuffle_null(path_410, '410M')
    null_160 = shuffle_null(path_160, '160M')
    print('  410M:', null_410)
    print('  160M:', null_160)

    print('\n# (D) CROSS-MODEL AIC-INVERSION\n')
    inv = cross_model_inversion(path_410, path_160)
    for k, v in inv.items():
        print(f'  {k}: {v}')

    print('\n# (E) POOLED + TIME-RESOLVED\n')
    traj_410, pooled_410 = time_resolved_and_pooled(path_410, '410M')
    traj_160, pooled_160 = time_resolved_and_pooled(path_160, '160M')
    pd.concat([traj_410, traj_160]).to_csv(
        os.path.join(TABLES, 'robust_time_resolved.csv'), index=False)
    plot_time_resolved(traj_410, traj_160, os.path.join(FIGS, 'robust_time_resolved.pdf'))
    print('Pooled (last 4 ckpts) 410M:')
    for k, v in pooled_410.items(): print(f'    {k:<10} {v}')
    print('Pooled (last 4 ckpts) 160M:')
    for k, v in pooled_160.items(): print(f'    {k:<10} {v}')

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'A_threshold_sweep_pivot_410M': df_thr[df_thr['model']=='410M'].pivot(
            index='crit', columns='class', values='delta_aic_t_minus_n').to_dict(),
        'A_threshold_sweep_pivot_160M': df_thr[df_thr['model']=='160M'].pivot(
            index='crit', columns='class', values='delta_aic_t_minus_n').to_dict(),
        'B_class_count_matching': df_match.to_dict(orient='records'),
        'C_null_shuffle_410M': null_410,
        'C_null_shuffle_160M': null_160,
        'D_cross_model_inversion': inv,
        'E_pooled_410M': pooled_410,
        'E_pooled_160M': pooled_160,
    }
    out = os.path.join(HERE, 'per_class_dfe_robustness_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
