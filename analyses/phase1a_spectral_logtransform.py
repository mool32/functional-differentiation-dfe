"""Phase 1a follow-up: repeat regressions with log-transformed outcome.

Heavy-tailed |Delta| broke raw-OLS absorb-label test (R² = 0.016 despite
Spearman 0.70). Log-transform compresses the tail so OLS captures the
rank-structure correctly. Same pre-registered decision rules apply.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')

df = pd.read_csv(os.path.join(TABLES, 'phase1a_spectral_invariants.csv'))

# Log-transformed outcome: handles heavy tail + zero-ish cases
eps = 1e-6
df['log_abs_delta'] = np.log(np.maximum(df['abs_delta'].values, eps))

def absorb_label_test(y, label_dummy, invariant_z, name):
    X_a = sm.add_constant(label_dummy)
    m_a = sm.OLS(y, X_a).fit()
    X_b = sm.add_constant(invariant_z)
    m_b = sm.OLS(y, X_b).fit()
    X_c = sm.add_constant(np.column_stack([label_dummy, invariant_z]))
    m_c = sm.OLS(y, X_c).fit()
    red = (m_a.params[1] - m_c.params[1]) / m_a.params[1] if m_a.params[1] else np.nan
    return {
        'invariant': name,
        'r2_label_only': float(m_a.rsquared),
        'r2_invariant_only': float(m_b.rsquared),
        'r2_joint': float(m_c.rsquared),
        'beta_label_alone': float(m_a.params[1]),
        'beta_label_joint': float(m_c.params[1]),
        'p_label_joint': float(m_c.pvalues[1]),
        'beta_invariant_alone': float(m_b.params[1]),
        'beta_invariant_joint': float(m_c.params[2]),
        'p_invariant_joint': float(m_c.pvalues[2]),
        'fractional_label_reduction': float(red),
    }


def zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


y = df['log_abs_delta'].values
lbl = df['emergent_dummy'].values.astype(float)

results = []
print('\n=== ABSORB-THE-LABEL ON LOG(|Delta|) ===')
print(f'{"invariant":<18} {"R2(L)":>7} {"R2(I)":>7} {"R2(L+I)":>8} '
      f'{"β_L alone":>10} {"β_L joint":>10} {"red %":>6} '
      f'{"β_I joint":>10} {"p_I joint":>10}')
print('-' * 110)
for inv in ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy', 'random_control']:
    r = absorb_label_test(y, lbl, zscore(df[inv].values), inv)
    results.append(r)
    print(f'{inv:<18} '
          f'{r["r2_label_only"]:>7.3f} {r["r2_invariant_only"]:>7.3f} {r["r2_joint"]:>8.3f} '
          f'{r["beta_label_alone"]:>+10.4f} {r["beta_label_joint"]:>+10.4f} '
          f'{r["fractional_label_reduction"]*100:>+6.1f} '
          f'{r["beta_invariant_joint"]:>+10.4f} {r["p_invariant_joint"]:>10.2e}')

# Cross-test: per-class Spearman — is QK signal inside classes or between?
from scipy import stats as sp
print('\n=== QK_PR ρ(invariant, log|Δ|) WITHIN CLASSES ===')
print('(checks whether QK signal is cross-class confound or real within-class)')
for cls in ['never', 'born', 'emergent', 'growing']:
    d = df[df['class'] == cls]
    if len(d) < 5:
        print(f'  {cls:<10} n={len(d)} too few')
        continue
    rho_qk, p_qk = sp.spearmanr(d['QK_PR'], d['log_abs_delta'])
    rho_ov, p_ov = sp.spearmanr(d['OV_PR'], d['log_abs_delta'])
    print(f'  {cls:<10} n={len(d):>3}  QK_PR ρ={rho_qk:+.3f} p={p_qk:.3f}  |  '
          f'OV_PR ρ={rho_ov:+.3f} p={p_ov:.3f}')

# Class means of each invariant
print('\n=== INVARIANT MEANS BY CLASS ===')
print(f'{"class":<10} {"n":>4} {"OV_PR":>8} {"QK_PR":>8} {"OV_ent":>8} {"QK_ent":>8}')
for cls in ['never', 'born', 'emergent', 'growing']:
    d = df[df['class'] == cls]
    print(f'{cls:<10} {len(d):>4} '
          f'{d["OV_PR"].mean():>8.2f} {d["QK_PR"].mean():>8.2f} '
          f'{d["OV_entropy"].mean():>8.3f} {d["QK_entropy"].mean():>8.3f}')

# Save
out = os.path.join(HERE, 'phase1a_logtransform_summary.json')
with open(out, 'w') as f:
    json.dump({
        'generated': pd.Timestamp.utcnow().isoformat(),
        'outcome_transform': 'log(max(|Delta|, 1e-6))',
        'absorb_tests': results,
    }, f, indent=2, default=str)
print(f'\nwrote {out}')
