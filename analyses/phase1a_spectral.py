"""Phase 1a: spectral head-level invariants on Pythia 410M step143000.

Primary test:   OV participation ratio vs |Delta| (Spearman rho, predicted < 0)
Secondary:      QK participation, OV entropy, QK entropy
Control:        random N(0,1) per head

Absorb-the-label regression:
    |Delta| ~ emergent_dummy + z(invariant)
Compare fractional reduction in emergent_dummy coefficient.

Runs on CPU. ~3-5 min total.

Outputs:
    tables/phase1a_spectral_invariants.csv
    figures/phase1a_scatter_primary.pdf
    figures/phase1a_scatter_all_invariants.pdf
    analyses/phase1a_summary.json
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
REVISION = 'step143000'
FIXED_HEADS = [0, 3, 6, 9, 12, 15]
N_LAYERS = 24
N_HEADS_PER_LAYER = 16
D_MODEL = 1024
D_HEAD = 64
CRIT = 5e-4
BORN_LOW = 1e-4
N_BOOT = 10_000
RNG_SEED = 20260424


# ---------------------------------------------------------------------------
# Spectral measurements per head
# ---------------------------------------------------------------------------

def participation_ratio(sigma):
    """PR = (sum sigma^2)^2 / sum sigma^4. Bounded [1, rank]."""
    s2 = sigma ** 2
    denom = np.sum(s2 ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(s2) ** 2 / denom)


def spectral_entropy(sigma):
    """H = -sum p_i log p_i with p_i = sigma_i^2 / sum sigma_j^2."""
    s2 = sigma ** 2
    total = s2.sum()
    if total <= 0:
        return np.nan
    p = s2 / total
    p = p[p > 1e-20]
    return float(-(p * np.log(p)).sum())


def head_spectral_invariants(model, layer_idx, head_idx):
    """Return dict of 4 spectral scalars for a single head."""
    block = model.gpt_neox.layers[layer_idx]
    qkv_weight = block.attention.query_key_value.weight.detach().cpu().numpy()
    out_weight = block.attention.dense.weight.detach().cpu().numpy()

    h = head_idx
    # Interleaved QKV layout in GPT-NeoX:
    # W[3*head_dim*i : 3*head_dim*(i+1)] is Q_i, K_i, V_i concatenated for head i.
    qkv_head_slice = qkv_weight[3 * D_HEAD * h : 3 * D_HEAD * (h + 1), :]
    W_Q = qkv_head_slice[0:D_HEAD, :]                      # (64, 1024)
    W_K = qkv_head_slice[D_HEAD:2 * D_HEAD, :]             # (64, 1024)
    W_V = qkv_head_slice[2 * D_HEAD:3 * D_HEAD, :]         # (64, 1024)
    W_O = out_weight[:, h * D_HEAD : (h + 1) * D_HEAD]     # (1024, 64)

    # OV circuit: (1024, 1024), rank <= 64
    M_OV = W_O @ W_V
    # QK circuit: (1024, 1024), rank <= 64
    M_QK = W_Q.T @ W_K

    sigma_OV = np.linalg.svd(M_OV, compute_uv=False)
    sigma_QK = np.linalg.svd(M_QK, compute_uv=False)

    return {
        'OV_PR': participation_ratio(sigma_OV),
        'QK_PR': participation_ratio(sigma_QK),
        'OV_entropy': spectral_entropy(sigma_OV),
        'QK_entropy': spectral_entropy(sigma_QK),
        'sigma_OV_max': float(sigma_OV[0]),
        'sigma_QK_max': float(sigma_QK[0]),
    }


# ---------------------------------------------------------------------------
# Outcome: per-head |Delta| + class assignment
# ---------------------------------------------------------------------------

def build_outcome_table():
    df = pd.read_csv(os.path.join(DATA, 'all_ablations.csv'))
    heads = df[df['perturbation_type'] == 'head'].copy()

    # Per-head |Delta| at step143000
    final_abs = (heads[heads['checkpoint'] == 143000]
                 .set_index(['layer_idx', 'head_idx'])['delta'].abs())

    # Class assignment using full trajectory
    pivot = heads.pivot_table(index=['layer_idx', 'head_idx'],
                              columns='checkpoint', values='delta').abs()
    first, last = pivot.columns.min(), pivot.columns.max()

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

    return final_abs, classes


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman_with_ci(x, y, n_boot=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho, p = sp.spearmanr(x, y)
    boots = np.empty(n_boot)
    n = len(x)
    idx = np.arange(n)
    for i in range(n_boot):
        ii = rng.choice(idx, size=n, replace=True)
        if len(np.unique(ii)) < 4:
            boots[i] = np.nan
            continue
        boots[i] = sp.spearmanr(x[ii], y[ii]).statistic
    boots = boots[np.isfinite(boots)]
    return {
        'rho': float(rho), 'p': float(p),
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
    }


def absorb_label_test(y, label_dummy, invariant_z):
    """Return dict of three OLS fits and the fractional reduction in label beta."""
    import statsmodels.api as sm
    X_a = sm.add_constant(label_dummy)
    m_a = sm.OLS(y, X_a).fit()
    beta_label_a = m_a.params[1]
    r2_a = m_a.rsquared

    X_b = sm.add_constant(invariant_z)
    m_b = sm.OLS(y, X_b).fit()
    beta_inv_b = m_b.params[1]
    r2_b = m_b.rsquared

    X_c = sm.add_constant(np.column_stack([label_dummy, invariant_z]))
    m_c = sm.OLS(y, X_c).fit()
    beta_label_c = m_c.params[1]
    beta_inv_c = m_c.params[2]
    p_label_c = m_c.pvalues[1]
    p_inv_c = m_c.pvalues[2]
    r2_c = m_c.rsquared

    frac_reduction = (beta_label_a - beta_label_c) / beta_label_a if beta_label_a != 0 else np.nan

    return {
        'label_only': {'beta_label': float(beta_label_a), 'r2': float(r2_a)},
        'invariant_only': {'beta_invariant': float(beta_inv_b), 'r2': float(r2_b)},
        'joint': {'beta_label': float(beta_label_c),
                  'p_label': float(p_label_c),
                  'beta_invariant': float(beta_inv_c),
                  'p_invariant': float(p_inv_c),
                  'r2': float(r2_c)},
        'fractional_label_reduction': float(frac_reduction),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_primary(df, out_path):
    colors = {'never': '#999999', 'born': '#1f77b4',
              'growing': '#ff7f0e', 'emergent': '#d62728'}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for cls in ['never', 'born', 'growing', 'emergent']:
        d = df[df['class'] == cls]
        ax.scatter(d['OV_PR'], d['abs_delta'], color=colors[cls],
                   label=f'{cls} (n={len(d)})', s=35, alpha=0.75, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('OV participation ratio (primary invariant)')
    ax.set_ylabel(r'$|\Delta_h|$ at step 143000')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    rho, p = sp.spearmanr(df['OV_PR'], df['abs_delta'])
    ax.set_title(rf'Primary: OV PR vs $|\Delta|$  —  Spearman $\rho$={rho:.3f}, p={p:.2e}')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all(df, out_path):
    panels = [('OV_PR', 'OV participation ratio (PRIMARY)'),
              ('QK_PR', 'QK participation ratio'),
              ('OV_entropy', 'OV spectral entropy'),
              ('QK_entropy', 'QK spectral entropy'),
              ('random_control', 'random N(0,1) CONTROL'),
              ('emergent_dummy', 'emergent/growing indicator (label baseline)')]
    colors = {'never': '#999999', 'born': '#1f77b4',
              'growing': '#ff7f0e', 'emergent': '#d62728'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        for cls in ['never', 'born', 'growing', 'emergent']:
            d = df[df['class'] == cls]
            ax.scatter(d[col], d['abs_delta'], color=colors[cls], s=22, alpha=0.7)
        rho, p = sp.spearmanr(df[col], df['abs_delta'])
        ax.set_xlabel(col)
        ax.set_ylabel(r'$|\Delta|$')
        ax.set_yscale('log')
        ax.set_title(f'{title}\nρ={rho:.3f}, p={p:.2e}', fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend([f'{c} (n={sum(df["class"]==c)})' for c in
                       ['never', 'born', 'growing', 'emergent']], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM
    print('Loading Pythia 410M step143000 on CPU (float32)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=REVISION, torch_dtype=torch.float32
    ).eval()
    print('  loaded.', flush=True)

    rng = np.random.default_rng(RNG_SEED)

    print('Building outcome table (|Delta|, class)...', flush=True)
    abs_delta, classes = build_outcome_table()
    print(f'  {len(abs_delta)} head rows,  class counts: '
          + str({c: sum(1 for v in classes.values() if v == c)
                 for c in ['never', 'born', 'emergent', 'growing']}), flush=True)

    rows = []
    print('Computing spectral invariants per head...', flush=True)
    for L in range(N_LAYERS):
        for H in FIXED_HEADS:
            inv = head_spectral_invariants(model, L, H)
            cls = classes.get((L, H), 'unknown')
            ad = float(abs_delta.get((L, H), np.nan))
            rows.append({
                'layer_idx': L, 'head_idx': H, 'class': cls, 'abs_delta': ad,
                'random_control': float(rng.standard_normal()),
                **inv,
            })
        print(f'  layer {L} done', flush=True)

    df = pd.DataFrame(rows)
    df['emergent_dummy'] = df['class'].isin(['emergent', 'growing']).astype(int)
    out_csv = os.path.join(TABLES, 'phase1a_spectral_invariants.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}', flush=True)

    # Spearman + CIs
    results = {}
    invariants = ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy',
                  'random_control', 'emergent_dummy']
    print('\n=== UNIVARIATE SPEARMAN vs |Delta| ===')
    print(f'{"invariant":<22} {"rho":>7} {"95% CI":>24} {"p":>10}')
    print('-' * 70)
    for inv in invariants:
        x = df[inv].values
        y = df['abs_delta'].values
        rng_local = np.random.default_rng(RNG_SEED + hash(inv) % 1_000_000)
        r = spearman_with_ci(x, y, rng=rng_local)
        results[inv] = {'spearman': r}
        print(f'{inv:<22} {r["rho"]:>+7.3f} '
              f'[{r["ci_lo"]:+6.3f}, {r["ci_hi"]:+6.3f}] {r["p"]:>10.2e}')

    # Absorb-the-label tests (skip emergent_dummy itself)
    print('\n=== ABSORB-THE-LABEL REGRESSIONS (|Delta| ~ label + z(invariant)) ===')
    y_arr = df['abs_delta'].values
    lbl = df['emergent_dummy'].values.astype(float)
    for inv in ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy', 'random_control']:
        x = df[inv].values
        xz = (x - np.nanmean(x)) / np.nanstd(x)
        r = absorb_label_test(y_arr, lbl, xz)
        results[inv]['absorb_test'] = r
        a = r['label_only']; c = r['joint']
        print(f'{inv:<18} | R² label-only={a["r2"]:.3f}  joint={c["r2"]:.3f}  '
              f'β_label: {a["beta_label"]:+.4f} -> {c["beta_label"]:+.4f}  '
              f'(reduced {r["fractional_label_reduction"]*100:.0f}%)  '
              f'p_inv={c["p_invariant"]:.3e}')

    # Figures
    plot_primary(df, os.path.join(FIGS, 'phase1a_scatter_primary.pdf'))
    plot_all(df, os.path.join(FIGS, 'phase1a_scatter_all_invariants.pdf'))

    # Summary JSON
    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'model': MODEL_NAME, 'revision': REVISION,
        'n_heads': int(len(df)),
        'class_counts': {c: int(sum(df['class'] == c))
                         for c in ['never', 'born', 'emergent', 'growing']},
        'preregistered_primary': 'OV_PR',
        'preregistered_direction': 'negative',
        'results_per_invariant': results,
    }
    out_json = os.path.join(HERE, 'phase1a_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out_json}')

    # Verdict against pre-registration
    print('\n=== PRE-REGISTERED VERDICT ===')
    r = results['OV_PR']['spearman']
    direction_ok = r['rho'] < 0
    magnitude = abs(r['rho'])
    if direction_ok and magnitude > 0.6:
        verdict = 'STRONG (|rho|>0.6, correct direction)'
    elif direction_ok and magnitude > 0.4:
        verdict = 'MEDIUM (|rho|>0.4, correct direction)'
    elif direction_ok:
        verdict = 'WEAK (direction correct but |rho|<0.4)'
    elif magnitude > 0.4:
        verdict = f'OPPOSITE DIRECTION (|rho|={magnitude:.2f}, predicted negative, got positive)'
    else:
        verdict = 'NULL (|rho|<0.4 both directions)'
    print(f'Primary OV_PR: rho={r["rho"]:+.3f}, |rho|={magnitude:.3f}')
    print(f'Random control: rho={results["random_control"]["spearman"]["rho"]:+.3f}')
    print(f'>>> {verdict} <<<')


if __name__ == '__main__':
    main()
