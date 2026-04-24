"""Phase 1b-C: same spectral-invariants pipeline applied to Pythia 160M step143000.

Tests whether the Phase 1a active-vs-dead discrimination pattern (QK_PR ~20
for never vs ~40 for active) replicates at 160M. Three possible outcomes:

    (1) QK_PR discriminates active/dead equivalently   -> universal architectural
    (2) QK_PR does not discriminate at 160M            -> scale-dependent emergence
    (3) QK_PR discriminates with inverted direction    -> parallels DFE inversion

The test is written against the Phase 1a locked framing — we report active-vs-dead
discrimination separately from heavy-tail prediction.
"""
import json, os
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(ROOT, 'tables')

MODEL_NAME = 'EleutherAI/pythia-160m-deduped'
REVISION = 'step143000'
CRIT = 5e-4
BORN_LOW = 1e-4
RNG_SEED = 20260424


def participation_ratio(sigma):
    s2 = sigma ** 2
    d = np.sum(s2 ** 2)
    return float(np.sum(s2) ** 2 / d) if d > 0 else np.nan


def spectral_entropy(sigma):
    s2 = sigma ** 2
    total = s2.sum()
    if total <= 0: return np.nan
    p = s2 / total
    p = p[p > 1e-20]
    return float(-(p * np.log(p)).sum())


def head_invariants(model, layer_idx, head_idx, d_head, d_model):
    block = model.gpt_neox.layers[layer_idx]
    qkv = block.attention.query_key_value.weight.detach().cpu().numpy()
    out = block.attention.dense.weight.detach().cpu().numpy()
    h = head_idx
    sl = qkv[3 * d_head * h : 3 * d_head * (h + 1), :]
    W_Q = sl[0:d_head]
    W_K = sl[d_head:2*d_head]
    W_V = sl[2*d_head:3*d_head]
    W_O = out[:, h*d_head:(h+1)*d_head]
    M_OV = W_O @ W_V
    M_QK = W_Q.T @ W_K
    sv_OV = np.linalg.svd(M_OV, compute_uv=False)
    sv_QK = np.linalg.svd(M_QK, compute_uv=False)
    return {
        'OV_PR': participation_ratio(sv_OV),
        'QK_PR': participation_ratio(sv_QK),
        'OV_entropy': spectral_entropy(sv_OV),
        'QK_entropy': spectral_entropy(sv_QK),
    }


def build_outcome_160m():
    df = pd.read_csv(os.path.join(DATA, 'tier2_t21_scaling_160m.csv'))
    heads = df[df['perturbation_type'] == 'head'].copy()
    pivot = heads.pivot_table(index=['layer_idx','head_idx'],
                              columns='checkpoint', values='delta').abs()
    first, last = pivot.columns.min(), pivot.columns.max()
    final_abs = pivot[last]
    classes = {}
    for idx, row in pivot.iterrows():
        init, fin = row[first], row[last]
        if fin < CRIT: c = 'never'
        elif init > CRIT and fin > CRIT: c = 'born'
        elif init < BORN_LOW and fin > CRIT: c = 'emergent'
        else: c = 'growing'
        classes[idx] = c
    return final_abs, classes


def main():
    print(f'Loading {MODEL_NAME} {REVISION} (CPU, float32)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=REVISION,
                                                 torch_dtype=torch.float32).eval()
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    d_head = d_model // n_heads
    n_layers = len(model.gpt_neox.layers)
    print(f'  arch: {n_layers}L x {n_heads}H, d_head={d_head}, d_model={d_model}', flush=True)

    abs_delta, classes = build_outcome_160m()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for L in range(n_layers):
        for H in range(n_heads):
            inv = head_invariants(model, L, H, d_head, d_model)
            rows.append({
                'layer_idx': L, 'head_idx': H,
                'class': classes.get((L, H), 'unknown'),
                'abs_delta': float(abs_delta.get((L, H), np.nan)),
                'random_control': float(rng.standard_normal()),
                **inv,
            })
        print(f'  layer {L} done', flush=True)

    df = pd.DataFrame(rows)
    df['emergent_dummy'] = df['class'].isin(['emergent', 'growing']).astype(int)
    df['log_abs_delta'] = np.log(np.maximum(df['abs_delta'].values, 1e-6))
    out_csv = os.path.join(TABLES, 'phase1b_C_160m_invariants.csv')
    df.to_csv(out_csv, index=False)

    print('\n=== 160M: UNIVARIATE SPEARMAN vs |Delta| ===')
    for inv in ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy', 'random_control', 'emergent_dummy']:
        r, p = sp.spearmanr(df[inv], df['abs_delta'])
        print(f'  {inv:<16} rho={r:+.3f}  p={p:.2e}')

    print('\n=== 160M: INVARIANT MEANS BY CLASS ===')
    print(f'{"class":<10} {"n":>4} {"OV_PR":>8} {"QK_PR":>8} {"OV_ent":>8} {"QK_ent":>8}')
    for cls in ['never', 'born', 'emergent', 'growing']:
        d = df[df['class'] == cls]
        if len(d) == 0: continue
        print(f'{cls:<10} {len(d):>4} {d["OV_PR"].mean():>8.2f} {d["QK_PR"].mean():>8.2f} '
              f'{d["OV_entropy"].mean():>8.3f} {d["QK_entropy"].mean():>8.3f}')

    # Active vs dead specifically for QK_PR
    active = df[df['class'] != 'never']
    dead = df[df['class'] == 'never']
    if len(dead) >= 3 and len(active) >= 10:
        t_qk_pr = sp.mannwhitneyu(active['QK_PR'], dead['QK_PR'], alternative='two-sided')
        t_qk_en = sp.mannwhitneyu(active['QK_entropy'], dead['QK_entropy'], alternative='two-sided')
        t_ov_pr = sp.mannwhitneyu(active['OV_PR'], dead['OV_PR'], alternative='two-sided')
        delta_mean_qk_pr = active['QK_PR'].mean() - dead['QK_PR'].mean()
        print(f'\n=== 160M: ACTIVE vs DEAD (Mann-Whitney U) ===')
        print(f'  QK_PR      active mean {active["QK_PR"].mean():.2f}  dead mean {dead["QK_PR"].mean():.2f}  '
              f'delta={delta_mean_qk_pr:+.2f}  p={t_qk_pr.pvalue:.2e}')
        print(f'  QK_entropy active mean {active["QK_entropy"].mean():.3f}  dead mean {dead["QK_entropy"].mean():.3f}  '
              f'p={t_qk_en.pvalue:.2e}')
        print(f'  OV_PR      active mean {active["OV_PR"].mean():.2f}  dead mean {dead["OV_PR"].mean():.2f}  '
              f'p={t_ov_pr.pvalue:.2e}')

        # Compare to 410M: was delta_mean_QK_PR ~+20 at 410M (active ~40, dead ~20)?
        print(f'\n410M baseline (from Phase 1a): QK_PR dead=20, active=38-46, delta ~+20')
        print(f'160M result:                  QK_PR dead={dead["QK_PR"].mean():.1f}, active={active["QK_PR"].mean():.1f}, '
              f'delta={delta_mean_qk_pr:+.1f}')

        if delta_mean_qk_pr > 10:
            outcome = '(1) UNIVERSAL: QK_PR discriminates active/dead at both scales'
        elif abs(delta_mean_qk_pr) < 5:
            outcome = '(2) SCALE-DEPENDENT: QK_PR discrimination absent at 160M'
        elif delta_mean_qk_pr < -10:
            outcome = '(3) INVERTED: QK_PR discrimination flips sign at 160M'
        else:
            outcome = 'INTERMEDIATE'
        print(f'\n>>> OUTCOME: {outcome} <<<')

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'model': MODEL_NAME, 'revision': REVISION,
        'class_counts': {c: int(sum(df['class'] == c))
                         for c in ['never', 'born', 'emergent', 'growing']},
        'per_invariant_spearman': {
            inv: {'rho': float(sp.spearmanr(df[inv], df['abs_delta']).statistic),
                  'p': float(sp.spearmanr(df[inv], df['abs_delta']).pvalue)}
            for inv in ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy',
                        'random_control', 'emergent_dummy']},
        'class_means': {c: {k: float(df[df['class']==c][k].mean())
                            for k in ['OV_PR','QK_PR','OV_entropy','QK_entropy']}
                        for c in ['never','born','emergent','growing']
                        if sum(df['class']==c) > 0},
    }
    out_json = os.path.join(HERE, 'phase1b_C_160m_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out_csv}\nwrote {out_json}')


if __name__ == '__main__':
    main()
