"""Phase 1b-B: QK_PR stability across checkpoints on Pythia 410M.

Loads step 16000 and step 64000 weights, computes QK_PR per head, then tests:
  (1) Spearman rho(QK_PR at checkpoint X, QK_PR at step 143000) across 144 heads
  (2) Mean QK_PR per class at checkpoint X (classes determined by full trajectory)
  (3) Spearman rho(QK_PR at checkpoint X, |Delta| at checkpoint X)

Causal-ordering question: does QK discrimination develop before or after
|Delta| discrimination emerges?
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

MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
FIXED_HEADS = [0, 3, 6, 9, 12, 15]
N_LAYERS = 24
D_HEAD = 64
CRIT = 5e-4
BORN_LOW = 1e-4


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


def head_invariants(model, L, H):
    block = model.gpt_neox.layers[L]
    qkv = block.attention.query_key_value.weight.detach().cpu().numpy()
    out = block.attention.dense.weight.detach().cpu().numpy()
    sl = qkv[3*D_HEAD*H : 3*D_HEAD*(H+1), :]
    W_Q = sl[0:D_HEAD]; W_K = sl[D_HEAD:2*D_HEAD]; W_V = sl[2*D_HEAD:3*D_HEAD]
    W_O = out[:, H*D_HEAD:(H+1)*D_HEAD]
    sv_OV = np.linalg.svd(W_O @ W_V, compute_uv=False)
    sv_QK = np.linalg.svd(W_Q.T @ W_K, compute_uv=False)
    return participation_ratio(sv_OV), participation_ratio(sv_QK), \
           spectral_entropy(sv_OV), spectral_entropy(sv_QK)


def build_classes_and_per_ckpt_delta():
    df = pd.read_csv(os.path.join(DATA, 'all_ablations.csv'))
    heads = df[df['perturbation_type'] == 'head'].copy()
    pivot = heads.pivot_table(index=['layer_idx','head_idx'],
                              columns='checkpoint', values='delta').abs()
    first, last = pivot.columns.min(), pivot.columns.max()
    classes = {}
    for idx, row in pivot.iterrows():
        init, fin = row[first], row[last]
        if fin < CRIT: c = 'never'
        elif init > CRIT and fin > CRIT: c = 'born'
        elif init < BORN_LOW and fin > CRIT: c = 'emergent'
        else: c = 'growing'
        classes[idx] = c
    return classes, pivot  # pivot[checkpoint] gives |delta| at that step


def snapshot_invariants(revision):
    print(f'Loading {MODEL_NAME} {revision} (CPU)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=revision, torch_dtype=torch.float32).eval()
    rows = []
    for L in range(N_LAYERS):
        for H in FIXED_HEADS:
            ov_pr, qk_pr, ov_en, qk_en = head_invariants(model, L, H)
            rows.append({'layer_idx': L, 'head_idx': H,
                         'OV_PR': ov_pr, 'QK_PR': qk_pr,
                         'OV_entropy': ov_en, 'QK_entropy': qk_en})
    del model
    return pd.DataFrame(rows).set_index(['layer_idx','head_idx'])


def main():
    classes, abs_delta_pivot = build_classes_and_per_ckpt_delta()
    class_lookup = lambda k: classes.get(k, 'unknown')

    # Phase 1a results for step143000 already exist in tables
    p1a = pd.read_csv(os.path.join(TABLES, 'phase1a_spectral_invariants.csv'))
    p1a = p1a.set_index(['layer_idx', 'head_idx'])
    ref = p1a[['OV_PR','QK_PR','OV_entropy','QK_entropy']].rename(
        columns={'OV_PR':'OV_PR_143k','QK_PR':'QK_PR_143k',
                 'OV_entropy':'OV_en_143k','QK_entropy':'QK_en_143k'})

    results_per_ckpt = {}
    for ckpt in [16000, 64000]:
        inv = snapshot_invariants(f'step{ckpt}')
        # merge with reference
        df = inv.join(ref, how='inner').reset_index()
        df['class'] = df.apply(lambda r: class_lookup((int(r['layer_idx']),
                                                       int(r['head_idx']))), axis=1)
        df['abs_delta_this_ckpt'] = df.apply(
            lambda r: abs_delta_pivot.loc[(int(r['layer_idx']), int(r['head_idx'])), ckpt], axis=1)

        out_csv = os.path.join(TABLES, f'phase1b_B_stability_step{ckpt}.csv')
        df.to_csv(out_csv, index=False)

        print(f'\n=== STEP {ckpt} ===')
        # (1) correlation with step 143k values
        print('rank corr with step 143k:')
        for ki, lab in [('OV_PR','OV_PR'), ('QK_PR','QK_PR'),
                        ('OV_entropy','OV_en'), ('QK_entropy','QK_en')]:
            r, p = sp.spearmanr(df[ki], df[lab+'_143k'])
            print(f'  {ki:<12} rho({ki}@step{ckpt}, {ki}@step143k) = {r:+.3f}  p={p:.2e}')

        # (2) mean per class at this checkpoint
        print(f'\nClass means at step {ckpt}:')
        print(f'{"class":<10} {"n":>4} {"OV_PR":>8} {"QK_PR":>8} {"OV_ent":>8} {"QK_ent":>8}')
        for cls in ['never', 'born', 'emergent', 'growing']:
            d = df[df['class'] == cls]
            if len(d) == 0: continue
            print(f'{cls:<10} {len(d):>4} {d["OV_PR"].mean():>8.2f} {d["QK_PR"].mean():>8.2f} '
                  f'{d["OV_entropy"].mean():>8.3f} {d["QK_entropy"].mean():>8.3f}')

        # (2b) active vs dead discrimination at this checkpoint
        active = df[df['class'] != 'never']
        dead = df[df['class'] == 'never']
        delta_qk = active['QK_PR'].mean() - dead['QK_PR'].mean() if len(dead) else np.nan
        print(f'  active vs dead QK_PR gap at step{ckpt}: {delta_qk:+.2f}  (at step143k: ~+20)')

        # (3) invariant vs |Delta| at SAME checkpoint
        print(f'\nrho(QK_PR@step{ckpt}, |Delta|@step{ckpt}):')
        for inv_col in ['OV_PR', 'QK_PR']:
            r_same, p_same = sp.spearmanr(df[inv_col], df['abs_delta_this_ckpt'])
            r_final_inv_same_delta = sp.spearmanr(df[inv_col+'_143k' if inv_col != 'QK_PR' else 'QK_PR_143k'],
                                                  df['abs_delta_this_ckpt']).statistic
            print(f'  {inv_col}@step{ckpt} vs |Delta|@step{ckpt}:     rho={r_same:+.3f}  p={p_same:.2e}')
            print(f'  {inv_col}@step143k vs |Delta|@step{ckpt}:       rho={r_final_inv_same_delta:+.3f}  '
                  f'(shows whether FINAL weights already carry the signal at this checkpoint)')

        results_per_ckpt[ckpt] = {
            'class_means': {c: {k: float(df[df['class']==c][k].mean())
                                for k in ['OV_PR','QK_PR','OV_entropy','QK_entropy']}
                            for c in ['never','born','emergent','growing']
                            if sum(df['class']==c) > 0},
            'active_minus_dead_QK_PR': float(delta_qk),
            'rank_corr_with_143k': {
                inv: float(sp.spearmanr(df[inv],
                    df[inv.replace('_entropy','_en')+'_143k' if '_entropy' in inv else inv+'_143k']).statistic)
                for inv in ['OV_PR','QK_PR','OV_entropy','QK_entropy']},
            'rho_QK_PR_same_ckpt_delta': float(sp.spearmanr(df['QK_PR'], df['abs_delta_this_ckpt']).statistic),
            'rho_OV_PR_same_ckpt_delta': float(sp.spearmanr(df['OV_PR'], df['abs_delta_this_ckpt']).statistic),
        }

    out_json = os.path.join(HERE, 'phase1b_B_stability_summary.json')
    with open(out_json, 'w') as f:
        json.dump({
            'generated': pd.Timestamp.utcnow().isoformat(),
            'per_checkpoint': results_per_ckpt,
        }, f, indent=2, default=str)
    print(f'\nwrote {out_json}')


if __name__ == '__main__':
    main()
