"""Phase 1b-D: dense-checkpoint scan of spectral invariants vs per-head |Delta|.

Tests whether Finding D (OV_PR transient predictive signal at step 64k) is a
real training-stage-dependent phenomenon or a one-point anecdote.

For each model:
  - Load weights at each of Paper 2's 8 checkpoints (512..143000)
  - Compute 4 spectral invariants per head
  - Compute Spearman rho(invariant_@stepX, |Delta|_@stepX) per checkpoint
  - Record class means per checkpoint (classes from full trajectory)
  - Also scan step 0 (random init) as control for Finding B hardening
      -> if QK_PR class gap ~0 at step 0, discrimination is training-driven

Output: curves per invariant per model, and step-0 control means table.
"""
import json, os
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp
from transformers import AutoModelForCausalLM, AutoConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')

CHECKPOINTS = [512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]
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


def head_invariants(model, L, H, d_head):
    block = model.gpt_neox.layers[L]
    qkv = block.attention.query_key_value.weight.detach().cpu().numpy()
    out = block.attention.dense.weight.detach().cpu().numpy()
    sl = qkv[3*d_head*H : 3*d_head*(H+1), :]
    W_Q, W_K, W_V = sl[0:d_head], sl[d_head:2*d_head], sl[2*d_head:3*d_head]
    W_O = out[:, H*d_head:(H+1)*d_head]
    sv_OV = np.linalg.svd(W_O @ W_V, compute_uv=False)
    sv_QK = np.linalg.svd(W_Q.T @ W_K, compute_uv=False)
    return {
        'OV_PR': participation_ratio(sv_OV),
        'QK_PR': participation_ratio(sv_QK),
        'OV_entropy': spectral_entropy(sv_OV),
        'QK_entropy': spectral_entropy(sv_QK),
    }


def classify_from_csv(csv_path):
    df = pd.read_csv(csv_path)
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
    return classes, pivot


def scan_model(model_name, csv_path, head_list, d_head, label, run_step0=True):
    classes, abs_delta_pivot = classify_from_csv(csv_path)
    rows = []
    per_ckpt_rho = {}

    # Step 0 control FIRST (random init)
    if run_step0:
        try:
            print(f'\n[{label}] Loading step 0 (random init)...', flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision='step0', torch_dtype=torch.float32).eval()
            inv_by_head = {}
            for (L, H) in head_list:
                inv_by_head[(L, H)] = head_invariants(model, L, H, d_head)
            del model
            per_class_step0 = {}
            for cls in ['never', 'born', 'emergent', 'growing']:
                vals = []
                for (L, H), v in inv_by_head.items():
                    if classes.get((L, H)) == cls:
                        vals.append(v)
                if vals:
                    per_class_step0[cls] = {
                        k: float(np.mean([v[k] for v in vals])) for k in
                        ['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy']
                    }
                    per_class_step0[cls]['n'] = len(vals)
            for (L, H), v in inv_by_head.items():
                rows.append({'model': label, 'checkpoint': 0,
                             'layer_idx': L, 'head_idx': H,
                             'class': classes.get((L, H), 'unknown'),
                             **v,
                             'abs_delta_this_ckpt': np.nan})
            print(f'[{label}] step 0 class means QK_PR:')
            for cls in ['never', 'born', 'emergent', 'growing']:
                if cls in per_class_step0:
                    v = per_class_step0[cls]
                    print(f'    {cls:<10} n={v["n"]:>3}  QK_PR={v["QK_PR"]:.2f}  '
                          f'OV_PR={v["OV_PR"]:.2f}')
            # Active vs dead gap at step 0
            active_qk = np.mean([v['QK_PR'] for (L,H), v in inv_by_head.items()
                                 if classes.get((L, H)) != 'never'])
            dead_qk = np.mean([v['QK_PR'] for (L,H), v in inv_by_head.items()
                               if classes.get((L, H)) == 'never'])
            print(f'[{label}] step 0: active QK_PR {active_qk:.2f} - dead QK_PR {dead_qk:.2f} = {active_qk-dead_qk:+.2f}')
            print(f'[{label}] step 0: control result (gap ~0 => training-driven discrimination)')
        except Exception as e:
            print(f'[{label}] step 0 load failed: {e}', flush=True)

    # Paper-2 checkpoints with |Delta|
    for ckpt in CHECKPOINTS:
        print(f'\n[{label}] Loading step {ckpt}...', flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=f'step{ckpt}', torch_dtype=torch.float32).eval()
        except Exception as e:
            print(f'  load failed: {e}', flush=True)
            continue
        inv_by_head = {}
        for (L, H) in head_list:
            inv_by_head[(L, H)] = head_invariants(model, L, H, d_head)
        del model

        # Assemble per-head rows
        for (L, H), v in inv_by_head.items():
            try:
                abs_d_here = float(abs_delta_pivot.loc[(L, H), ckpt])
            except KeyError:
                abs_d_here = np.nan
            rows.append({'model': label, 'checkpoint': ckpt,
                         'layer_idx': L, 'head_idx': H,
                         'class': classes.get((L, H), 'unknown'),
                         **v, 'abs_delta_this_ckpt': abs_d_here})

        # Per-checkpoint rho(invariant, |Delta|)
        per_ckpt_rho[ckpt] = {}
        arr = []
        for (L, H), v in inv_by_head.items():
            try:
                abs_d_here = float(abs_delta_pivot.loc[(L, H), ckpt])
            except KeyError:
                abs_d_here = np.nan
            arr.append((v['OV_PR'], v['QK_PR'], v['OV_entropy'], v['QK_entropy'], abs_d_here))
        arr = np.array(arr)
        mask = np.isfinite(arr[:, 4])
        for j, nm in enumerate(['OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy']):
            rho, p = sp.spearmanr(arr[mask, j], arr[mask, 4])
            per_ckpt_rho[ckpt][nm] = {'rho': float(rho), 'p': float(p)}
        # active vs dead QK gap
        actives = [v['QK_PR'] for (L,H), v in inv_by_head.items()
                   if classes.get((L,H)) != 'never']
        deads = [v['QK_PR'] for (L,H), v in inv_by_head.items()
                 if classes.get((L,H)) == 'never']
        gap = np.mean(actives) - np.mean(deads) if actives and deads else np.nan
        per_ckpt_rho[ckpt]['QK_gap_active_minus_dead'] = float(gap)

        print(f'[{label}] step {ckpt}: '
              f'ρ(OV_PR, |Δ|)={per_ckpt_rho[ckpt]["OV_PR"]["rho"]:+.3f}  '
              f'ρ(QK_PR, |Δ|)={per_ckpt_rho[ckpt]["QK_PR"]["rho"]:+.3f}  '
              f'QK gap={gap:+.2f}', flush=True)

    return pd.DataFrame(rows), per_ckpt_rho


def plot_curves(rho_410, rho_160, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ckpts_arr = np.array(CHECKPOINTS)
    for ax, rho_dict, title in [(axes[0], rho_410, '410M'),
                                (axes[1], rho_160, '160M')]:
        for name, style in [('OV_PR', 'o-'), ('QK_PR', 's-'),
                            ('OV_entropy', '^--'), ('QK_entropy', 'd--')]:
            y = [rho_dict[ck][name]['rho'] for ck in CHECKPOINTS if ck in rho_dict]
            x = [ck for ck in CHECKPOINTS if ck in rho_dict]
            ax.plot(x, y, style, label=name, linewidth=1.6, markersize=6)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('training step')
        ax.set_ylabel(r'Spearman $\rho$(invariant at step t, $|\Delta|$ at step t)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('Dense scan: spectral-invariant predictive power vs training step')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    path_410 = os.path.join(DATA, 'all_ablations.csv')
    path_160 = os.path.join(DATA, 'tier2_t21_scaling_160m.csv')

    head_list_410 = [(L, H) for L in range(24) for H in [0, 3, 6, 9, 12, 15]]
    head_list_160 = [(L, H) for L in range(12) for H in range(12)]

    df_410, rho_410 = scan_model(
        'EleutherAI/pythia-410m-deduped', path_410,
        head_list_410, d_head=64, label='410M')

    df_160, rho_160 = scan_model(
        'EleutherAI/pythia-160m-deduped', path_160,
        head_list_160, d_head=64, label='160M')

    df_all = pd.concat([df_410, df_160], ignore_index=True)
    df_all.to_csv(os.path.join(TABLES, 'phase1b_D_dense_scan.csv'), index=False)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'checkpoints': CHECKPOINTS,
        '410M_per_ckpt_rho': rho_410,
        '160M_per_ckpt_rho': rho_160,
    }
    with open(os.path.join(HERE, 'phase1b_D_dense_scan_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    plot_curves(rho_410, rho_160, os.path.join(FIGS, 'phase1b_D_dense_scan.pdf'))

    print('\n=== DENSE-SCAN CURVE: rho(invariant, |Delta|) per checkpoint ===\n')
    print(f'{"model":<7} {"step":>6} {"OV_PR":>8} {"QK_PR":>8} {"OV_ent":>8} {"QK_ent":>8}  QK gap')
    print('-' * 70)
    for lab, rho_dict in [('410M', rho_410), ('160M', rho_160)]:
        for ck in CHECKPOINTS:
            if ck not in rho_dict: continue
            r = rho_dict[ck]
            print(f'{lab:<7} {ck:>6} '
                  f'{r["OV_PR"]["rho"]:>+8.3f} {r["QK_PR"]["rho"]:>+8.3f} '
                  f'{r["OV_entropy"]["rho"]:>+8.3f} {r["QK_entropy"]["rho"]:>+8.3f}  '
                  f'{r["QK_gap_active_minus_dead"]:+6.2f}')


if __name__ == '__main__':
    main()
