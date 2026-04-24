"""Phase 1b-E: two CPU-only tests that must run BEFORE 1.4B pre-registration.

Test 1 — Intra-class OV_PR at step 1000.
  Question: is the step-1000 OV_PR signal (rho = -0.56 on 410M) an independent
  predictor of |Delta|, or does it reduce to the active-vs-dead class label
  (Finding B)?

  Method:
    a) Spearman rho(OV_PR @ step1000, |Delta| @ step1000) WITHIN each class.
    b) OLS: log|Delta| ~ z(OV_PR_step1000) + class_dummy. Does OV_PR
       coefficient remain significant when label enters?

Test 2 — Lottery-ticket vs specialization baseline.
  Question: is the step-1000 OV_PR signal driven by initialization (lottery)
  or by the first 1000 steps of training (early specialization)?

  Method:
    a) Load Pythia 410M step0 (actual init Pythia used), compute OV_PR per head,
       correlate with |Delta| at step 143000 AND step 1000.
       - If rho_step0 ~ rho_step1000, signal is init-driven (lottery ticket).
       - If rho_step0 ~ 0, signal is training-driven (specialization).
    b) Null control: fresh random re-init of Pythia 410M config (different
       seed), compute OV_PR, correlate with actual |Delta|. Expected rho ~ 0.
       Serves as noise-floor anchor.

Runs on CPU, ~5 min for both tests.
"""
import json, os
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp
from transformers import AutoConfig, AutoModelForCausalLM
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(ROOT, 'tables')

FIXED_HEADS = [0, 3, 6, 9, 12, 15]
D_HEAD = 64
CRIT = 5e-4
BORN_LOW = 1e-4
RNG_SEED = 20260424


def participation_ratio(sigma):
    s2 = sigma ** 2
    d = np.sum(s2 ** 2)
    return float(np.sum(s2) ** 2 / d) if d > 0 else np.nan


def head_ov_pr(model, L, H):
    block = model.gpt_neox.layers[L]
    qkv = block.attention.query_key_value.weight.detach().cpu().numpy()
    out = block.attention.dense.weight.detach().cpu().numpy()
    sl = qkv[3*D_HEAD*H : 3*D_HEAD*(H+1), :]
    W_V = sl[2*D_HEAD:3*D_HEAD]
    W_O = out[:, H*D_HEAD:(H+1)*D_HEAD]
    sv = np.linalg.svd(W_O @ W_V, compute_uv=False)
    return participation_ratio(sv)


def load_410m_outcome():
    """Return classes per (L,H) and |Δ| at step 1000 and step 143000."""
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
    abs_delta_1k = pivot[1000]
    abs_delta_143k = pivot[last]
    return classes, abs_delta_1k, abs_delta_143k


# ---------------------------------------------------------------------------
# Test 1 — intra-class + absorb-the-label on step1000 OV_PR
# ---------------------------------------------------------------------------

def test1_intraclass(classes, abs_delta_1k):
    # Load step-1000 OV_PR from existing dense-scan table
    df = pd.read_csv(os.path.join(TABLES, 'phase1b_D_dense_scan.csv'))
    df = df[(df['model'] == '410M') & (df['checkpoint'] == 1000)].copy()
    df['cls'] = df.apply(lambda r: classes[(int(r['layer_idx']),
                                           int(r['head_idx']))], axis=1)
    df['log_ad'] = np.log(np.maximum(df['abs_delta_this_ckpt'].values, 1e-6))

    print('\n=== TEST 1 — INTRA-CLASS Spearman (step 1000 OV_PR vs |Delta|_step1000) ===')
    res_intra = {}
    for cls in ['never', 'born', 'emergent', 'growing']:
        d = df[df['cls'] == cls]
        if len(d) < 4:
            print(f'  {cls:<10} n={len(d):>3} too few')
            res_intra[cls] = {'n': len(d), 'rho': None, 'p': None}
            continue
        rho, p = sp.spearmanr(d['OV_PR'], d['abs_delta_this_ckpt'])
        res_intra[cls] = {'n': int(len(d)), 'rho': float(rho), 'p': float(p)}
        print(f'  {cls:<10} n={len(d):>3}  rho={rho:+.3f}  p={p:.3f}')

    # OLS: does OV_PR survive when class dummies enter?
    df['dummy_born'] = (df['cls'] == 'born').astype(float)
    df['dummy_emergent'] = (df['cls'] == 'emergent').astype(float)
    df['dummy_growing'] = (df['cls'] == 'growing').astype(float)
    df['z_OV_PR'] = (df['OV_PR'] - df['OV_PR'].mean()) / df['OV_PR'].std()

    X_inv = sm.add_constant(df['z_OV_PR'])
    X_cls = sm.add_constant(df[['dummy_born', 'dummy_emergent', 'dummy_growing']])
    X_both = sm.add_constant(pd.concat(
        [df['z_OV_PR'], df[['dummy_born', 'dummy_emergent', 'dummy_growing']]], axis=1))

    m_inv = sm.OLS(df['log_ad'], X_inv).fit()
    m_cls = sm.OLS(df['log_ad'], X_cls).fit()
    m_both = sm.OLS(df['log_ad'], X_both).fit()

    beta_inv_alone = m_inv.params['z_OV_PR']
    beta_inv_joint = m_both.params['z_OV_PR']
    p_inv_joint = m_both.pvalues['z_OV_PR']

    print('\n=== TEST 1 — OLS absorb-the-label (outcome: log|Delta|_step1000) ===')
    print(f'  invariant only: R²={m_inv.rsquared:.3f}  β_OV_PR={beta_inv_alone:+.3f}')
    print(f'  class only:     R²={m_cls.rsquared:.3f}')
    print(f'  joint:          R²={m_both.rsquared:.3f}  '
          f'β_OV_PR={beta_inv_joint:+.3f}  p_OV_PR={p_inv_joint:.3e}')
    print(f'  fractional β_OV_PR change in joint vs alone: '
          f'{(beta_inv_alone-beta_inv_joint)/beta_inv_alone*100:+.0f}%')

    return {
        'intraclass': res_intra,
        'invariant_alone': {'r2': float(m_inv.rsquared),
                            'beta_OV_PR': float(beta_inv_alone)},
        'class_alone': {'r2': float(m_cls.rsquared)},
        'joint': {'r2': float(m_both.rsquared),
                  'beta_OV_PR': float(beta_inv_joint),
                  'p_OV_PR': float(p_inv_joint)},
    }


# ---------------------------------------------------------------------------
# Test 2 — lottery ticket vs specialization
# ---------------------------------------------------------------------------

def compute_ov_pr_for_all(model, head_list):
    return {(L, H): head_ov_pr(model, L, H) for (L, H) in head_list}


def test2_lottery(classes, abs_delta_1k, abs_delta_143k):
    head_list = [(L, H) for L in range(24) for H in FIXED_HEADS]

    # (a) Pythia 410M step 0 — ACTUAL init used by training
    print('\n=== TEST 2a — Pythia 410M step 0 (actual init) ===')
    print('Loading step 0...', flush=True)
    model0 = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-410m-deduped', revision='step0',
        torch_dtype=torch.float32).eval()
    ov_pr_step0 = compute_ov_pr_for_all(model0, head_list)
    del model0

    x = np.array([ov_pr_step0[(L, H)] for (L, H) in head_list])
    y_1k = np.array([float(abs_delta_1k.get((L, H), np.nan)) for (L, H) in head_list])
    y_143k = np.array([float(abs_delta_143k.get((L, H), np.nan)) for (L, H) in head_list])

    rho_step0_vs_d1k = sp.spearmanr(x, y_1k)
    rho_step0_vs_d143k = sp.spearmanr(x, y_143k)
    print(f'  rho(OV_PR@step0, |Δ|@step1000)   = {rho_step0_vs_d1k.statistic:+.3f}  p={rho_step0_vs_d1k.pvalue:.3e}')
    print(f'  rho(OV_PR@step0, |Δ|@step143000) = {rho_step0_vs_d143k.statistic:+.3f}  p={rho_step0_vs_d143k.pvalue:.3e}')
    print('  (for comparison, at step1000 OV_PR used was itself from step1000 -> rho = -0.555)')

    # Per-class OV_PR at step 0 — is there already class separation?
    print('\n  Step-0 OV_PR class means (checks if class future is encoded at init):')
    class_means_step0 = {}
    for cls in ['never', 'born', 'emergent', 'growing']:
        vals = [ov_pr_step0[(L, H)] for (L, H) in head_list
                if classes.get((L, H)) == cls]
        m = float(np.mean(vals)) if vals else np.nan
        class_means_step0[cls] = {'n': len(vals), 'mean_OV_PR': m}
        print(f'    {cls:<10} n={len(vals):>3}  OV_PR={m:.2f}')

    # (b) Fresh random-init 410M with different seed — null control
    print('\n=== TEST 2b — fresh random init, different seed ===')
    cfg = AutoConfig.from_pretrained('EleutherAI/pythia-410m-deduped')
    torch.manual_seed(77777)  # different from whatever Pythia used
    np.random.seed(77777)
    model_fresh = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    # Ensure we actually re-initialized (transformers from_config does init)
    ov_pr_fresh = compute_ov_pr_for_all(model_fresh, head_list)
    del model_fresh

    xf = np.array([ov_pr_fresh[(L, H)] for (L, H) in head_list])
    rho_fresh_vs_d1k = sp.spearmanr(xf, y_1k)
    rho_fresh_vs_d143k = sp.spearmanr(xf, y_143k)
    print(f'  rho(OV_PR@fresh_random, |Δ|@step1000)   = {rho_fresh_vs_d1k.statistic:+.3f}  p={rho_fresh_vs_d1k.pvalue:.3e}')
    print(f'  rho(OV_PR@fresh_random, |Δ|@step143000) = {rho_fresh_vs_d143k.statistic:+.3f}  p={rho_fresh_vs_d143k.pvalue:.3e}')

    # VERDICT
    print('\n=== TEST 2 VERDICT ===')
    if abs(rho_step0_vs_d1k.statistic) > 0.25:
        verdict = 'LOTTERY-TICKET-LIKE: signal present at actual init'
    elif abs(rho_fresh_vs_d1k.statistic) > 0.25:
        verdict = 'PROBLEM: fresh random init correlates with |Δ| — spectral concentration of random weights alone predicts ablation effect (interpretation unclear)'
    else:
        verdict = 'TRAINING-DRIVEN: no init signal, correlation at step 1000 reflects first-1000-step specialization'
    print(f'  {verdict}')

    return {
        'actual_init_step0': {
            'rho_vs_d1k': float(rho_step0_vs_d1k.statistic),
            'p_vs_d1k': float(rho_step0_vs_d1k.pvalue),
            'rho_vs_d143k': float(rho_step0_vs_d143k.statistic),
            'p_vs_d143k': float(rho_step0_vs_d143k.pvalue),
            'class_means': class_means_step0,
        },
        'fresh_random_init': {
            'rho_vs_d1k': float(rho_fresh_vs_d1k.statistic),
            'p_vs_d1k': float(rho_fresh_vs_d1k.pvalue),
            'rho_vs_d143k': float(rho_fresh_vs_d143k.statistic),
            'p_vs_d143k': float(rho_fresh_vs_d143k.pvalue),
        },
        'verdict': verdict,
    }


def main():
    classes, d1k, d143k = load_410m_outcome()
    r1 = test1_intraclass(classes, d1k)
    r2 = test2_lottery(classes, d1k, d143k)

    out = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'test1_intraclass_and_absorb': r1,
        'test2_lottery_vs_specialization': r2,
    }
    with open(os.path.join(HERE, 'phase1b_E_lottery_intraclass_summary.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('\nwrote phase1b_E_lottery_intraclass_summary.json')


if __name__ == '__main__':
    main()
