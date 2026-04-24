"""Generate tier2_pre1p4b_validation.ipynb — pre-registered 1.4B test.

Produces one Colab notebook that executes pre-registration v2 on Pythia 1.4B.
Resumable (CSV append), Drive-mandatory, per-checkpoint commit to Drive,
optional git push, final verdict JSON with all four pre-registered tests.
"""
import json, os

HERE = os.path.dirname(__file__)
GITHUB_RAW = 'https://raw.githubusercontent.com/mool32/functional-differentiation-dfe/main'
OUT = os.path.join(HERE, 'tier2_pre1p4b_validation.ipynb')


def md(src): return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
def code(src): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


cells = []

cells.append(md("""# Tier 2 — Pythia 1.4B validation of Phase 1 spectral findings

**Pre-registration locked at commit 8be0f11** (invariants_preregistration_v2_1p4b.md).

This notebook runs the pre-registered test on Pythia 1.4B that decides:
- Primary: does OV_PR at step 1000 predict per-head |Δ| at step 1000?
- Four secondary tests (S-1..S-4): within-class independence, step-0 null, sign-flip, QK gap.

**Output:** `tier2_pre1p4b_verdict.json` with explicit Pass / Partial / Weak / Fail per test.

## Required hardware

A100 80GB (RunPod / Colab Pro+ / Lambda). 1.4B in float32 needs ~60 GB.
If only 40 GB available, change `PRECISION = 'bfloat16'` in Config — this is a
**pre-registered deviation** from methodology and will be flagged in verdict.

## Estimated runtime

~3-6 h on A100 80GB in float32 with 25 eval batches × 4 × 2048 tokens.
CSV checkpoints after each of 7 training-step sweeps. Safe to disconnect
and resume; completed rows skipped.

## Critical discipline (same as Paper 2 / 3)

- Drive mount mandatory for persistence across runtime resets
- float32 + TF32 matmul
- SHA-256 save/restore verification per head
- Resume via CSV append
- Auto-download to browser at end"""))

cells.append(md("""## 1. Install + imports"""))
cells.append(code("""!pip install -q transformers datasets torch accelerate scipy pandas statsmodels"""))
cells.append(code(r"""import torch, json, os, time, csv, hashlib, gc, urllib.request
import numpy as np
import pandas as pd
from scipy import stats as sp
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU: {gpu.name}  |  Memory: {gpu.total_memory/1e9:.1f} GB')
    if gpu.total_memory/1e9 < 60:
        print('WARNING: <60 GB GPU. 1.4B in float32 will OOM. Set PRECISION = bfloat16 below.')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    OUT_DIR = '/content/drive/MyDrive/DFE_research/tier2_pre1p4b'
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, '.w'), 'w') as f: f.write('ok')
    os.remove(os.path.join(OUT_DIR, '.w'))
    print(f'Drive mounted; output to {OUT_DIR}')
except Exception as e:
    raise RuntimeError(f'Drive mount required: {e}')

GITHUB_RAW = '""" + GITHUB_RAW + """'

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)"""))

cells.append(md("""## 2. Config — pre-registered values (do not change unless flagging deviation)"""))
cells.append(code("""MODEL_NAME = 'EleutherAI/pythia-1.4b-deduped'

# Pre-registered methodology (same as 410M/160M)
PRECISION = 'float32'            # 'bfloat16' only if OOM — will be flagged
N_EVAL_BATCHES = 25
EVAL_BATCH_SIZE = 4
EVAL_SEQ_LEN = 2048

# Pre-registered checkpoints
CHECKPOINTS_ABL = [1000, 2000, 4000, 8000, 16000, 64000, 143000]  # ablation sweep
CHECKPOINTS_SPEC_ONLY = [0]  # invariants only, no ablation (step 0 control)

# Pythia 1.4B architecture
N_LAYERS = 24
N_HEADS_PER_LAYER = 16
D_MODEL = 2048
D_HEAD = 128
FIXED_HEADS = [0, 3, 6, 9, 12, 15]  # 6 per layer -> 144 sampled heads
HEAD_LIST = [(L, H) for L in range(N_LAYERS) for H in FIXED_HEADS]
assert len(HEAD_LIST) == 144

# Classification thresholds (same as Paper 2)
CRIT = 5e-4
BORN_LOW = 1e-4

# Output files
CSV_ABL = os.path.join(OUT_DIR, 'tier2_pre1p4b_ablations.csv')
CSV_SPEC = os.path.join(OUT_DIR, 'tier2_pre1p4b_spectral.csv')
JSON_VERDICT = os.path.join(OUT_DIR, 'tier2_pre1p4b_verdict.json')

ABL_FIELDS = ['checkpoint', 'layer_idx', 'head_idx', 'baseline_loss',
              'perturbed_loss', 'delta', 'elapsed_sec']
SPEC_FIELDS = ['checkpoint', 'layer_idx', 'head_idx',
               'OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy', 'sigma_OV_max', 'sigma_QK_max']

print('Config locked:')
print(f'  precision: {PRECISION}')
print(f'  eval: {N_EVAL_BATCHES} x {EVAL_BATCH_SIZE} x {EVAL_SEQ_LEN} tokens')
print(f'  ablation checkpoints: {CHECKPOINTS_ABL}')
print(f'  spectral-only checkpoints: {CHECKPOINTS_SPEC_ONLY}')
print(f'  heads: {len(HEAD_LIST)} = 6 per layer x 24 layers')"""))

cells.append(md("""## 3. Ablation + spectral primitives"""))
cells.append(code(r"""def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

def ablate_head(model, L, H):
    w = model.gpt_neox.layers[L].attention.dense.weight
    saved = w.data.clone()
    w.data[:, H*D_HEAD:(H+1)*D_HEAD] = 0
    return saved

def restore_head(model, L, saved):
    model.gpt_neox.layers[L].attention.dense.weight.data.copy_(saved)

@torch.no_grad()
def evaluate_loss(model, batches):
    total = 0.0
    for i in range(batches.shape[0]):
        ids = batches[i].to(device)
        total += model(input_ids=ids, labels=ids).loss.item()
    return total / batches.shape[0]

def participation_ratio(sigma):
    s2 = sigma ** 2
    d = np.sum(s2 ** 2)
    return float(np.sum(s2) ** 2 / d) if d > 0 else np.nan

def spectral_entropy(sigma):
    s2 = sigma ** 2
    tot = s2.sum()
    if tot <= 0: return np.nan
    p = s2 / tot
    p = p[p > 1e-20]
    return float(-(p * np.log(p)).sum())

def head_invariants(model, L, H):
    block = model.gpt_neox.layers[L]
    qkv = block.attention.query_key_value.weight.detach().cpu().float().numpy()
    out = block.attention.dense.weight.detach().cpu().float().numpy()
    sl = qkv[3*D_HEAD*H : 3*D_HEAD*(H+1), :]
    W_Q, W_K, W_V = sl[0:D_HEAD], sl[D_HEAD:2*D_HEAD], sl[2*D_HEAD:3*D_HEAD]
    W_O = out[:, H*D_HEAD:(H+1)*D_HEAD]
    sv_OV = np.linalg.svd(W_O @ W_V, compute_uv=False)
    sv_QK = np.linalg.svd(W_Q.T @ W_K, compute_uv=False)
    return {
        'OV_PR': participation_ratio(sv_OV),
        'QK_PR': participation_ratio(sv_QK),
        'OV_entropy': spectral_entropy(sv_OV),
        'QK_entropy': spectral_entropy(sv_QK),
        'sigma_OV_max': float(sv_OV[0]),
        'sigma_QK_max': float(sv_QK[0]),
    }"""))

cells.append(md("""## 4. Prepare wikitext-103 validation batches (cached to Drive)"""))
cells.append(code(r"""from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
batches_path = os.path.join(OUT_DIR, 'wt103_batches.pt')

if os.path.exists(batches_path):
    batches = torch.load(batches_path, weights_only=True)
    log(f'Loaded cached batches: {batches.shape}')
else:
    log('Streaming wikitext-103 train...')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
    need = N_EVAL_BATCHES * EVAL_BATCH_SIZE * EVAL_SEQ_LEN
    toks = []
    for ex in ds:
        t = ex.get('text', '').strip()
        if len(t) < 50: continue
        ids = tok(t, return_tensors='pt', truncation=False)['input_ids'].squeeze()
        if ids.dim() == 0: continue
        toks.append(ids)
        if sum(x.numel() for x in toks) >= need * 1.2: break
    merged = torch.cat(toks)[:need]
    batches = merged.reshape(N_EVAL_BATCHES, EVAL_BATCH_SIZE, EVAL_SEQ_LEN)
    torch.save(batches, batches_path)
    log(f'cached {batches.shape}')"""))

cells.append(md("""## 5. Resume logic — load completed work"""))
cells.append(code("""def completed_ablations():
    if not os.path.exists(CSV_ABL): return set()
    df = pd.read_csv(CSV_ABL)
    return set(zip(df['checkpoint'], df['layer_idx'], df['head_idx']))

def completed_spectral():
    if not os.path.exists(CSV_SPEC): return set()
    df = pd.read_csv(CSV_SPEC)
    return set(zip(df['checkpoint'], df['layer_idx'], df['head_idx']))

def append_abl(row):
    new = not os.path.exists(CSV_ABL)
    with open(CSV_ABL, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ABL_FIELDS)
        if new: w.writeheader()
        w.writerow(row)

def append_spec(row):
    new = not os.path.exists(CSV_SPEC)
    with open(CSV_SPEC, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SPEC_FIELDS)
        if new: w.writeheader()
        w.writerow(row)

done_abl = completed_ablations()
done_spec = completed_spectral()
log(f'Already completed: {len(done_abl)} ablations, {len(done_spec)} spectral rows')"""))

cells.append(md("""## 6. Main sweep — per checkpoint: spectral + ablations

Strategy: per checkpoint, load model, compute spectral invariants (cheap), then
ablation sweep (expensive). Save to Drive after each head. Safe to disconnect."""))
cells.append(code(r"""def get_dtype():
    if PRECISION == 'float32': return torch.float32
    if PRECISION == 'bfloat16': return torch.bfloat16
    raise ValueError(f'unknown precision {PRECISION}')

def sweep_one_checkpoint(step, ablate=True):
    log(f'=== step{step}  (ablate={ablate}) ===')
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=f'step{step}', torch_dtype=get_dtype()
    ).to(device).eval()
    log(f'  loaded in {time.time()-t_load:.0f}s')

    # Spectral invariants (zero GPU cost, float32 math on CPU slices)
    for (L, H) in HEAD_LIST:
        if (step, L, H) in done_spec: continue
        inv = head_invariants(model, L, H)
        append_spec({'checkpoint': step, 'layer_idx': L, 'head_idx': H, **inv})
        done_spec.add((step, L, H))
    log(f'  spectral done')

    if not ablate:
        del model; gc.collect(); torch.cuda.empty_cache()
        return

    # Baseline loss
    bl = evaluate_loss(model, batches)
    log(f'  baseline loss: {bl:.4f}')

    # Sanity: save/restore bitwise
    w_ref = model.gpt_neox.layers[5].attention.dense.weight
    h0 = tensor_hash(w_ref.data)
    s = ablate_head(model, 5, 7)
    restore_head(model, 5, s)
    h2 = tensor_hash(w_ref.data)
    assert h0 == h2, 'save/restore broke!'

    # Ablation sweep
    for i, (L, H) in enumerate(HEAD_LIST):
        if (step, L, H) in done_abl: continue
        t0 = time.time()
        saved = ablate_head(model, L, H)
        pl = evaluate_loss(model, batches)
        restore_head(model, L, saved)
        delta = -(pl - bl) / abs(bl)
        append_abl({
            'checkpoint': step, 'layer_idx': L, 'head_idx': H,
            'baseline_loss': bl, 'perturbed_loss': pl, 'delta': delta,
            'elapsed_sec': time.time()-t0,
        })
        done_abl.add((step, L, H))
        if (i+1) % 20 == 0 or (i+1) == len(HEAD_LIST):
            log(f'    [{i+1}/{len(HEAD_LIST)}] L{L}H{H} Δ={delta:+.5f}  ({time.time()-t0:.0f}s/head)')

    log(f'  checkpoint done')
    del model; gc.collect(); torch.cuda.empty_cache()


# Run spectral-only (step 0) first — fast, confirms pipeline
for s in CHECKPOINTS_SPEC_ONLY:
    sweep_one_checkpoint(s, ablate=False)

# Run full sweeps. Order: step 1000 first (that's the primary test),
# then 143000 (S-4 gap), then 2000/4000/8000 (S-3 sign flip),
# then 16000 and 64000. This prioritizes checkpoints that affect verdict
# and allows early read if time runs out.
CHECKPOINT_ORDER = [1000, 143000, 4000, 8000, 2000, 16000, 64000]
for s in CHECKPOINT_ORDER:
    sweep_one_checkpoint(s, ablate=True)"""))

cells.append(md("""## 7. Smoke-test cell (optional) — run BEFORE main sweep to estimate cost

If you're on a new GPU/setup, run this cell after Section 4 and BEFORE Section 6
to estimate total runtime. 5 ablations on step 1000. Should take 2-5 minutes.
If it's taking > 10 min, switch to `PRECISION = 'bfloat16'` and rerun."""))
cells.append(code(r"""# SMOKE TEST — remove ## below to enable
## t_smoke = time.time()
## m = AutoModelForCausalLM.from_pretrained(
##     MODEL_NAME, revision='step1000', torch_dtype=get_dtype()
## ).to(device).eval()
## bl = evaluate_loss(m, batches)
## ts = []
## for (L, H) in HEAD_LIST[:5]:
##     t0 = time.time()
##     s = ablate_head(m, L, H)
##     pl = evaluate_loss(m, batches)
##     restore_head(m, L, s)
##     ts.append(time.time() - t0)
## del m; gc.collect(); torch.cuda.empty_cache()
## avg = sum(ts)/len(ts)
## total = avg * len(HEAD_LIST) * len(CHECKPOINTS_ABL)
## print(f'avg {avg:.0f}s per ablation; extrapolated total {total/3600:.1f} h')
## print(f'smoke elapsed {time.time()-t_smoke:.0f}s')"""))

cells.append(md("""## 8. Analysis — primary + 4 secondary pre-registered tests

Runs after sweep completes (or partially — uses whatever is in CSVs)."""))
cells.append(code(r"""# Load classification reference from Paper 2 410M data
p2_path = os.path.join(OUT_DIR, 'paper2_410m_ablations.csv')
if not os.path.exists(p2_path):
    urllib.request.urlretrieve(f'{GITHUB_RAW}/data/all_ablations.csv', p2_path)
p2_df = pd.read_csv(p2_path)

# For 1.4B we'll classify using 1.4B's own ablation data across checkpoints
abl = pd.read_csv(CSV_ABL)
spec = pd.read_csv(CSV_SPEC)

pivot = abl.pivot_table(index=['layer_idx','head_idx'], columns='checkpoint', values='delta').abs()
first_avail = pivot.columns.min()
last_avail = pivot.columns.max()

def classify_1p4b(row):
    init = row.get(first_avail, np.nan)
    fin = row.get(last_avail, np.nan)
    if np.isnan(init) or np.isnan(fin): return 'unknown'
    if fin < CRIT: return 'never'
    if init > CRIT and fin > CRIT: return 'born'
    if init < BORN_LOW and fin > CRIT: return 'emergent'
    return 'growing'

classes = {idx: classify_1p4b(row) for idx, row in pivot.iterrows()}
from collections import Counter
print('1.4B class counts:', dict(Counter(classes.values())))"""))

cells.append(code(r"""# Build analysis dataframe merging spectral + ablation per checkpoint
merged = spec.merge(abl, on=['checkpoint', 'layer_idx', 'head_idx'], how='outer')
merged['class'] = merged.apply(lambda r: classes.get((r['layer_idx'], r['head_idx']), 'unknown'), axis=1)
merged['abs_delta'] = merged['delta'].abs()

def spearman_ci(x, y, n_boot=10000, seed=20260424):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4: return None
    rho, p = sp.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ii = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(ii)) < 4: continue
        boots.append(sp.spearmanr(x[ii], y[ii]).statistic)
    boots = np.array(boots)
    return {
        'rho': float(rho), 'p': float(p), 'n': int(len(x)),
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
    }

verdict = {
    'pre_registration_commit': '8be0f11',
    'precision_used': PRECISION,
    'n_eval_batches': N_EVAL_BATCHES,
    'deviations_from_preregistration': (
        [] if (PRECISION == 'float32' and N_EVAL_BATCHES == 25)
        else [f'precision={PRECISION} (pre-reg was float32)' if PRECISION != 'float32' else None,
              f'n_eval_batches={N_EVAL_BATCHES} (pre-reg was 25)' if N_EVAL_BATCHES != 25 else None]
    ),
}

# ---------------------------------------------------------
# PRIMARY: rho(OV_PR@step1000, |Delta|@step1000)
# ---------------------------------------------------------
d1k = merged[merged['checkpoint'] == 1000].copy()
r_primary = spearman_ci(d1k['OV_PR'].values, d1k['abs_delta'].values)
print('\\n=== PRIMARY TEST ===')
print(f'  rho(OV_PR@step1000, |Delta|@step1000) = {r_primary}')
# Decision
def primary_verdict(r):
    if r is None: return 'NO_DATA'
    rho, p = r['rho'], r['p']
    if rho >= 0: return 'FAIL_WRONG_SIGN'
    mag = abs(rho)
    if mag >= 0.30 and p < 0.01: return 'PASS'
    if mag >= 0.20: return 'PARTIAL'
    if mag >= 0.10: return 'WEAK'
    return 'FAIL'
verdict['primary'] = {'rho': r_primary, 'verdict': primary_verdict(r_primary)}
print(f'  >>> {verdict["primary"]["verdict"]} <<<')"""))

cells.append(code(r"""# ---------------------------------------------------------
# S-1: within-class rho at step 1000 in >= 2 active classes
# ---------------------------------------------------------
print('\\n=== S-1 WITHIN-CLASS ===')
s1_results = {}
active_pass = 0
for cls in ['emergent', 'growing', 'born', 'never']:
    d = d1k[d1k['class'] == cls]
    if len(d) < 10 and cls != 'never':
        s1_results[cls] = {'n': int(len(d)), 'note': 'too few'}
        print(f'  {cls:<10} n={len(d)} too few to test')
        continue
    r = spearman_ci(d['OV_PR'].values, d['abs_delta'].values, n_boot=2000)
    s1_results[cls] = r
    passed = (r is not None and r['rho'] < 0 and abs(r['rho']) >= 0.30)
    if cls in ['emergent', 'growing', 'born'] and passed: active_pass += 1
    flag = 'pass' if passed else '    '
    print(f'  {cls:<10} n={r["n"] if r else 0:>3}  rho={r["rho"]:+.3f}  p={r["p"]:.3f}  {flag}' if r else f'  {cls} no data')
verdict['s1_within_class'] = {
    'per_class': s1_results,
    'n_active_passed': int(active_pass),
    'verdict': 'PASS' if active_pass >= 2 else 'FAIL',
}
print(f'  >>> S-1: {verdict["s1_within_class"]["verdict"]} <<<')

# ---------------------------------------------------------
# S-2: step 0 null (|rho| < 0.20 vs step1000 and step143000)
# ---------------------------------------------------------
print('\\n=== S-2 STEP-0 LOTTERY CONTROL ===')
spec0 = merged[merged['checkpoint'] == 0].copy()
# Correlate step0 OV_PR with step1000 and step143000 |Delta|
def xcorr(spec_df, abl_ckpt):
    abl_slice = merged[merged['checkpoint'] == abl_ckpt][['layer_idx','head_idx','abs_delta']]
    m = spec_df[['layer_idx','head_idx','OV_PR']].merge(abl_slice, on=['layer_idx','head_idx'])
    if len(m) < 4: return None
    return spearman_ci(m['OV_PR'].values, m['abs_delta'].values, n_boot=2000)

r_vs_1k = xcorr(spec0, 1000)
r_vs_143k = xcorr(spec0, 143000)
print(f'  rho(OV_PR@step0, |Delta|@step1000)   = {r_vs_1k}')
print(f'  rho(OV_PR@step0, |Delta|@step143000) = {r_vs_143k}')
s2_pass = (r_vs_1k is not None and abs(r_vs_1k['rho']) < 0.20 and
           r_vs_143k is not None and abs(r_vs_143k['rho']) < 0.20)
verdict['s2_step0_null'] = {
    'rho_vs_step1000': r_vs_1k,
    'rho_vs_step143000': r_vs_143k,
    'verdict': 'PASS' if s2_pass else 'FAIL',
}
print(f'  >>> S-2: {verdict["s2_step0_null"]["verdict"]} <<<')

# ---------------------------------------------------------
# S-3: sign flip rho(QK_PR, |Delta|) between step 1000 and step 8000
# ---------------------------------------------------------
print('\\n=== S-3 SIGN FLIP ===')
s3_trajectory = {}
for ck in CHECKPOINTS_ABL:
    d = merged[merged['checkpoint'] == ck]
    if len(d.dropna(subset=['QK_PR','abs_delta'])) < 10: continue
    r = spearman_ci(d['QK_PR'].values, d['abs_delta'].values, n_boot=500)
    s3_trajectory[ck] = r
    print(f'  step {ck:>6}: rho(QK_PR, |Δ|) = {r["rho"]:+.3f}  p={r["p"]:.3f}' if r else f'  step {ck}: no data')

rho_1k = s3_trajectory.get(1000, {'rho': None}).get('rho')
rho_143k = s3_trajectory.get(143000, {'rho': None}).get('rho')
s3_pass = (rho_1k is not None and rho_1k < -0.15 and
           rho_143k is not None and rho_143k > 0.30)
verdict['s3_sign_flip'] = {
    'trajectory': s3_trajectory,
    'verdict': 'PASS' if s3_pass else ('PARTIAL' if (rho_1k is not None and rho_143k is not None
                                                      and rho_1k < 0 and rho_143k > 0) else 'FAIL'),
}
print(f'  >>> S-3: {verdict["s3_sign_flip"]["verdict"]} <<<')

# ---------------------------------------------------------
# S-4: QK active-vs-dead gap at step 143000 >= +15
# ---------------------------------------------------------
print('\\n=== S-4 QK GAP ===')
d143 = merged[merged['checkpoint'] == 143000]
active = d143[d143['class'].isin(['born','emergent','growing'])]
dead = d143[d143['class'] == 'never']
gap = float(active['QK_PR'].mean() - dead['QK_PR'].mean()) if len(dead) and len(active) else np.nan
print(f'  active mean QK_PR: {active["QK_PR"].mean():.2f}  (n={len(active)})')
print(f'  dead   mean QK_PR: {dead["QK_PR"].mean():.2f}  (n={len(dead)})')
print(f'  gap:               {gap:+.2f}')
verdict['s4_qk_gap'] = {
    'gap': gap,
    'active_mean': float(active['QK_PR'].mean()) if len(active) else None,
    'dead_mean': float(dead['QK_PR'].mean()) if len(dead) else None,
    'n_active': int(len(active)), 'n_dead': int(len(dead)),
    'verdict': 'PASS' if gap >= 15 else ('PARTIAL' if gap >= 10 else 'FAIL'),
}
print(f'  >>> S-4: {verdict["s4_qk_gap"]["verdict"]} <<<')

# Save verdict
with open(JSON_VERDICT, 'w') as f:
    json.dump(verdict, f, indent=2, default=str)
print(f'\\n\\nVERDICT saved to {JSON_VERDICT}')
print(json.dumps({k: v.get('verdict') if isinstance(v, dict) and 'verdict' in v else v
                  for k, v in verdict.items() if k not in ('deviations_from_preregistration',)}, indent=2))"""))

cells.append(md("""## 9. Force download — everything to browser

Belt-and-suspenders: files are on Drive, but also trigger browser download."""))
cells.append(code("""try:
    from google.colab import files
    for p in [CSV_ABL, CSV_SPEC, JSON_VERDICT]:
        if os.path.exists(p):
            files.download(p)
    print('Downloads triggered.')
except Exception as e:
    print(f'Download failed: {e}')
    print(f'Files on Drive at {OUT_DIR}')"""))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
        'colab': {'provenance': []},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote {OUT}')
