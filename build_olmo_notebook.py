"""Build tier2_olmo_validation.ipynb — cross-architecture replication on OLMo-2 1B.

One-shot Colab notebook implementing pre-registration v3
(invariants_preregistration_v3_olmo.md).

Outputs: paper/tier2_olmo_validation.ipynb
"""
import json, os

HERE = os.path.dirname(__file__)
GITHUB_RAW = 'https://raw.githubusercontent.com/mool32/functional-differentiation-dfe/main'
OUT = os.path.join(HERE, 'tier2_olmo_validation.ipynb')


def md(src): return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
def code(src): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


cells = []

cells.append(md("""# Tier 2 — Cross-architecture validation on OLMo-2 1B

**Pre-registration:** `invariants_preregistration_v3_olmo.md` (commit hash recorded in verdict JSON).

Tests whether the ML early-window invariant (established on Pythia 160M/410M/1.4B)
replicates on an architecturally-independent transformer:
`allenai/OLMo-2-0425-1B-early-training`.

**Binding rules (accepted 2026-04-24):**
1. Direction pre-registered absolute (negative). Positive = FAIL, no reformulation.
2. Single primary test, single decision.
3. Numeric decision rule locked.
4. Null is legitimate outcome.
5. No post-hoc rescue.
6. Pre-reg commit hash baked into verdict output.

**Primary test:** ρ(OV_PR_h @ step 1000, |Δ_h| @ step 1000) < 0, |ρ| ≥ 0.30 for PASS.
Pythia reference: ρ = −0.42 / −0.55 / −0.48 across 160M/410M/1.4B.

## Required hardware

A100 80GB. Model is 1.24B params; fits in float32 with room for activations.

## Estimated runtime

~5-8 hours full sweep. CSV append-resume; safe across disconnects.
Checkpoint ordering prioritized: step 1000 first (primary), then final (step 37000),
then sign-flip window."""))

cells.append(md("""## 1. Install + imports"""))
cells.append(code("""!pip install -q transformers datasets torch accelerate scipy pandas huggingface_hub"""))
cells.append(code(r"""import torch, json, os, time, csv, hashlib, gc, urllib.request
import numpy as np
import pandas as pd
from scipy import stats as sp
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU: {gpu.name}  |  Memory: {gpu.total_memory/1e9:.1f} GB')
    if gpu.total_memory/1e9 < 35:
        print('WARNING: <35 GB GPU. OLMo-2 1B in float32 may be tight. Consider bfloat16 if OOM.')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    OUT_DIR = '/content/drive/MyDrive/DFE_research/tier2_olmo'
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, '.w'), 'w') as f: f.write('ok')
    os.remove(os.path.join(OUT_DIR, '.w'))
    print(f'Drive mounted; output to {OUT_DIR}')
except Exception as e:
    raise RuntimeError(f'Drive mount required: {e}')

GITHUB_RAW = '""" + GITHUB_RAW + """'
PRE_REG_COMMIT = 'tbd — will be filled in by verdict JSON from local git hash'

def log(msg):
    print(f'[{time.strftime(\"%H:%M:%S\")}] {msg}', flush=True)"""))

cells.append(md("""## 2. Config (locked per pre-registration v3)"""))
cells.append(code("""MODEL_NAME = 'allenai/OLMo-2-0425-1B-early-training'

# Pre-registered methodology
PRECISION = 'float32'             # fallback to bfloat16 only on OOM, flagged as deviation
N_EVAL_BATCHES = 25
EVAL_BATCH_SIZE = 4
EVAL_SEQ_LEN = 2048

# Pre-registered checkpoints
CHECKPOINTS_ABL = [1000, 2000, 4000, 8000, 16000, 37000]  # ablation + spectral
CHECKPOINTS_SPEC_ONLY = [0]                                 # step 0 control

# Output files
CSV_ABL = os.path.join(OUT_DIR, 'tier2_olmo_ablations.csv')
CSV_SPEC = os.path.join(OUT_DIR, 'tier2_olmo_spectral.csv')
JSON_VERDICT = os.path.join(OUT_DIR, 'tier2_olmo_verdict.json')

ABL_FIELDS = ['checkpoint', 'layer_idx', 'head_idx', 'baseline_loss',
              'perturbed_loss', 'delta', 'elapsed_sec']
SPEC_FIELDS = ['checkpoint', 'layer_idx', 'head_idx',
               'OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy',
               'sigma_OV_max', 'sigma_QK_max']

print('Pre-registered config:')
print(f'  precision: {PRECISION}')
print(f'  eval: {N_EVAL_BATCHES} x {EVAL_BATCH_SIZE} x {EVAL_SEQ_LEN} tokens')
print(f'  ablation checkpoints: {CHECKPOINTS_ABL}')
print(f'  spectral-only checkpoints: {CHECKPOINTS_SPEC_ONLY}')"""))

cells.append(md("""## 3. Resolve exact revision names via list_repo_refs

OLMo-2 uses naming `stage1-step{N}-tokens{M}B`. Exact `M` values vary (~2.1B per 1000 steps
but not exact). Query HuggingFace to find the actual revision matching each desired step.

If a requested step has no available revision, it will be logged as MISSING and skipped."""))
cells.append(code(r"""log('Fetching HuggingFace revision list...')
refs = list_repo_refs(MODEL_NAME)
all_branches = [b.name for b in refs.branches]
log(f'Total revisions available: {len(all_branches)}')

# Match revision by step prefix
def resolve_step(step):
    target = f'stage1-step{step}-'
    for b in all_branches:
        if b.startswith(target):
            return b
    return None

CHECKPOINT_REVISIONS = {}
for s in CHECKPOINTS_SPEC_ONLY + CHECKPOINTS_ABL:
    rev = resolve_step(s)
    CHECKPOINT_REVISIONS[s] = rev
    if rev is None:
        log(f'  step{s}: NOT FOUND in repo')
    else:
        log(f'  step{s} -> {rev}')

missing = [s for s, r in CHECKPOINT_REVISIONS.items() if r is None]
if missing:
    print(f'\\nWARNING: {len(missing)} requested checkpoints missing: {missing}')
    print('Will skip these in run. Primary test requires step 1000, step 0, step 37000.')"""))

cells.append(md("""## 4. Architecture auto-detection"""))
cells.append(code(r"""log('Loading final checkpoint to detect architecture...')
_final_rev = CHECKPOINT_REVISIONS.get(37000) or all_branches[-1]
_tmp = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=_final_rev, torch_dtype=torch.float32)
N_LAYERS = _tmp.config.num_hidden_layers
N_HEADS = _tmp.config.num_attention_heads
HIDDEN = _tmp.config.hidden_size
D_HEAD = HIDDEN // N_HEADS
MODEL_CLASS = _tmp.__class__.__name__
log(f'arch: {MODEL_CLASS}, {N_LAYERS} layers × {N_HEADS} heads, hidden={HIDDEN}, d_head={D_HEAD}')

# Inspect naming convention
_first_layer = _tmp.model.layers[0]
log(f'first-layer attn children: {list(_first_layer.self_attn._modules.keys())}')

# Decide head sampling per pre-reg
if N_LAYERS * N_HEADS <= 300:
    HEAD_LIST = [(L, H) for L in range(N_LAYERS) for H in range(N_HEADS)]
    HEAD_SAMPLING = 'ALL'
else:
    FIXED_HEADS = [0, 3, 6, 9, 12, 15][:max(1, min(6, N_HEADS))]
    HEAD_LIST = [(L, H) for L in range(N_LAYERS) for H in FIXED_HEADS]
    HEAD_SAMPLING = 'SAMPLED_6_PER_LAYER'

log(f'head sampling: {HEAD_SAMPLING} -> {len(HEAD_LIST)} total heads')

del _tmp
gc.collect()
torch.cuda.empty_cache()"""))

cells.append(md("""## 5. Primitives (ablation + spectral invariants)"""))
cells.append(code(r"""def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

def get_layer_attn(model, L):
    return model.model.layers[L].self_attn

def ablate_head(model, L, H):
    attn = get_layer_attn(model, L)
    w = attn.o_proj.weight
    saved = w.data.clone()
    w.data[:, H*D_HEAD:(H+1)*D_HEAD] = 0
    return saved

def restore_head(model, L, saved):
    get_layer_attn(model, L).o_proj.weight.data.copy_(saved)

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
    attn = get_layer_attn(model, L)
    W_Q = attn.q_proj.weight.detach().cpu().float().numpy()[H*D_HEAD:(H+1)*D_HEAD, :]
    W_K = attn.k_proj.weight.detach().cpu().float().numpy()[H*D_HEAD:(H+1)*D_HEAD, :]
    W_V = attn.v_proj.weight.detach().cpu().float().numpy()[H*D_HEAD:(H+1)*D_HEAD, :]
    W_O = attn.o_proj.weight.detach().cpu().float().numpy()[:, H*D_HEAD:(H+1)*D_HEAD]

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

cells.append(md("""## 6. Prepare wikitext-103 batches (cached on Drive)"""))
cells.append(code(r"""from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
batches_path = os.path.join(OUT_DIR, 'wt103_batches.pt')

if os.path.exists(batches_path):
    batches = torch.load(batches_path, weights_only=True)
    log(f'Loaded cached batches: {batches.shape}')
else:
    log('Streaming wikitext-103...')
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
    log(f'Cached {batches.shape}')"""))

cells.append(md("""## 7. Resume logic"""))
cells.append(code(r"""def completed_ablations():
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

cells.append(md("""## 8. Smoke test — run BEFORE main sweep to estimate cost

Uncomment (remove ## below) to estimate full runtime before committing the sweep."""))
cells.append(code(r"""## t_smoke = time.time()
## _rev = CHECKPOINT_REVISIONS.get(1000)
## m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=_rev, torch_dtype=torch.float32).to(device).eval()
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
## print(f'avg {avg:.1f}s per ablation; extrapolated total {total/3600:.1f} h')"""))

cells.append(md("""## 9. Main sweep

Order: step 1000 (primary), step 37000 (final), then 2000/4000/8000/16000 (trajectory),
then step 0 (spectral only, no ablation). Primary verdict computable after step 1000 alone."""))
cells.append(code(r"""def get_dtype():
    return torch.float32 if PRECISION == 'float32' else torch.bfloat16

def sweep_one_checkpoint(step, ablate=True):
    rev = CHECKPOINT_REVISIONS.get(step)
    if rev is None:
        log(f'SKIP step{step} (revision not available)')
        return
    log(f'=== step{step}  rev={rev}  ablate={ablate} ===')
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=rev, torch_dtype=get_dtype()
    ).to(device).eval()
    log(f'  loaded in {time.time()-t_load:.0f}s')

    # Spectral invariants (CPU-side, fast)
    for (L, H) in HEAD_LIST:
        if (step, L, H) in done_spec: continue
        inv = head_invariants(model, L, H)
        append_spec({'checkpoint': step, 'layer_idx': L, 'head_idx': H, **inv})
        done_spec.add((step, L, H))
    log(f'  spectral done')

    if not ablate:
        del model; gc.collect(); torch.cuda.empty_cache()
        return

    # Baseline
    bl = evaluate_loss(model, batches)
    log(f'  baseline loss: {bl:.4f}')

    # Sanity: bitwise save/restore
    attn5 = get_layer_attn(model, min(5, N_LAYERS-1))
    w_ref = attn5.o_proj.weight
    h0 = tensor_hash(w_ref.data)
    H0 = 0
    s = ablate_head(model, min(5, N_LAYERS-1), H0)
    restore_head(model, min(5, N_LAYERS-1), s)
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


# Step 0 (spectral only, fast, builds step-0 lottery control)
for s in CHECKPOINTS_SPEC_ONLY:
    sweep_one_checkpoint(s, ablate=False)

# Ablation checkpoints ordered by primary-verdict priority
CHECKPOINT_ORDER = [1000, 37000, 4000, 8000, 2000, 16000]
for s in CHECKPOINT_ORDER:
    sweep_one_checkpoint(s, ablate=True)"""))

cells.append(md("""## 10. Analysis — primary + methodology null + secondaries

Runs after any sweep completion (even partial). Primary test only requires step 1000
data, so verdict computable even if run truncates."""))
cells.append(code(r"""abl = pd.read_csv(CSV_ABL) if os.path.exists(CSV_ABL) else pd.DataFrame()
spec = pd.read_csv(CSV_SPEC) if os.path.exists(CSV_SPEC) else pd.DataFrame()
log(f'abl rows: {len(abl)}, spec rows: {len(spec)}')

merged = spec.merge(abl, on=['checkpoint', 'layer_idx', 'head_idx'], how='outer')
merged['abs_delta'] = merged['delta'].abs()

def spearman_ci(x, y, n_boot=10000, seed=20260424):
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

d1k = merged[merged['checkpoint'] == 1000].copy()

# -------- METHODOLOGY NULL GATE --------
log('Methodology null: shuffle |Δ| among heads, n=200')
if len(d1k) >= 20 and 'OV_PR' in d1k.columns and 'abs_delta' in d1k.columns:
    rng = np.random.default_rng(20260424)
    null_rhos = np.empty(200)
    d_valid = d1k.dropna(subset=['OV_PR', 'abs_delta'])
    for i in range(200):
        shuf = rng.permutation(d_valid['abs_delta'].values)
        null_rhos[i] = sp.spearmanr(d_valid['OV_PR'].values, shuf).statistic
    null_mean = float(null_rhos.mean())
    null_std = float(null_rhos.std())
    null_ci = (float(np.percentile(null_rhos, 2.5)), float(np.percentile(null_rhos, 97.5)))
    null_span = max(abs(null_ci[0]), abs(null_ci[1]))
    if null_span <= 0.10:
        gate = 'PASS'
        th = {'strong': 0.30, 'partial': 0.20, 'weak': 0.10}
    elif null_span <= 0.15:
        gate = 'CAUTION'
        th = {'strong': 0.35, 'partial': 0.25, 'weak': 0.15}
    else:
        gate = 'FAIL'
        th = None
    log(f'null mean={null_mean:+.4f}  CI [{null_ci[0]:+.4f}, {null_ci[1]:+.4f}]  span={null_span:.4f}  gate={gate}')
else:
    log('insufficient data for null gate')
    gate = 'NO_DATA'; th = None; null_mean = None; null_std = None; null_ci = (None, None)

# -------- PRIMARY --------
r_primary = spearman_ci(d1k['OV_PR'].values if 'OV_PR' in d1k.columns else np.array([]),
                         d1k['abs_delta'].values if 'abs_delta' in d1k.columns else np.array([]),
                         n_boot=2000)
print(f'\\n=== PRIMARY (OV_PR @ step1000 vs |Δ| @ step1000) ===')
if r_primary is None:
    print('  NO_DATA')
    verdict = 'NO_DATA'
else:
    print(f'  ρ = {r_primary[\"rho\"]:+.4f}  CI [{r_primary[\"ci_lo\"]:+.4f}, {r_primary[\"ci_hi\"]:+.4f}]  '
          f'p = {r_primary[\"p\"]:.2e}  n = {r_primary[\"n\"]}')

    # Decision per pre-reg rules
    if gate == 'FAIL':
        verdict = 'HOLD (methodology gate failed)'
    elif gate == 'NO_DATA':
        verdict = 'NO_DATA'
    else:
        rho, p = r_primary['rho'], r_primary['p']
        mag = abs(rho)
        if rho >= 0 and mag >= 0.10:
            verdict = 'FAIL_WRONG_DIRECTION'
        elif mag >= th['strong'] and rho < 0 and p < 0.01:
            verdict = 'PASS'
        elif mag >= th['partial'] and rho < 0:
            verdict = 'PARTIAL'
        elif mag >= th['weak'] and rho < 0:
            verdict = 'WEAK'
        else:
            verdict = 'NULL'
    print(f'  >>> {verdict} <<<')

# -------- SECONDARY: trajectory --------
s3_trajectory = {}
for ck in [1000, 2000, 4000, 8000, 16000, 37000]:
    d = merged[merged['checkpoint'] == ck]
    if 'QK_PR' not in d.columns or len(d.dropna(subset=['QK_PR','abs_delta'])) < 10: continue
    r = spearman_ci(d['QK_PR'].values, d['abs_delta'].values, n_boot=500)
    if r is None: continue
    s3_trajectory[ck] = r
    print(f'  step {ck:>6}: ρ(QK_PR, |Δ|) = {r[\"rho\"]:+.3f}  p={r[\"p\"]:.3f}')"""))

cells.append(md("""## 11. Save verdict JSON + force browser download"""))
cells.append(code(r"""verdict_data = {
    'pre_registration_commit': PRE_REG_COMMIT,
    'pre_registration_file': 'invariants_preregistration_v3_olmo.md',
    'model': MODEL_NAME,
    'architecture': {
        'class': MODEL_CLASS,
        'n_layers': int(N_LAYERS),
        'n_heads': int(N_HEADS),
        'hidden_size': int(HIDDEN),
        'd_head': int(D_HEAD),
        'head_sampling': HEAD_SAMPLING,
        'n_heads_sampled': len(HEAD_LIST),
    },
    'precision': PRECISION,
    'checkpoint_revisions': CHECKPOINT_REVISIONS,
    'primary_result': r_primary,
    'primary_verdict': verdict,
    'methodology_null': {
        'mean': null_mean, 'std': null_std,
        'ci_95': null_ci if null_ci[0] is not None else None,
        'gate': gate,
        'adjusted_thresholds': th,
    } if gate != 'NO_DATA' else {'gate': 'NO_DATA'},
    'secondary_trajectory_QK_PR': s3_trajectory,
}

with open(JSON_VERDICT, 'w') as f:
    json.dump(verdict_data, f, indent=2, default=str)
log(f'verdict saved to {JSON_VERDICT}')

try:
    from google.colab import files
    for p in [CSV_ABL, CSV_SPEC, JSON_VERDICT]:
        if os.path.exists(p):
            files.download(p)
except Exception as e:
    print(f'Download trigger failed: {e}')
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
