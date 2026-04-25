"""Build tier2_tinyllama_validation.ipynb for OLMo-style cross-team test on TinyLlama-1.1B.
Pre-registration v6 (invariants_preregistration_v6_tinyllama.md).
"""
import json, os

HERE = os.path.dirname(__file__)
GITHUB_RAW = 'https://raw.githubusercontent.com/mool32/functional-differentiation-dfe/main'
PRE_REG_COMMIT_PLACEHOLDER = '7523931'  # patched after first commit
OUT = os.path.join(HERE, 'tier2_tinyllama_validation.ipynb')


def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': s}
def code(s): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s}


cells = []

cells.append(md("""# Tier 2 — Cross-team replication on TinyLlama-1.1B

**Pre-registration v6** (commit hash recorded in verdict JSON).

Tests whether the ML early-window invariant — established on Pythia 160M/410M/1.4B
(EleutherAI, GPT-NeoX, Pile) and OLMo-2 1B (AllenAI, OLMo-2, Dolma) — replicates
on a third independent team's model:

`TinyLlama/tinyLlama-intermediate-checkpoints` — Singapore U / TinyLlama Project,
Llama architecture (RMSNorm, SwiGLU, RoPE), SlimPajama + StarCoder training data.

**Rules 1–6 binding:**
1. Direction NEGATIVE absolute. Positive = FAIL.
2. Single primary test, single decision.
3. Numeric thresholds locked.
4. Null is legitimate.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

**Primary test:** ρ(OV_PR @ step-10k, |Δ| @ step-10k). Predicted NEGATIVE.

Reference (locked from prior tests):
- Pythia 160M: ρ = −0.42
- Pythia 410M: ρ = −0.55
- Pythia 1.4B: ρ = −0.48
- OLMo-2 1B: ρ = −0.49

**Window for replication: ρ ∈ [−0.55, −0.42]** if invariant universal.

## Required hardware

A100 (40 or 80 GB). 1.1B params fits comfortably in float32.

## Estimated runtime

~3 hours full sweep. CSV append-resume; safe across Colab disconnects.
Checkpoint order: step-10k first (primary computable), then step-1431k (final),
then trajectory."""))

cells.append(md("""## 1. Install + setup"""))
cells.append(code("""!pip install -q transformers datasets torch accelerate scipy pandas huggingface_hub"""))
cells.append(code(r"""import torch, json, os, time, csv, hashlib, gc
import numpy as np
import pandas as pd
from scipy import stats as sp
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

# PyTorch 2.6+ defaults weights_only=True which breaks legacy pickle
# checkpoints (pytorch_model.bin format). TinyLlama uses this format.
# Patch defaults back to weights_only=False (idempotent — safe to re-run).
if not getattr(torch.load, '__patched_legacy_pickle__', False):
    _orig_torch_load = torch.load
    def _patched_torch_load(*a, **kw):
        kw['weights_only'] = False
        return _orig_torch_load(*a, **kw)
    _patched_torch_load.__patched_legacy_pickle__ = True
    torch.load = _patched_torch_load

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU: {gpu.name}  |  Memory: {gpu.total_memory/1e9:.1f} GB')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    OUT_DIR = '/content/drive/MyDrive/DFE_research/tier2_tinyllama'
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, '.w'), 'w') as f: f.write('ok')
    os.remove(os.path.join(OUT_DIR, '.w'))
    print(f'Drive mounted; output to {OUT_DIR}')
except Exception as e:
    raise RuntimeError(f'Drive mount required: {e}')

PRE_REG_COMMIT = '__PRE_REG_COMMIT__'

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)"""))

cells.append(md("""## 2. Config (locked per pre-reg v6)"""))
cells.append(code(r"""MODEL_NAME = 'TinyLlama/tinyLlama-intermediate-checkpoints'

PRECISION = 'float32'
N_EVAL_BATCHES = 25
EVAL_BATCH_SIZE = 4
EVAL_SEQ_LEN = 2048

# Pre-registered checkpoint mapping (fraction-of-training equivalent to Pythia)
# Pythia: 1000 / 143000 = 0.7%, etc.
# TinyLlama: 1431k total. Equivalent fractions = step-10k / 20k / 40k / 80k / 160k / 640k / 1431k.
CHECKPOINTS_K = [10, 20, 40, 80, 160, 640, 1431]   # in thousands of steps

# Heads sampling per pre-reg: 8 heads per layer at indices [0, 4, 8, 12, 16, 20, 24, 28]
SAMPLED_HEAD_IDX = [0, 4, 8, 12, 16, 20, 24, 28]

CSV_ABL = os.path.join(OUT_DIR, 'tier2_tinyllama_ablations.csv')
CSV_SPEC = os.path.join(OUT_DIR, 'tier2_tinyllama_spectral.csv')
JSON_VERDICT = os.path.join(OUT_DIR, 'tier2_tinyllama_verdict.json')

ABL_FIELDS = ['checkpoint_k', 'revision', 'layer_idx', 'head_idx',
              'baseline_loss', 'perturbed_loss', 'delta', 'elapsed_sec']
SPEC_FIELDS = ['checkpoint_k', 'revision', 'layer_idx', 'head_idx',
               'OV_PR', 'QK_PR', 'OV_entropy', 'QK_entropy',
               'sigma_OV_max', 'sigma_QK_max']

print('Config locked.')
print(f'  Model: {MODEL_NAME}')
print(f'  Pre-reg commit: {PRE_REG_COMMIT}')
print(f'  Checkpoints (k steps): {CHECKPOINTS_K}')
print(f'  Sampled head indices per layer: {SAMPLED_HEAD_IDX}')"""))

cells.append(md("""## 3. Resolve revision names"""))
cells.append(code(r"""log('Listing TinyLlama revisions...')
refs = list_repo_refs(MODEL_NAME)
all_branches = sorted(b.name for b in refs.branches)
log(f'  Total revisions: {len(all_branches)}')

import re
def resolve(ck_k):
    # Find revision matching step-{ck_k}k. If exact missing, find closest.
    target = f'step-{ck_k}k-'
    for b in all_branches:
        if b.startswith(target):
            return b, 0
    # Closest-available substitution
    closest = None
    closest_diff = None
    for b in all_branches:
        m = re.match(r'^step-(\d+)k', b)
        if m:
            n = int(m.group(1))
            d = abs(n - ck_k)
            if closest_diff is None or d < closest_diff:
                closest_diff = d
                closest = b
    return closest, closest_diff

CHECKPOINT_REVISIONS = {}
deviations = {}
for ck in CHECKPOINTS_K:
    rev, diff = resolve(ck)
    CHECKPOINT_REVISIONS[ck] = rev
    if diff and diff > 0:
        deviations[ck] = {'requested': f'step-{ck}k', 'used': rev, 'distance_k': diff}
        log(f'  step-{ck}k -> {rev}  (deviation {diff}k)')
    else:
        log(f'  step-{ck}k -> {rev}')

if deviations:
    print(f'\\nWARNING: {len(deviations)} checkpoint substitutions logged.')"""))

cells.append(md("""## 4. Architecture auto-detect"""))
cells.append(code(r"""log('Loading reference (final) checkpoint to detect architecture...')
ref_rev = CHECKPOINT_REVISIONS[CHECKPOINTS_K[-1]]
_tmp = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=ref_rev, torch_dtype=torch.float32)
N_LAYERS = _tmp.config.num_hidden_layers
N_HEADS = _tmp.config.num_attention_heads
N_KV = getattr(_tmp.config, 'num_key_value_heads', N_HEADS)
HIDDEN = _tmp.config.hidden_size
D_HEAD = HIDDEN // N_HEADS
MODEL_CLASS = _tmp.__class__.__name__
log(f'  arch: {MODEL_CLASS}, {N_LAYERS}L × {N_HEADS}H, hidden={HIDDEN}, d_head={D_HEAD}')
log(f'  num_key_value_heads = {N_KV}  ({"GQA" if N_KV < N_HEADS else "MHA"})')

# Inspect attention naming (Llama-style: q_proj/k_proj/v_proj/o_proj)
_attn_keys = list(_tmp.model.layers[0].self_attn._modules.keys())
log(f'  attention modules: {_attn_keys}')

# Validate sampled head indices
HEAD_LIST = []
for L in range(N_LAYERS):
    for H in SAMPLED_HEAD_IDX:
        if H < N_HEADS:
            HEAD_LIST.append((L, H))
log(f'  head sample: {len(HEAD_LIST)} heads ({len(SAMPLED_HEAD_IDX)} per layer × {N_LAYERS} layers)')

del _tmp
gc.collect()
torch.cuda.empty_cache()"""))

cells.append(md("""## 5. Primitives (ablation + spectral)"""))
cells.append(code(r"""def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

def get_attn(model, L):
    return model.model.layers[L].self_attn

def ablate_head(model, L, H):
    attn = get_attn(model, L)
    w = attn.o_proj.weight
    saved = w.data.clone()
    w.data[:, H*D_HEAD:(H+1)*D_HEAD] = 0
    return saved

def restore_head(model, L, saved):
    get_attn(model, L).o_proj.weight.data.copy_(saved)

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
    attn = get_attn(model, L)
    Wq = attn.q_proj.weight.detach().cpu().float().numpy()[H*D_HEAD:(H+1)*D_HEAD, :]
    Wo = attn.o_proj.weight.detach().cpu().float().numpy()[:, H*D_HEAD:(H+1)*D_HEAD]
    # GQA handling: each KV head has dimension d_head (not HIDDEN/N_KV).
    # Q head H belongs to KV group (H * N_KV // N_HEADS).
    kv_group = H * N_KV // N_HEADS
    Wv = attn.v_proj.weight.detach().cpu().float().numpy()[kv_group*D_HEAD:(kv_group+1)*D_HEAD, :]
    Wk = attn.k_proj.weight.detach().cpu().float().numpy()[kv_group*D_HEAD:(kv_group+1)*D_HEAD, :]
    sv_OV = np.linalg.svd(Wo @ Wv, compute_uv=False)
    sv_QK = np.linalg.svd(Wq.T @ Wk, compute_uv=False)
    return {
        'OV_PR': participation_ratio(sv_OV),
        'QK_PR': participation_ratio(sv_QK),
        'OV_entropy': spectral_entropy(sv_OV),
        'QK_entropy': spectral_entropy(sv_QK),
        'sigma_OV_max': float(sv_OV[0]),
        'sigma_QK_max': float(sv_QK[0]),
    }"""))

cells.append(md("""## 6. wikitext-103 batches"""))
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

cells.append(md("""## 7. Resume helpers"""))
cells.append(code(r"""def completed_abl():
    if not os.path.exists(CSV_ABL): return set()
    df = pd.read_csv(CSV_ABL)
    return set(zip(df['checkpoint_k'], df['layer_idx'], df['head_idx']))

def completed_spec():
    if not os.path.exists(CSV_SPEC): return set()
    df = pd.read_csv(CSV_SPEC)
    return set(zip(df['checkpoint_k'], df['layer_idx'], df['head_idx']))

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

done_abl = completed_abl()
done_spec = completed_spec()
log(f'completed: {len(done_abl)} ablations, {len(done_spec)} spectral rows')"""))

cells.append(md("""## 8. Main sweep — primary first"""))
cells.append(code(r"""def get_dtype():
    return torch.float32 if PRECISION == 'float32' else torch.bfloat16

def sweep_one_checkpoint(ck_k, ablate=True):
    rev = CHECKPOINT_REVISIONS.get(ck_k)
    if rev is None:
        log(f'SKIP step-{ck_k}k (no revision)')
        return
    log(f'=== step-{ck_k}k  rev={rev}  ablate={ablate} ===')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=rev, torch_dtype=get_dtype()).to(device).eval()
    log(f'  loaded in {time.time()-t0:.0f}s')

    # Spectral
    for (L, H) in HEAD_LIST:
        if (ck_k, L, H) in done_spec: continue
        inv = head_invariants(model, L, H)
        append_spec({'checkpoint_k': ck_k, 'revision': rev, 'layer_idx': L, 'head_idx': H, **inv})
        done_spec.add((ck_k, L, H))
    log(f'  spectral done')

    if not ablate:
        del model; gc.collect(); torch.cuda.empty_cache(); return

    bl = evaluate_loss(model, batches)
    log(f'  baseline loss: {bl:.4f}')

    # Sanity
    L0, H0 = HEAD_LIST[0]
    w_ref = get_attn(model, L0).o_proj.weight
    h0 = tensor_hash(w_ref.data)
    s = ablate_head(model, L0, H0); restore_head(model, L0, s)
    h2 = tensor_hash(w_ref.data)
    assert h0 == h2, 'save/restore broke!'

    # Ablate
    for i, (L, H) in enumerate(HEAD_LIST):
        if (ck_k, L, H) in done_abl: continue
        ts = time.time()
        saved = ablate_head(model, L, H)
        pl = evaluate_loss(model, batches)
        restore_head(model, L, saved)
        delta = -(pl - bl) / abs(bl)
        append_abl({
            'checkpoint_k': ck_k, 'revision': rev, 'layer_idx': L, 'head_idx': H,
            'baseline_loss': bl, 'perturbed_loss': pl, 'delta': delta,
            'elapsed_sec': time.time() - ts,
        })
        done_abl.add((ck_k, L, H))
        if (i+1) % 20 == 0 or (i+1) == len(HEAD_LIST):
            log(f'    [{i+1}/{len(HEAD_LIST)}]  L{L}H{H}  Δ={delta:+.5f}  ({time.time()-ts:.0f}s/head)')

    log(f'  checkpoint done')
    del model; gc.collect(); torch.cuda.empty_cache()


# Order: step-10k first (primary), step-1431k second (final), then trajectory
ORDER = [10, 1431, 40, 80, 20, 160, 640]
for ck in ORDER:
    sweep_one_checkpoint(ck, ablate=True)"""))

cells.append(md("""## 9. Verdict — primary, methodology null, secondary"""))
cells.append(code(r"""abl = pd.read_csv(CSV_ABL) if os.path.exists(CSV_ABL) else pd.DataFrame()
spec = pd.read_csv(CSV_SPEC) if os.path.exists(CSV_SPEC) else pd.DataFrame()
log(f'abl rows: {len(abl)}  spec rows: {len(spec)}')

merged = spec.merge(abl, on=['checkpoint_k', 'revision', 'layer_idx', 'head_idx'], how='outer')
merged['abs_delta'] = merged['delta'].abs()

def spearman_ci(x, y, n_boot=10000, seed=20260425):
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

# PRIMARY at step-10k
d10k = merged[merged['checkpoint_k'] == 10]
r_primary = spearman_ci(d10k['OV_PR'].values, d10k['abs_delta'].values, n_boot=2000)
print('=== PRIMARY (OV_PR @ step-10k vs |Δ| @ step-10k) ===')
if r_primary:
    print(f'  ρ = {r_primary["rho"]:+.4f}  CI [{r_primary["ci_lo"]:+.4f}, {r_primary["ci_hi"]:+.4f}]  '
          f'p = {r_primary["p"]:.2e}  n = {r_primary["n"]}')

# Methodology null gate
print('\\n=== METHODOLOGY NULL GATE ===')
gate = 'NO_DATA'; th = None; null_mean = None; null_std = None; null_ci = (None, None)
if r_primary and r_primary['n'] >= 20:
    rng = np.random.default_rng(20260425)
    d_clean = d10k.dropna(subset=['OV_PR','abs_delta'])
    nulls = np.empty(200)
    for i in range(200):
        nulls[i] = sp.spearmanr(d_clean['OV_PR'].values, rng.permutation(d_clean['abs_delta'].values)).statistic
    null_mean = float(nulls.mean()); null_std = float(nulls.std())
    null_ci = (float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5)))
    null_span = max(abs(null_ci[0]), abs(null_ci[1]))
    print(f'  null mean={null_mean:+.4f}  CI [{null_ci[0]:+.4f}, {null_ci[1]:+.4f}]  span={null_span:.4f}')
    if null_span <= 0.10:
        gate = 'PASS'; th = {'strong':0.30,'partial':0.20,'weak':0.10}
    elif null_span <= 0.15:
        gate = 'CAUTION'; th = {'strong':0.35,'partial':0.25,'weak':0.15}
    else:
        gate = 'FAIL'; th = None
    print(f'  gate: {gate}')

# Decision
verdict = 'NO_DATA'
if r_primary:
    rho, p = r_primary['rho'], r_primary['p']; mag = abs(rho)
    if gate == 'FAIL':
        verdict = 'HOLD (methodology gate failed)'
    elif rho > 0 and mag >= 0.10:
        verdict = 'FAIL_WRONG_DIRECTION'
    elif th and mag >= th['strong'] and rho < 0 and p < 0.01:
        verdict = 'PASS'
    elif th and mag >= th['partial'] and rho < 0:
        verdict = 'PARTIAL'
    elif mag >= 0.10 and rho < 0:
        verdict = 'WEAK'
    else:
        verdict = 'NULL'
print(f'\\n>>> VERDICT: {verdict} <<<')

# Secondary: trajectory + QK
print('\\n=== SECONDARY (exploratory) ===')
print('Trajectory ρ(OV_PR_t, |Δ|_t):')
for ck in CHECKPOINTS_K:
    d = merged[merged['checkpoint_k'] == ck].dropna(subset=['OV_PR','abs_delta'])
    if len(d) < 10: continue
    r = sp.spearmanr(d['OV_PR'], d['abs_delta'])
    print(f'  step-{ck}k:  rho={r.statistic:+.4f}  p={r.pvalue:.2e}  n={len(d)}')

print('Trajectory ρ(QK_PR_t, |Δ|_t):')
for ck in CHECKPOINTS_K:
    d = merged[merged['checkpoint_k'] == ck].dropna(subset=['QK_PR','abs_delta'])
    if len(d) < 10: continue
    r = sp.spearmanr(d['QK_PR'], d['abs_delta'])
    print(f'  step-{ck}k:  rho_QK={r.statistic:+.4f}')"""))

cells.append(md("""## 10. Save verdict + browser download"""))
cells.append(code(r"""verdict_data = {
    'pre_registration_commit': PRE_REG_COMMIT,
    'pre_registration_file': 'invariants_preregistration_v6_tinyllama.md',
    'model': MODEL_NAME,
    'architecture': {
        'class': MODEL_CLASS, 'n_layers': int(N_LAYERS), 'n_heads': int(N_HEADS),
        'n_kv_heads': int(N_KV), 'hidden_size': int(HIDDEN), 'd_head': int(D_HEAD),
        'gqa': bool(N_KV < N_HEADS),
    },
    'precision': PRECISION,
    'checkpoint_revisions': CHECKPOINT_REVISIONS,
    'checkpoint_deviations': deviations,
    'primary_result': r_primary,
    'primary_verdict': verdict,
    'methodology_null': {
        'mean': null_mean, 'std': null_std,
        'ci_95': null_ci if null_ci[0] is not None else None,
        'gate': gate, 'adjusted_thresholds': th,
    },
}

with open(JSON_VERDICT, 'w') as f:
    json.dump(verdict_data, f, indent=2, default=str)
log(f'verdict saved: {JSON_VERDICT}')

try:
    from google.colab import files
    for p in [CSV_ABL, CSV_SPEC, JSON_VERDICT]:
        if os.path.exists(p):
            files.download(p)
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

payload = json.dumps(nb, indent=1)
payload = payload.replace('__PRE_REG_COMMIT__', PRE_REG_COMMIT_PLACEHOLDER)
with open(OUT, 'w') as f:
    f.write(payload)
print(f'wrote {OUT}')
