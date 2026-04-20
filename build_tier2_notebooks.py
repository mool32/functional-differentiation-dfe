"""Build four Colab notebooks for Tier 2 replication.

Outputs (next to this script):
  tier2_t21_scaling.ipynb         T2.1 Scaling 160M + 1.4B
  tier2_t22_crossdataset.ipynb    T2.2 Cross-dataset on Pile
  tier2_t23_inverse_meta.ipynb    T2.3 Inverse-meta heads analysis
  tier2_t24_self_14b.ipynb        T2.4 Self-modeling replication on 1.4B

Each notebook is self-contained, follows Paper 2/3 discipline:
  * Mandatory Drive mount
  * Float32 + TF32 matmul
  * SHA-256 save/restore verification
  * CSV append-resume
  * Auto-download via GitHub RAW
  * Browser force-download at end
"""

import json, os

HERE = os.path.dirname(__file__)
GITHUB_RAW = 'https://raw.githubusercontent.com/mool32/functional-differentiation-dfe/main'


def nb(cells):
    return {
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


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}


def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


def save(name, cells):
    path = os.path.join(HERE, name)
    with open(path, 'w') as f:
        json.dump(nb(cells), f, indent=1)
    print(f'wrote {path}')


# =============================================================================
# Shared preamble (Drive + TF32 + logging)
# =============================================================================

PREAMBLE_INSTALL = """!pip install -q transformers datasets torch accelerate scipy pandas"""

PREAMBLE_SETUP_TEMPLATE = """import torch, json, os, time, csv, hashlib, gc, urllib.request
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f'GPU: {torch.cuda.get_device_name()}  |  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    OUT_DIR = '/content/drive/MyDrive/DFE_research/__SUBDIR__'
    os.makedirs(OUT_DIR, exist_ok=True)
    test = os.path.join(OUT_DIR, '.w')
    with open(test, 'w') as f: f.write('ok')
    os.remove(test)
    print(f'Drive mounted and verified; output to {OUT_DIR}')
except Exception as e:
    raise RuntimeError(f'Drive mount required: {e}')

GITHUB_RAW = '""" + GITHUB_RAW + """'

def log(msg):
    print(f'[{time.strftime(\"%H:%M:%S\")}] {msg}', flush=True)"""


def preamble_setup(subdir):
    return PREAMBLE_SETUP_TEMPLATE.replace('__SUBDIR__', subdir)


# =============================================================================
# T2.1 — Scaling on Pythia 160M + 1.4B
# =============================================================================

def build_t21():
    cells = []
    cells.append(md("""# Tier 2 — T2.1: Scaling Replication on Pythia 160M + 1.4B

Replicates the Paper 2 main pilot on two additional model sizes to test whether the four central findings (emergent > born-critical, dual differentiation, DFE shape emergence, scale hierarchy) are Pythia-410M-specific or hold across sizes.

| Size | Architecture | Heads ablated | Layer abl. | Type A | Checkpoints | Cost |
|------|--------------|---------------|-----------|--------|-------------|------|
| 160M-deduped | auto-detect | all | all layers | 30 | 8 | ~40 min A100 |
| 1.4B-deduped | 24L × 16H | 144 (6/layer, [0,3,6,9,12,15]) | all 24 | 30 | 8 | ~4 h A100 80GB |

**Runtime:** ~5 h total on A100 80GB. Can run 160M only on regular A100 / T4.

**Output:** `tier2_t21_scaling_{160m,1p4b}.csv` + `tier2_t21_summary.json` with DFE fits per checkpoint, dual-differentiation test, emergence-vs-born-critical counts.

**Resume:** every ablation appended to CSV; rerunning skips completed rows."""))

    cells.append(code(PREAMBLE_INSTALL))
    cells.append(code(preamble_setup('tier2')))

    cells.append(md("""## Shared config + helpers"""))
    cells.append(code("""CHECKPOINTS = [512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]
EVAL_N_BATCHES = 25
EVAL_BATCH_SIZE = 4
EVAL_SEQ_LEN = 2048
TYPE_A_ALPHA = 1.0
N_TYPE_A = 30
SEED_BASE = 42000  # same as Paper 2

CSV_FIELDS = ['checkpoint', 'perturbation_type', 'subtype', 'layer_idx', 'head_idx',
              'seed', 'baseline_loss', 'perturbed_loss', 'delta', 'elapsed_sec']

def csv_append(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new: w.writeheader()
        w.writerow(row)

def load_completed(path):
    if not os.path.exists(path): return set()
    df = pd.read_csv(path)
    return set(zip(df['checkpoint'], df['perturbation_type'], df['subtype'],
                   df['layer_idx'].fillna(-1).astype(int),
                   df['head_idx'].fillna(-1).astype(int),
                   df['seed'].fillna(-1).astype(int)))

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]"""))

    cells.append(md("""## Ablation + restore primitives"""))
    cells.append(code("""def ablate_head(model, L, H):
    hd = model.config.hidden_size // model.config.num_attention_heads
    w = model.gpt_neox.layers[L].attention.dense.weight
    saved = w.data.clone()
    w.data[:, H*hd:(H+1)*hd] = 0
    return saved

def restore_head(model, L, saved):
    model.gpt_neox.layers[L].attention.dense.weight.data.copy_(saved)

def ablate_layer(model, L):
    block = model.gpt_neox.layers[L]
    saved = {n: p.data.clone() for n, p in block.named_parameters()}
    for _, p in block.named_parameters():
        p.data.zero_()
    return saved

def restore_block(model, L, saved):
    block = model.gpt_neox.layers[L]
    for n, p in block.named_parameters():
        p.data.copy_(saved[n])

def perturb_type_a(model, L, alpha, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    block = model.gpt_neox.layers[L]
    saved = {n: p.data.clone() for n, p in block.named_parameters()}
    for _, p in block.named_parameters():
        noise = torch.randn(p.shape, generator=g, device=device, dtype=p.dtype) * (p.data.std() * alpha)
        p.data.add_(noise)
    return saved

@torch.no_grad()
def evaluate_loss(model, batches, device):
    total = 0.0
    for i in range(batches.shape[0]):
        ids = batches[i].to(device)
        total += model(input_ids=ids, labels=ids).loss.item()
    return total / batches.shape[0]"""))

    cells.append(md("""## Prepare eval batches once (wikitext-103)"""))
    cells.append(code("""from datasets import load_dataset

def stream_batches(tok, spec, n_batches=EVAL_N_BATCHES, batch_size=EVAL_BATCH_SIZE, seq_len=EVAL_SEQ_LEN):
    ds = load_dataset(*spec[:-1], split=spec[-1], streaming=True)
    need = n_batches * batch_size * seq_len
    toks = []
    for ex in ds:
        txt = ex.get('text', ex.get('content', ''))
        if not txt or len(txt.strip()) < 50: continue
        ids = tok(txt, return_tensors='pt', truncation=False)['input_ids'].squeeze()
        if ids.dim() == 0: continue
        toks.append(ids)
        if sum(t.numel() for t in toks) >= need * 1.2: break
    merged = torch.cat(toks)[:need]
    return merged.reshape(n_batches, batch_size, seq_len)"""))

    cells.append(md("""## Main sweep loop — reusable across sizes"""))
    cells.append(code("""def run_sweep(model_name, head_sample, layer_sample, csv_path, device='cuda'):
    tok = AutoTokenizer.from_pretrained(model_name)
    log(f'Tokenizer loaded: {model_name}')

    batches_path = os.path.join(OUT_DIR, f'batches_{model_name.split(\"/\")[-1]}.pt')
    if os.path.exists(batches_path):
        batches = torch.load(batches_path, weights_only=True)
        log(f'Loaded cached batches: {batches.shape}')
    else:
        batches = stream_batches(tok, ('wikitext', 'wikitext-103-raw-v1', 'train'))
        torch.save(batches, batches_path)
        log(f'Cached batches: {batches.shape}')

    completed = load_completed(csv_path)
    log(f'CSV has {len(completed)} completed rows; will skip.')

    for step in CHECKPOINTS:
        log(f'=== CHECKPOINT step{step} ===')
        t_ckpt = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=f'step{step}', torch_dtype=torch.float32
        ).to(device).eval()

        bl = evaluate_loss(model, batches, device)
        log(f'  baseline loss: {bl:.4f}')

        # Head ablations
        for (L, H) in head_sample:
            key = (step, 'head', 'output_zeroing', L, H, -1)
            if key in completed: continue
            t0 = time.time()
            saved = ablate_head(model, L, H)
            pl = evaluate_loss(model, batches, device)
            restore_head(model, L, saved)
            csv_append(csv_path, {
                'checkpoint': step, 'perturbation_type': 'head', 'subtype': 'output_zeroing',
                'layer_idx': L, 'head_idx': H, 'seed': -1,
                'baseline_loss': bl, 'perturbed_loss': pl,
                'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
            })

        # Layer ablations
        for L in layer_sample:
            key = (step, 'layer', 'full_zero', L, -1, -1)
            if key in completed: continue
            t0 = time.time()
            saved = ablate_layer(model, L)
            pl = evaluate_loss(model, batches, device)
            restore_block(model, L, saved)
            csv_append(csv_path, {
                'checkpoint': step, 'perturbation_type': 'layer', 'subtype': 'full_zero',
                'layer_idx': L, 'head_idx': -1, 'seed': -1,
                'baseline_loss': bl, 'perturbed_loss': pl,
                'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
            })

        # Type A
        n_layers = len(model.gpt_neox.layers)
        for i in range(N_TYPE_A):
            L = i % n_layers
            seed = SEED_BASE + i
            key = (step, 'type_a', 'block_noise', L, -1, seed)
            if key in completed: continue
            t0 = time.time()
            saved = perturb_type_a(model, L, TYPE_A_ALPHA, seed, device)
            pl = evaluate_loss(model, batches, device)
            restore_block(model, L, saved)
            csv_append(csv_path, {
                'checkpoint': step, 'perturbation_type': 'type_a', 'subtype': 'block_noise',
                'layer_idx': L, 'head_idx': -1, 'seed': seed,
                'baseline_loss': bl, 'perturbed_loss': pl,
                'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
            })

        log(f'  checkpoint done in {(time.time() - t_ckpt)/60:.1f} min')
        del model; gc.collect(); torch.cuda.empty_cache()

    log('sweep complete')"""))

    cells.append(md("""## Run Pythia 160M-deduped

Auto-detects layer/head counts from config and ablates all heads + all layers."""))
    cells.append(code("""MODEL_160M = 'EleutherAI/pythia-160m-deduped'
CSV_160M = os.path.join(OUT_DIR, 'tier2_t21_scaling_160m.csv')

_tmp = AutoModelForCausalLM.from_pretrained(MODEL_160M, revision='step143000', torch_dtype=torch.float32)
N_L_160M = len(_tmp.gpt_neox.layers)
N_H_160M = _tmp.config.num_attention_heads
del _tmp; gc.collect()
log(f'160M architecture: {N_L_160M} layers x {N_H_160M} heads = {N_L_160M*N_H_160M} heads total')

head_sample_160m = [(L, H) for L in range(N_L_160M) for H in range(N_H_160M)]
layer_sample_160m = list(range(N_L_160M))
log(f'160M total ablations per checkpoint: {len(head_sample_160m) + len(layer_sample_160m) + N_TYPE_A}')

run_sweep(MODEL_160M, head_sample_160m, layer_sample_160m, CSV_160M)"""))

    cells.append(md("""## Run Pythia 1.4B-deduped (optional — needs A100 80GB)

Samples 144 heads ([0,3,6,9,12,15] × 24 layers) matching Paper 2's 410M sampling. Skip this cell on smaller GPUs."""))
    cells.append(code("""MODEL_14B = 'EleutherAI/pythia-1.4b-deduped'
CSV_14B = os.path.join(OUT_DIR, 'tier2_t21_scaling_1p4b.csv')

FIXED_HEADS_14B = [0, 3, 6, 9, 12, 15]  # 6 per layer -> 144 total
N_L_14B = 24  # Pythia 1.4B
N_H_14B = 16

head_sample_14b = [(L, H) for L in range(N_L_14B) for H in FIXED_HEADS_14B]
layer_sample_14b = list(range(N_L_14B))
log(f'1.4B total ablations per checkpoint: {len(head_sample_14b) + len(layer_sample_14b) + N_TYPE_A}')

# Guard: require A100 80GB
if device == 'cuda':
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb < 60:
        print(f'WARNING: GPU has {gb:.0f} GB; 1.4B needs ~60 GB for float32 forward. Skipping.')
    else:
        run_sweep(MODEL_14B, head_sample_14b, layer_sample_14b, CSV_14B)"""))

    cells.append(md("""## Analysis — DFE fits + dual differentiation + emergence counts"""))
    cells.append(code("""def analyze(csv_path, label):
    if not os.path.exists(csv_path):
        log(f'{label}: CSV not found, skipping')
        return None
    df = pd.read_csv(csv_path)
    log(f'{label}: {len(df)} rows, {df[\"checkpoint\"].nunique()} checkpoints')
    results = {'label': label, 'per_checkpoint': {}, 'findings': {}}

    for step in sorted(df['checkpoint'].unique()):
        d = df[df['checkpoint'] == step]
        heads = d[d['perturbation_type'] == 'head']['delta'].values
        type_a = d[d['perturbation_type'] == 'type_a']['delta'].values

        # Threshold-free: Gini of |negative deltas| + effective N
        neg = -heads[heads < -1e-4]
        if len(neg) >= 5:
            sorted_neg = np.sort(neg)
            n = len(sorted_neg)
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_neg)) / (n * sorted_neg.sum()) - (n + 1) / n
            eff_n = sorted_neg.sum() ** 2 / (sorted_neg ** 2).sum()
            count_neg = len(neg)
            # beta fit (Gamma shape)
            try:
                b, _, _ = sp_stats.gamma.fit(neg, floc=0)
            except Exception:
                b = float('nan')
        else:
            gini = eff_n = count_neg = b = float('nan')

        # Type A Student's t vs normal
        if len(type_a) >= 10:
            df_t, loc_t, sc_t = sp_stats.t.fit(type_a)
            aic_t = 2*3 - 2*np.sum(sp_stats.t.logpdf(type_a, df_t, loc_t, sc_t))
            mu, si = sp_stats.norm.fit(type_a)
            aic_n = 2*2 - 2*np.sum(sp_stats.norm.logpdf(type_a, mu, si))
            d_aic = float(aic_n - aic_t)
        else:
            df_t = d_aic = float('nan')

        results['per_checkpoint'][int(step)] = {
            'n_heads': len(heads), 'n_type_a': len(type_a),
            'heads_gini': float(gini), 'heads_eff_n': float(eff_n),
            'heads_count_neg': int(count_neg) if not np.isnan(count_neg) else -1,
            'heads_beta': float(b),
            'type_a_student_t_df': float(df_t), 'type_a_delta_aic_t_vs_n': d_aic,
        }

    # Dual differentiation test: from first to last checkpoint
    first_step = min(df['checkpoint'])
    last_step = max(df['checkpoint'])
    f = results['per_checkpoint'][int(first_step)]
    l = results['per_checkpoint'][int(last_step)]
    results['findings']['dual_differentiation'] = {
        'count_neg_first': f['heads_count_neg'], 'count_neg_last': l['heads_count_neg'],
        'eff_n_first': f['heads_eff_n'], 'eff_n_last': l['heads_eff_n'],
        'gini_first': f['heads_gini'], 'gini_last': l['heads_gini'],
        'signature_present': (l['heads_count_neg'] > f['heads_count_neg']
                              and l['heads_eff_n'] < f['heads_eff_n']
                              and abs(l['heads_gini'] - f['heads_gini']) < 0.1),
    }

    # Emergence: head critical at final but not at first
    crit_thresh = 5e-4
    crit_final = df[(df['checkpoint'] == last_step) & (df['perturbation_type'] == 'head')
                    & (df['delta'].abs() > crit_thresh)]
    crit_initial = df[(df['checkpoint'] == first_step) & (df['perturbation_type'] == 'head')
                      & (df['delta'].abs() > crit_thresh)]
    final_keys = set(zip(crit_final['layer_idx'], crit_final['head_idx']))
    initial_keys = set(zip(crit_initial['layer_idx'], crit_initial['head_idx']))
    emergent = final_keys - initial_keys
    born = final_keys & initial_keys
    results['findings']['emergence'] = {
        'n_born_critical': len(born),
        'n_emergent': len(emergent),
        'ratio': (len(emergent) / max(len(born), 1)),
    }

    return results

res_160m = analyze(os.path.join(OUT_DIR, 'tier2_t21_scaling_160m.csv'), '160M')
res_14b = analyze(os.path.join(OUT_DIR, 'tier2_t21_scaling_1p4b.csv'), '1.4B')

summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    '160m': res_160m,
    '1p4b': res_14b,
}

summary_path = os.path.join(OUT_DIR, 'tier2_t21_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print('\\n=== T2.1 SUMMARY ===')
for key in ('160m', '1p4b'):
    r = summary[key]
    if r is None: continue
    print(f'\\n[{r[\"label\"]}]')
    dd = r['findings']['dual_differentiation']
    print(f'  dual_differentiation signature present: {dd[\"signature_present\"]}')
    print(f'    count_neg: {dd[\"count_neg_first\"]} -> {dd[\"count_neg_last\"]}')
    print(f'    eff_n: {dd[\"eff_n_first\"]:.2f} -> {dd[\"eff_n_last\"]:.2f}')
    print(f'    gini: {dd[\"gini_first\"]:.3f} -> {dd[\"gini_last\"]:.3f}')
    em = r['findings']['emergence']
    print(f'  emergence: born={em[\"n_born_critical\"]}  emergent={em[\"n_emergent\"]}  ratio={em[\"ratio\"]:.2f}')

log(f'saved to {summary_path}')

try:
    from google.colab import files
    for p in [os.path.join(OUT_DIR, 'tier2_t21_scaling_160m.csv'),
              os.path.join(OUT_DIR, 'tier2_t21_scaling_1p4b.csv'),
              summary_path]:
        if os.path.exists(p):
            files.download(p)
except Exception:
    pass"""))

    save('tier2_t21_scaling.ipynb', cells)


# =============================================================================
# T2.2 — Cross-dataset (Pile) on Pythia 410M
# =============================================================================

def build_t22():
    cells = []
    cells.append(md("""# Tier 2 — T2.2: Cross-dataset Validation on Pile

Re-runs Paper 2's ablation structure (144 heads + 24 layers + 30 Type A) across all 8 checkpoints of Pythia 410M, but **evaluates on Pile validation** instead of wikitext-103.

Tests the reviewer objection: "wikitext-103 specific; what about Pile validation which is closer to the training distribution?"

Central question: does the **dual differentiation signature** (count↑, eff_N↓, Gini stable) replicate?

**Compute:** ~45 min A100 (1,584 ablations). **Cost:** ~$3.

**Output:** `tier2_t22_pile.csv` + `tier2_t22_summary.json` with per-checkpoint Gini/Eff_N/count_neg and a side-by-side comparison against Paper 2's wikitext numbers."""))

    cells.append(code(PREAMBLE_INSTALL))
    cells.append(code(preamble_setup('tier2')))

    cells.append(md("""## Config (matches Paper 2 exactly except eval dataset)"""))
    cells.append(code("""MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
N_LAYERS = 24
FIXED_HEADS = [0, 3, 6, 9, 12, 15]
CHECKPOINTS = [512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]
EVAL_N_BATCHES = 25
EVAL_BATCH_SIZE = 4
EVAL_SEQ_LEN = 2048
TYPE_A_ALPHA = 1.0
N_TYPE_A = 30
SEED_BASE = 42000

CSV_PATH = os.path.join(OUT_DIR, 'tier2_t22_pile.csv')
CSV_FIELDS = ['checkpoint', 'perturbation_type', 'subtype', 'layer_idx', 'head_idx',
              'seed', 'baseline_loss', 'perturbed_loss', 'delta', 'elapsed_sec']

def csv_append(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new: w.writeheader()
        w.writerow(row)

def load_completed():
    if not os.path.exists(CSV_PATH): return set()
    df = pd.read_csv(CSV_PATH)
    return set(zip(df['checkpoint'], df['perturbation_type'], df['subtype'],
                   df['layer_idx'].fillna(-1).astype(int),
                   df['head_idx'].fillna(-1).astype(int),
                   df['seed'].fillna(-1).astype(int)))"""))

    cells.append(md("""## Prepare Pile validation batches

Tries `monology/pile-uncopyrighted` (open mirror of Pile validation). Falls back to manually listed alternates."""))
    cells.append(code("""from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
batches_path = os.path.join(OUT_DIR, 'pile_val_batches.pt')

PILE_SPECS = [
    ('monology/pile-uncopyrighted', None, 'validation'),
    ('NeelNanda/pile-10k', None, 'train'),
    ('mit-han-lab/pile-val-backup', None, 'validation'),
]

if os.path.exists(batches_path):
    batches = torch.load(batches_path, weights_only=True)
    log(f'Cached Pile batches: {batches.shape}')
else:
    batches = None
    for spec in PILE_SPECS:
        try:
            log(f'Trying {spec[0]}...')
            ds = load_dataset(spec[0], spec[1], split=spec[2], streaming=True)
            need = EVAL_N_BATCHES * EVAL_BATCH_SIZE * EVAL_SEQ_LEN
            toks = []
            for ex in ds:
                text = ex.get('text', ex.get('content', ''))
                if not text or len(text.strip()) < 50: continue
                ids = tok(text, return_tensors='pt', truncation=False)['input_ids'].squeeze()
                if ids.dim() == 0: continue
                toks.append(ids)
                if sum(t.numel() for t in toks) >= need * 1.2: break
            if sum(t.numel() for t in toks) >= need:
                batches = torch.cat(toks)[:need].reshape(EVAL_N_BATCHES, EVAL_BATCH_SIZE, EVAL_SEQ_LEN)
                torch.save(batches, batches_path)
                log(f'OK: {batches.shape} from {spec[0]}')
                break
        except Exception as e:
            log(f'  {spec[0]} failed: {e}')
    if batches is None:
        raise RuntimeError('Could not load any Pile variant. Check HF access.')"""))

    cells.append(md("""## Ablation primitives"""))
    cells.append(code("""HEAD_DIM = None  # set after model load

def ablate_head(model, L, H):
    hd = model.config.hidden_size // model.config.num_attention_heads
    w = model.gpt_neox.layers[L].attention.dense.weight
    saved = w.data.clone()
    w.data[:, H*hd:(H+1)*hd] = 0
    return saved

def restore_head(model, L, saved):
    model.gpt_neox.layers[L].attention.dense.weight.data.copy_(saved)

def ablate_layer(model, L):
    block = model.gpt_neox.layers[L]
    saved = {n: p.data.clone() for n, p in block.named_parameters()}
    for _, p in block.named_parameters():
        p.data.zero_()
    return saved

def restore_block(model, L, saved):
    block = model.gpt_neox.layers[L]
    for n, p in block.named_parameters():
        p.data.copy_(saved[n])

def perturb_type_a(model, L, alpha, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    block = model.gpt_neox.layers[L]
    saved = {n: p.data.clone() for n, p in block.named_parameters()}
    for _, p in block.named_parameters():
        noise = torch.randn(p.shape, generator=g, device=device, dtype=p.dtype) * (p.data.std() * alpha)
        p.data.add_(noise)
    return saved

@torch.no_grad()
def evaluate_loss(model):
    total = 0.0
    for i in range(batches.shape[0]):
        ids = batches[i].to(device)
        total += model(input_ids=ids, labels=ids).loss.item()
    return total / batches.shape[0]"""))

    cells.append(md("""## Main sweep"""))
    cells.append(code("""completed = load_completed()
log(f'{len(completed)} rows already done, will skip')

for step in CHECKPOINTS:
    log(f'=== step{step} ===')
    t_ck = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=f'step{step}', torch_dtype=torch.float32
    ).to(device).eval()
    bl = evaluate_loss(model)
    log(f'  baseline loss (Pile val): {bl:.4f}')

    # 144 heads
    for L in range(N_LAYERS):
        for H in FIXED_HEADS:
            key = (step, 'head', 'output_zeroing', L, H, -1)
            if key in completed: continue
            t0 = time.time()
            saved = ablate_head(model, L, H)
            pl = evaluate_loss(model)
            restore_head(model, L, saved)
            csv_append({
                'checkpoint': step, 'perturbation_type': 'head', 'subtype': 'output_zeroing',
                'layer_idx': L, 'head_idx': H, 'seed': -1,
                'baseline_loss': bl, 'perturbed_loss': pl,
                'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
            })

    # 24 layers
    for L in range(N_LAYERS):
        key = (step, 'layer', 'full_zero', L, -1, -1)
        if key in completed: continue
        t0 = time.time()
        saved = ablate_layer(model, L)
        pl = evaluate_loss(model)
        restore_block(model, L, saved)
        csv_append({
            'checkpoint': step, 'perturbation_type': 'layer', 'subtype': 'full_zero',
            'layer_idx': L, 'head_idx': -1, 'seed': -1,
            'baseline_loss': bl, 'perturbed_loss': pl,
            'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
        })

    # 30 Type A
    for i in range(N_TYPE_A):
        L = i % N_LAYERS
        seed = SEED_BASE + i
        key = (step, 'type_a', 'block_noise', L, -1, seed)
        if key in completed: continue
        t0 = time.time()
        saved = perturb_type_a(model, L, TYPE_A_ALPHA, seed)
        pl = evaluate_loss(model)
        restore_block(model, L, saved)
        csv_append({
            'checkpoint': step, 'perturbation_type': 'type_a', 'subtype': 'block_noise',
            'layer_idx': L, 'head_idx': -1, 'seed': seed,
            'baseline_loss': bl, 'perturbed_loss': pl,
            'delta': -(pl - bl) / abs(bl), 'elapsed_sec': time.time() - t0,
        })

    log(f'  checkpoint done in {(time.time() - t_ck)/60:.1f} min')
    del model; gc.collect(); torch.cuda.empty_cache()

log('sweep complete')"""))

    cells.append(md("""## Analysis — compare Pile vs wikitext (Paper 2)"""))
    cells.append(code("""# Pull Paper 2 wikitext for side-by-side
p2_path = os.path.join(OUT_DIR, 'paper2_all_ablations.csv')
if not os.path.exists(p2_path):
    urllib.request.urlretrieve(f'{GITHUB_RAW}/data/all_ablations.csv', p2_path)
paper2 = pd.read_csv(p2_path)

pile = pd.read_csv(CSV_PATH)

def metrics(df):
    heads = df[df['perturbation_type'] == 'head']['delta'].values
    neg = -heads[heads < -1e-4]
    if len(neg) < 5:
        return {'count_neg': 0, 'eff_n': float('nan'), 'gini': float('nan')}
    s = np.sort(neg)
    n = len(s)
    gini = (2 * np.sum((np.arange(1, n+1)) * s)) / (n * s.sum()) - (n + 1) / n
    eff_n = s.sum() ** 2 / (s ** 2).sum()
    return {'count_neg': int(len(neg)), 'eff_n': float(eff_n), 'gini': float(gini)}

comparison = {}
print(f'{\"step\":>6} | {\"WT count\":>8} {\"Pile count\":>10} | {\"WT eff_n\":>8} {\"Pile eff_n\":>10} | {\"WT gini\":>8} {\"Pile gini\":>8}')
print('-' * 85)
for step in CHECKPOINTS:
    wt = metrics(paper2[paper2['checkpoint'] == step])
    pl = metrics(pile[pile['checkpoint'] == step])
    comparison[step] = {'wikitext': wt, 'pile': pl}
    print(f'{step:>6} | {wt[\"count_neg\"]:>8} {pl[\"count_neg\"]:>10} | '
          f'{wt[\"eff_n\"]:>8.2f} {pl[\"eff_n\"]:>10.2f} | '
          f'{wt[\"gini\"]:>8.3f} {pl[\"gini\"]:>8.3f}')

# Dual differentiation verdict for Pile
first, last = CHECKPOINTS[0], CHECKPOINTS[-1]
pf, pl = comparison[first]['pile'], comparison[last]['pile']
dd_signature = (pl['count_neg'] > pf['count_neg']
                and pl['eff_n'] < pf['eff_n']
                and abs(pl['gini'] - pf['gini']) < 0.1)
print(f'\\nPile dual-differentiation signature: {dd_signature}')

summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'comparison_per_checkpoint': comparison,
    'pile_dual_differentiation_signature': bool(dd_signature),
}
summary_path = os.path.join(OUT_DIR, 'tier2_t22_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
log(f'saved {summary_path}')

try:
    from google.colab import files
    files.download(CSV_PATH)
    files.download(summary_path)
except Exception:
    pass"""))

    save('tier2_t22_crossdataset.ipynb', cells)


# =============================================================================
# T2.3 — Inverse meta-heads analysis
# =============================================================================

def build_t23():
    cells = []
    cells.append(md("""# Tier 2 — T2.3: Inverse Meta-Heads Analysis (Paper 3)

Paper 3's main sweep produced **20 "inverse meta-heads"** — heads whose ablation *improves* self-consistency on disagreement questions (Δ_self < 0). They were observed but not discussed.

This notebook does three things:

1. **Characterize the 20 inverse heads** — Paper 2 class distribution, layer position, Δ_self/Δ_cross/Δ_task signature.
2. **Disagreement pattern analysis** — on which questions does ablation help?
3. **De-biasing hypothesis test** — do inverse-meta heads lock in specific answer patterns? Measure **answer variance across sampling seeds** with vs without ablation. Hypothesis: ablation decreases variance (removes a "de-biasing" component) OR increases it (removes a "locked" bias). Either direction is informative.

**Compute:** ~15 min A100. Loads existing Paper 3 data from GitHub, only the variance probe needs model forward passes.

**Output:** `tier2_t23_inverse_meta.csv` (variance probe results) + `tier2_t23_summary.json`."""))

    cells.append(code(PREAMBLE_INSTALL))
    cells.append(code(preamble_setup('tier2')))

    cells.append(md("""## Load existing Paper 3 micropilot ablation data

Looks in three places in order:
1. User's Drive: `MyDrive/DFE_research/preflight/micropilot_ablations.csv` (produced by `micropilot_ablation_sweep.ipynb`)
2. GitHub raw (if eventually uploaded there)
3. Current Colab working directory (if user uploaded manually)

If nothing found, prints clear instructions and stops."""))
    cells.append(code("""abl_path = None
CANDIDATES_ABL = [
    '/content/drive/MyDrive/DFE_research/preflight/micropilot_ablations.csv',
    '/content/drive/MyDrive/DFE_research/tier2/micropilot_ablations.csv',
    os.path.join(OUT_DIR, 'micropilot_ablations.csv'),
    '/content/micropilot_ablations.csv',
]
for p in CANDIDATES_ABL:
    if os.path.exists(p):
        abl_path = p
        log(f'Found ablation CSV at {p}')
        break

if abl_path is None:
    # Last resort: try GitHub
    try:
        remote = f'{GITHUB_RAW}/data/micropilot/micropilot_ablations.csv'
        local = os.path.join(OUT_DIR, 'micropilot_ablations.csv')
        urllib.request.urlretrieve(remote, local)
        abl_path = local
        log(f'Downloaded from GitHub to {local}')
    except Exception as e:
        raise RuntimeError(
            'micropilot_ablations.csv not found. Expected locations:\\n'
            + '\\n'.join(f'  - {p}' for p in CANDIDATES_ABL)
            + '\\n\\nFix: copy the CSV from your previous Paper 3 run into one of these paths,'
            ' then re-run this cell. The file is produced by micropilot_ablation_sweep.ipynb'
            ' and normally lives at MyDrive/DFE_research/preflight/.'
        )

df = pd.read_csv(abl_path)
log(f'Loaded {len(df)} ablations')

# Inverse meta-heads: Δ_self < threshold (ablation improves self-alignment)
# Use same convention as Paper 3: delta_self = baseline_self - ablated_self
# so POSITIVE = ablation hurt self (classic meta). NEGATIVE = ablation helped self (inverse).
df_sorted = df.sort_values('delta_self').reset_index(drop=True)
inverse = df_sorted.head(20).copy()
print('\\n=== 20 INVERSE META-HEADS (lowest Delta_self = ablation improves self-consistency) ===')
print(f'{\"L\":>3} {\"H\":>3} | {\"Δself\":>7} {\"Δcross\":>7} {\"Δtask\":>7}')
print('-' * 50)
for _, r in inverse.iterrows():
    print(f'{int(r[\"layer_idx\"]):>3} {int(r[\"head_idx\"]):>3} | '
          f'{r[\"delta_self\"]:>+7.3f} {r[\"delta_cross\"]:>+7.3f} {r[\"delta_task\"]:>+7.4f}')"""))

    cells.append(md("""## Paper 2 class distribution of inverse heads"""))
    cells.append(code("""p2_path = os.path.join(OUT_DIR, 'paper2_all_ablations.csv')
urllib.request.urlretrieve(f'{GITHUB_RAW}/data/all_ablations.csv', p2_path)
p2 = pd.read_csv(p2_path)
p2_heads = p2[p2['perturbation_type'] == 'head'].copy()

classes = {}
for (L, H), g in p2_heads.groupby(['layer_idx', 'head_idx']):
    g = g.sort_values('checkpoint')
    init = abs(g.iloc[0]['delta'])
    final = abs(g.iloc[-1]['delta'])
    if final < 5e-4:
        cls = 'never-critical'
    elif init > 5e-4 and final > 5e-4:
        cls = 'born-critical'
    elif init < 1e-4 and final > 5e-4:
        cls = 'emergent'
    else:
        cls = 'growing'
    classes[(int(L), int(H))] = cls

inverse['paper2_class'] = inverse.apply(
    lambda r: classes.get((int(r['layer_idx']), int(r['head_idx'])), 'unknown'), axis=1
)

# Class distribution + enrichment vs all 144 sampled heads
all_classes = pd.Series([classes.get((L, H), 'unknown')
                         for L in range(24) for H in [0,3,6,9,12,15]]).value_counts()
inv_dist = inverse['paper2_class'].value_counts()
print('\\n=== PAPER 2 CLASS OF INVERSE META-HEADS ===')
print(f'{\"class\":<15} {\"obs\":>4} {\"exp\":>6} {\"ratio\":>6}')
print('-' * 40)
for cls in ['born-critical', 'emergent', 'growing', 'never-critical', 'unknown']:
    obs = inv_dist.get(cls, 0)
    exp = 20 * all_classes.get(cls, 0) / all_classes.sum()
    ratio = obs / exp if exp > 0 else float('inf')
    print(f'{cls:<15} {obs:>4} {exp:>6.1f} {ratio:>6.2f}x')

# Layer distribution
print('\\n=== LAYER DISTRIBUTION ===')
layer_dist = inverse['layer_idx'].value_counts().sort_index()
for L, c in layer_dist.items():
    print(f'  L{int(L):>2}: {\"#\" * int(c)} ({int(c)})')"""))

    cells.append(md("""## De-biasing hypothesis — variance across sampling seeds

For each inverse-meta head, generate 10 completions per question with multinomial sampling at different seeds, with and without ablation. Measure variance (entropy) of first-token distribution across seeds.

Hypothesis A (de-biasing): ablation → answer variance *decreases* → head was a noise/exploration component.
Hypothesis B (bias-locking): ablation → answer variance *increases* → head was enforcing a preference.
Null: no change → head's Δ_self was a spurious small effect.

Probe design: 5 questions × 10 seeds × 20 heads × 2 (with/without abl) = 2,000 forward passes @ ~0.05s = ~2 min."""))

    cells.append(code("""MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
log(f'Loading {MODEL_NAME} step143000 (float32)...')
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision='step143000', torch_dtype=torch.float32
).to(device).eval()

# Load question bank
qb_path = os.path.join(OUT_DIR, 'question_bank.json')
urllib.request.urlretrieve(f'{GITHUB_RAW}/data/micropilot/question_bank.json', qb_path)
with open(qb_path) as f:
    qb = json.load(f)

# Subset: use 8 questions (enough for variance estimation, keeps cost low)
PROBE_QUESTIONS = qb['questions'][:8]
log(f'Probing {len(PROBE_QUESTIONS)} questions')

PHASE_A = (\"If I asked you 'What is 5 + 3?', your first word would be: Eight\\n\"
           \"If I asked you 'Who wrote Hamlet?', your first word would be: Shakespeare\\n\"
           \"If I asked you 'What color is grass?', your first word would be: Green\\n\"
           \"If I asked you '{q}', your first word would be:\")

def ablate_head(L, H):
    hd = model.config.hidden_size // model.config.num_attention_heads
    w = model.gpt_neox.layers[L].attention.dense.weight
    saved = w.data.clone()
    w.data[:, H*hd:(H+1)*hd] = 0
    return saved

def restore_head(L, saved):
    model.gpt_neox.layers[L].attention.dense.weight.data.copy_(saved)

@torch.no_grad()
def first_token_distribution_entropy(prompt, n_samples=10, temperature=1.0):
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    logits = model(**inputs).logits[0, -1, :] / temperature
    # Sample n_samples first tokens, compute entropy of their empirical distribution
    probs = torch.softmax(logits, dim=-1)
    samples = torch.multinomial(probs, n_samples, replacement=True).cpu().numpy()
    unique, counts = np.unique(samples, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))  # empirical entropy"""))

    cells.append(code("""CSV_PROBE = os.path.join(OUT_DIR, 'tier2_t23_inverse_meta.csv')
FIELDS = ['layer_idx', 'head_idx', 'q_idx', 'question', 'baseline_entropy', 'ablated_entropy', 'delta_entropy']

completed = set()
if os.path.exists(CSV_PROBE):
    _d = pd.read_csv(CSV_PROBE)
    completed = set(zip(_d['layer_idx'], _d['head_idx'], _d['q_idx']))
    log(f'{len(completed)} probe rows already done')

def append_probe(row):
    new = not os.path.exists(CSV_PROBE) or os.path.getsize(CSV_PROBE) == 0
    with open(CSV_PROBE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)

for _, row in inverse.iterrows():
    L, H = int(row['layer_idx']), int(row['head_idx'])
    for qi, q in enumerate(PROBE_QUESTIONS):
        if (L, H, qi) in completed: continue
        prompt = PHASE_A.format(q=q['q'])
        torch.manual_seed(7000 + qi)
        ent_baseline = first_token_distribution_entropy(prompt, n_samples=10)
        saved = ablate_head(L, H)
        torch.manual_seed(7000 + qi)  # same multinomial seed for fair comparison
        ent_ablated = first_token_distribution_entropy(prompt, n_samples=10)
        restore_head(L, saved)
        append_probe({
            'layer_idx': L, 'head_idx': H, 'q_idx': qi, 'question': q['q'],
            'baseline_entropy': ent_baseline, 'ablated_entropy': ent_ablated,
            'delta_entropy': ent_ablated - ent_baseline,
        })
    log(f'  probed L{L}H{H}')

probe = pd.read_csv(CSV_PROBE)
log(f'probe done, {len(probe)} rows')"""))

    cells.append(md("""## Verdict on de-biasing hypothesis"""))
    cells.append(code("""# Per-head mean Δ_entropy, aggregated across questions
per_head = probe.groupby(['layer_idx', 'head_idx'])['delta_entropy'].agg(['mean', 'std', 'count']).reset_index()
per_head['abs_mean'] = per_head['mean'].abs()

print('=== PER-HEAD ENTROPY CHANGE ===')
print(f'{\"L\":>3} {\"H\":>3} | {\"mean Δentropy\":>14} {\"std\":>7} | direction')
print('-' * 55)
for _, r in per_head.sort_values('mean').iterrows():
    direction = 'VARIANCE DOWN (de-biasing)' if r['mean'] < -0.05 else ('VARIANCE UP (bias-locked)' if r['mean'] > 0.05 else 'null')
    print(f'{int(r[\"layer_idx\"]):>3} {int(r[\"head_idx\"]):>3} | {r[\"mean\"]:>+14.4f} {r[\"std\"]:>+7.4f} | {direction}')

mean_delta = probe['delta_entropy'].mean()
n_de = (per_head['mean'] < -0.05).sum()
n_lock = (per_head['mean'] > 0.05).sum()
n_null = 20 - n_de - n_lock

# One-sample t-test on mean delta across all probes
t_stat, p_val = sp_stats.ttest_1samp(probe['delta_entropy'].values, 0)

print(f'\\nGrand mean Δentropy: {mean_delta:+.4f}')
print(f'One-sample t-test H0=0: t={t_stat:.2f}, p={p_val:.4f}')
print(f'De-biasing heads: {n_de}/20   Bias-locking: {n_lock}/20   Null: {n_null}/20')

if n_de >= 12 and p_val < 0.05 and mean_delta < 0:
    verdict = 'DE-BIASING HYPOTHESIS SUPPORTED'
elif n_lock >= 12 and p_val < 0.05 and mean_delta > 0:
    verdict = 'BIAS-LOCKING HYPOTHESIS SUPPORTED'
else:
    verdict = 'NEITHER CLEAN — report as open question'

print(f'\\n>>> T2.3 VERDICT: {verdict} <<<')

summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_inverse_heads': 20,
    'paper2_class_distribution': inv_dist.to_dict(),
    'layer_distribution': layer_dist.to_dict(),
    'grand_mean_delta_entropy': float(mean_delta),
    't_stat': float(t_stat), 'p_value': float(p_val),
    'n_de_biasing': int(n_de), 'n_bias_locking': int(n_lock), 'n_null': int(n_null),
    'verdict': verdict,
    'inverse_heads': [
        {'layer': int(r['layer_idx']), 'head': int(r['head_idx']),
         'delta_self': float(r['delta_self']), 'delta_cross': float(r['delta_cross']),
         'delta_task': float(r['delta_task']), 'paper2_class': r['paper2_class']}
        for _, r in inverse.iterrows()
    ],
}
summary_path = os.path.join(OUT_DIR, 'tier2_t23_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
log(f'saved {summary_path}')

try:
    from google.colab import files
    files.download(CSV_PROBE)
    files.download(summary_path)
except Exception:
    pass"""))

    save('tier2_t23_inverse_meta.ipynb', cells)


# =============================================================================
# T2.4 — Self-modeling replication on Pythia 1.4B
# =============================================================================

def build_t24():
    cells = []
    cells.append(md("""# Tier 2 — T2.4: Self-Modeling Replication on Pythia 1.4B

Replicates the Paper 3 pipeline — Phase A/B self-modeling prompt, decisive control, ablation sweep — on Pythia 1.4B. Uses **Pythia 6.9B as the "other model"** for question-bank disagreement.

**Three stages, each gated by the previous:**

1. **Question bank for 1.4B vs 6.9B.** Find N≥20 questions where 1.4B gives a robust first-token answer AND 6.9B disagrees. If fewer than 20 disagreements satisfy the robustness filter, the experiment is under-powered; abort with a negative-result summary.
2. **Decisive control.** On the bank, measure self-match rate vs cross-match rate when asked "What would YOU say?" vs "What would Pythia-6.9B say?". Pass if p<0.05 by binomial test.
3. **Ablation sweep.** 144 heads (6 per layer × 24 layers, same as 410M): Δ_self, Δ_cross, Δ_task.

**Compute:** ~3 h A100 80GB ($15). **Requires A100 80GB or similar for float32 forward on 6.9B.** On smaller GPU, disable Stage 1's 6.9B call and skip the decisive control (use bf16 fallback).

**Output:** `tier2_t24_question_bank_14b.json`, `tier2_t24_decisive.json`, `tier2_t24_ablations.csv`, `tier2_t24_summary.json`."""))

    cells.append(code(PREAMBLE_INSTALL))
    cells.append(code(preamble_setup('tier2')))

    cells.append(md("""## Helpers"""))
    cells.append(code("""MODEL_14B = 'EleutherAI/pythia-1.4b-deduped'
MODEL_69B = 'EleutherAI/pythia-6.9b-deduped'
N_LAYERS_14B = 24
N_HEADS_14B = 16
FIXED_HEADS_14B = [0, 3, 6, 9, 12, 15]

PHASE_A = (\"If I asked you 'What is 5 + 3?', your first word would be: Eight\\n\"
           \"If I asked you 'Who wrote Hamlet?', your first word would be: Shakespeare\\n\"
           \"If I asked you 'What color is grass?', your first word would be: Green\\n\"
           \"If I asked you '{q}', your first word would be:\")

PHASE_B = (\"What is the first word of your answer to '{q}'? The first word is:\")

def check_gpu_memory_ok(min_gb):
    if device == 'cuda':
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return gb >= min_gb, gb
    return False, 0

@torch.no_grad()
def top_token(model, tok, prompt):
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    return tok.decode([model(**inp).logits[0, -1, :].argmax().item()]).strip().lower()

def free_model(m):
    del m; gc.collect(); torch.cuda.empty_cache()"""))

    cells.append(md("""## Stage 1 — Build question bank (1.4B vs 6.9B)

Streams candidate questions, filters by:
- 1.4B robust preference: `p_own > 2 × p_other` (same as 410M bank)
- First-token disagreement with 6.9B

Accept first 30 that pass both filters.

Uses a curated seed list of 60 candidate questions across domains. If the Stage 1 yield is <20, produces a negative-result note and stops."""))

    cells.append(code("""CANDIDATES = [
    (\"What is the main ingredient in bread?\", \"common_sense\"),
    (\"What do birds use to fly?\", \"common_sense\"),
    (\"What do fish breathe underwater?\", \"common_sense\"),
    (\"What do plants need to grow besides water?\", \"common_sense\"),
    (\"What do humans wear on their feet?\", \"common_sense\"),
    (\"What is 7 + 5?\", \"math\"),
    (\"What is 9 times 6?\", \"math\"),
    (\"What is 100 divided by 4?\", \"math\"),
    (\"What is 15 minus 8?\", \"math\"),
    (\"What is the square root of 81?\", \"math\"),
    (\"What is the capital of France?\", \"geography\"),
    (\"What is the capital of Japan?\", \"geography\"),
    (\"What is the largest ocean?\", \"geography\"),
    (\"What is the longest river in the world?\", \"geography\"),
    (\"What country is known for the Eiffel Tower?\", \"geography\"),
    (\"Who wrote Romeo and Juliet?\", \"literature\"),
    (\"Who wrote 1984?\", \"literature\"),
    (\"Who wrote The Great Gatsby?\", \"literature\"),
    (\"Who wrote Don Quixote?\", \"literature\"),
    (\"Who wrote Pride and Prejudice?\", \"literature\"),
    (\"Who was the first president of the United States?\", \"history\"),
    (\"In what year did World War II end?\", \"history\"),
    (\"Who painted the Mona Lisa?\", \"history\"),
    (\"Who invented the light bulb?\", \"history\"),
    (\"In what city was the Declaration of Independence signed?\", \"history\"),
    (\"What is the chemical symbol for gold?\", \"science\"),
    (\"What planet is known as the Red Planet?\", \"science\"),
    (\"What is the hardest natural substance?\", \"science\"),
    (\"What gas do plants absorb from the atmosphere?\", \"science\"),
    (\"What is the speed of light in a vacuum?\", \"science\"),
    (\"What is the largest planet in our solar system?\", \"science\"),
    (\"What element does 'O' represent in chemistry?\", \"science\"),
    (\"What is the boiling point of water in Celsius?\", \"science\"),
    (\"What is the study of living organisms called?\", \"science\"),
    (\"What do bees produce?\", \"common_sense\"),
    (\"What animal is known as the king of the jungle?\", \"common_sense\"),
    (\"What is the color of the sun at noon?\", \"common_sense\"),
    (\"What is the square of 12?\", \"math\"),
    (\"What is 2 to the power of 10?\", \"math\"),
    (\"What is the capital of Australia?\", \"geography\"),
    (\"What continent is Egypt in?\", \"geography\"),
    (\"Who wrote The Odyssey?\", \"literature\"),
    (\"Who wrote Crime and Punishment?\", \"literature\"),
    (\"Who discovered penicillin?\", \"history\"),
    (\"In what year did the Berlin Wall fall?\", \"history\"),
    (\"What metal is liquid at room temperature?\", \"science\"),
    (\"What instrument measures temperature?\", \"science\"),
    (\"What force pulls objects toward Earth?\", \"science\"),
    (\"What is H2O commonly known as?\", \"science\"),
    (\"What language is most spoken in Brazil?\", \"geography\"),
    (\"What currency is used in Japan?\", \"geography\"),
    (\"Who wrote The Divine Comedy?\", \"literature\"),
    (\"Who wrote Moby Dick?\", \"literature\"),
    (\"What is the main material of a glass bottle?\", \"common_sense\"),
    (\"What is the largest mammal?\", \"common_sense\"),
    (\"What is the smallest prime number?\", \"math\"),
    (\"What is 50 percent of 200?\", \"math\"),
    (\"What is the second planet from the sun?\", \"science\"),
    (\"What is the name of Earth's natural satellite?\", \"science\"),
    (\"Who was the first person to walk on the Moon?\", \"history\"),
    (\"What is the study of earthquakes called?\", \"science\"),
]
log(f'{len(CANDIDATES)} candidate questions prepared')"""))

    cells.append(code("""QB_PATH = os.path.join(OUT_DIR, 'tier2_t24_question_bank_14b.json')
log('Loading Pythia 1.4B step143000 (float32)...')
tok = AutoTokenizer.from_pretrained(MODEL_14B)
m14 = AutoModelForCausalLM.from_pretrained(
    MODEL_14B, revision='step143000', torch_dtype=torch.float32
).to(device).eval()

@torch.no_grad()
def probs_first_token(model, prompt, top_k_compare=None):
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    logits = model(**inp).logits[0, -1, :]
    p = torch.softmax(logits.float(), dim=-1)
    argmax = int(p.argmax().item())
    word = tok.decode([argmax]).strip().lower()
    return {'argmax_id': argmax, 'word': word, 'p_own': float(p[argmax].item())}

candidates_records = []
for q, domain in CANDIDATES:
    prompt = PHASE_A.format(q=q)
    r14 = probs_first_token(m14, prompt)
    candidates_records.append({'q': q, 'domain': domain, 'answer_14b': r14['word'],
                              'argmax_14b': r14['argmax_id'], 'p14_own': r14['p_own']})
log(f'14B first-token answers computed for {len(candidates_records)} candidates')
free_model(m14)"""))

    cells.append(code("""# Load 6.9B — try float32 first, fall back to bfloat16 if GPU too small
ok, gb = check_gpu_memory_ok(60)
if ok:
    log(f'Loading 6.9B float32 on {gb:.0f} GB GPU...')
    m69 = AutoModelForCausalLM.from_pretrained(
        MODEL_69B, revision='step143000', torch_dtype=torch.float32
    ).to(device).eval()
    dtype_used = 'float32'
else:
    log(f'Only {gb:.0f} GB available; loading 6.9B in bfloat16')
    m69 = AutoModelForCausalLM.from_pretrained(
        MODEL_69B, revision='step143000', torch_dtype=torch.bfloat16
    ).to(device).eval()
    dtype_used = 'bfloat16'

for rec in candidates_records:
    prompt = PHASE_A.format(q=rec['q'])
    r69 = probs_first_token(m69, prompt)
    rec['answer_69b'] = r69['word']
    rec['argmax_69b'] = r69['argmax_id']
    rec['p69_own'] = r69['p_own']

free_model(m69)
log(f'6.9B done ({dtype_used}); filtering...')

# Reload 1.4B to measure p_other (prob of 6.9B's answer) for robustness filter
m14 = AutoModelForCausalLM.from_pretrained(
    MODEL_14B, revision='step143000', torch_dtype=torch.float32
).to(device).eval()

def p_of_token(model, prompt, token_id):
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    logits = model(**inp).logits[0, -1, :]
    p = torch.softmax(logits.float(), dim=-1)
    return float(p[token_id].item())

kept = []
for rec in candidates_records:
    if rec['argmax_14b'] == rec['argmax_69b']: continue  # not a disagreement
    prompt = PHASE_A.format(q=rec['q'])
    rec['p14_other'] = p_of_token(m14, prompt, rec['argmax_69b'])
    rec['strength_14b'] = rec['p14_own'] / max(rec['p14_other'], 1e-9)
    if rec['strength_14b'] > 2.0:  # robust
        kept.append(rec)

log(f'{len(kept)} robust disagreement questions (strength>2)')
qb = {
    'n_questions': len(kept),
    'filter_criteria': {'rule': 'p_own>2*p_other AND 14B != 6.9B on first token'},
    'models': [MODEL_14B + ' step143000', MODEL_69B + ' step143000'],
    'dtype_used_69b': dtype_used,
    'questions': kept,
}
with open(QB_PATH, 'w') as f:
    json.dump(qb, f, indent=2)
log(f'question bank saved: {QB_PATH}')

if len(kept) < 20:
    print(f'\\n>>> STAGE 1 INSUFFICIENT: {len(kept)} < 20 robust disagreements <<<')
    print('1.4B-vs-6.9B disagreement surface too narrow for meaningful ablation signal.')
    print('Options: (a) expand candidate list, (b) use 2.8B as \"other\", (c) report as negative result.')
    STAGE1_PASS = False
else:
    print(f'\\n>>> STAGE 1 PASS: {len(kept)} questions <<<')
    STAGE1_PASS = True"""))

    cells.append(md("""## Stage 2 — Decisive control

Measures self-match vs cross-match rate on the bank. Pass if self-match > cross-match by at least 5pp (binomial p<0.05).

Gate: only run if Stage 1 passed."""))

    cells.append(code("""if STAGE1_PASS:
    # m14 already loaded; count self-match vs cross-match
    n_self, n_cross = 0, 0
    for rec in kept:
        pred = top_token(m14, tok, PHASE_A.format(q=rec['q']))
        if pred == rec['answer_14b']:
            n_self += 1
        if pred == rec['answer_69b']:
            n_cross += 1
    self_rate = n_self / len(kept)
    cross_rate = n_cross / len(kept)
    try:
        res = sp_stats.binomtest(n_self, len(kept), p=cross_rate if cross_rate > 0 else 0.1, alternative='greater')
        p_val = float(res.pvalue)
    except Exception:
        p_val = float('nan')

    stage2 = {
        'n_questions': len(kept),
        'n_self_match': n_self,
        'n_cross_match': n_cross,
        'self_rate': float(self_rate),
        'cross_rate': float(cross_rate),
        'p_value_self_vs_cross': p_val,
        'pass': bool(self_rate > cross_rate + 0.05 and p_val < 0.05),
    }
    with open(os.path.join(OUT_DIR, 'tier2_t24_decisive.json'), 'w') as f:
        json.dump(stage2, f, indent=2)
    log(f'decisive: self={self_rate:.3f}  cross={cross_rate:.3f}  p={p_val:.4f}')
    STAGE2_PASS = stage2['pass']
    print(f'>>> STAGE 2: {\"PASS\" if STAGE2_PASS else \"FAIL\"} <<<')
else:
    STAGE2_PASS = False
    stage2 = None"""))

    cells.append(md("""## Stage 3 — Ablation sweep on 144 heads

Same protocol as Paper 3 micropilot: Δ_self, Δ_cross, Δ_task per head."""))

    cells.append(code("""if STAGE2_PASS:
    # Baseline (without ablation)
    log('Computing baselines...')
    baseline_self = n_self / len(kept)
    baseline_cross = n_cross / len(kept)

    # Task baseline: wikitext loss on 1.4B
    from datasets import load_dataset
    batches_path = os.path.join(OUT_DIR, 'batches_1p4b.pt')
    if os.path.exists(batches_path):
        batches = torch.load(batches_path, weights_only=True)
    else:
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
        need = 25 * 4 * 2048
        toks = []
        for ex in ds:
            txt = ex.get('text', '')
            if len(txt.strip()) < 50: continue
            ids = tok(txt, return_tensors='pt', truncation=False)['input_ids'].squeeze()
            if ids.dim() == 0: continue
            toks.append(ids)
            if sum(t.numel() for t in toks) >= need * 1.2: break
        batches = torch.cat(toks)[:need].reshape(25, 4, 2048)
        torch.save(batches, batches_path)
    log(f'batches {batches.shape}')

    @torch.no_grad()
    def task_loss():
        tot = 0.0
        for i in range(batches.shape[0]):
            ids = batches[i].to(device)
            tot += m14(input_ids=ids, labels=ids).loss.item()
        return tot / batches.shape[0]

    baseline_task = task_loss()
    log(f'baseline: self={baseline_self:.3f}  cross={baseline_cross:.3f}  task={baseline_task:.4f}')

    # Verify save/restore bitwise
    HD = m14.config.hidden_size // m14.config.num_attention_heads
    def ablate_head(L, H):
        w = m14.gpt_neox.layers[L].attention.dense.weight
        s = w.data.clone()
        w.data[:, H*HD:(H+1)*HD] = 0
        return s
    def restore_head(L, s):
        m14.gpt_neox.layers[L].attention.dense.weight.data.copy_(s)

    def alignment_rates():
        ns, nc = 0, 0
        for rec in kept:
            pred = top_token(m14, tok, PHASE_A.format(q=rec['q']))
            if pred == rec['answer_14b']: ns += 1
            if pred == rec['answer_69b']: nc += 1
        return ns / len(kept), nc / len(kept)

    # Sanity
    w_ref = m14.gpt_neox.layers[5].attention.dense.weight
    h0 = hashlib.sha256(w_ref.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    s = ablate_head(5, 7); restore_head(5, s)
    h2 = hashlib.sha256(w_ref.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    assert h0 == h2, 'save/restore broken'
    log('save/restore bitwise verified')

    CSV_ABL = os.path.join(OUT_DIR, 'tier2_t24_ablations.csv')
    FIELDS = ['layer_idx', 'head_idx', 'ablated_self', 'ablated_cross', 'ablated_task',
              'delta_self', 'delta_cross', 'delta_task', 'elapsed_sec']

    completed = set()
    if os.path.exists(CSV_ABL):
        _d = pd.read_csv(CSV_ABL)
        completed = set(zip(_d['layer_idx'], _d['head_idx']))
    log(f'{len(completed)} heads done, {144 - len(completed)} remaining')

    for L in range(N_LAYERS_14B):
        for H in FIXED_HEADS_14B:
            if (L, H) in completed: continue
            t0 = time.time()
            saved = ablate_head(L, H)
            a_self, a_cross = alignment_rates()
            a_task = task_loss()
            restore_head(L, saved)
            row = {
                'layer_idx': L, 'head_idx': H,
                'ablated_self': a_self, 'ablated_cross': a_cross, 'ablated_task': a_task,
                'delta_self': baseline_self - a_self,
                'delta_cross': baseline_cross - a_cross,
                'delta_task': a_task - baseline_task,
                'elapsed_sec': time.time() - t0,
            }
            new = not os.path.exists(CSV_ABL) or os.path.getsize(CSV_ABL) == 0
            with open(CSV_ABL, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new: w.writeheader()
                w.writerow(row)

    abl_df = pd.read_csv(CSV_ABL)
    abl_df_sorted = abl_df.sort_values('delta_self', ascending=False).reset_index(drop=True)
    print('\\n=== TOP 15 META-HEAD CANDIDATES (Pythia 1.4B) ===')
    print(f'{\"L\":>3} {\"H\":>3} | {\"Δself\":>7} {\"Δcross\":>7} {\"Δtask\":>8}')
    print('-' * 50)
    for _, r in abl_df_sorted.head(15).iterrows():
        print(f'{int(r[\"layer_idx\"]):>3} {int(r[\"head_idx\"]):>3} | '
              f'{r[\"delta_self\"]:>+7.3f} {r[\"delta_cross\"]:>+7.3f} {r[\"delta_task\"]:>+8.5f}')

    # Gate: ≥5 heads with Δ_self>0.05 AND Δ_task<0.02, + top-10 signature
    meta = abl_df[(abl_df['delta_self'] > 0.05) & (abl_df['delta_task'] < 0.02)]
    top10_self_gt_cross = sum(1 for _, r in abl_df_sorted.head(10).iterrows()
                              if r['delta_self'] > r['delta_cross'])
    print(f'\\nMeta-heads (Δ_self>0.05, Δ_task<0.02): {len(meta)}')
    print(f'Top-10 with Δ_self>Δ_cross: {top10_self_gt_cross}/10')
    replicates = len(meta) >= 5 and top10_self_gt_cross >= 7
    print(f'\\n>>> 1.4B SELF-MODELING REPLICATION: {\"PASS\" if replicates else \"FAIL\"} <<<')
else:
    replicates = False
    abl_df = None"""))

    cells.append(md("""## Final summary"""))
    cells.append(code("""final = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_14B,
    'other_model': MODEL_69B,
    'stage1_question_bank_size': len(kept) if STAGE1_PASS else 0,
    'stage1_pass': bool(STAGE1_PASS),
    'stage2': stage2,
    'stage3_pass': bool(replicates) if STAGE2_PASS else False,
    'stage3_n_meta_heads': int(len(meta)) if STAGE2_PASS else None,
    'overall_verdict': (
        'REPLICATES' if (STAGE1_PASS and STAGE2_PASS and replicates)
        else ('PARTIAL' if STAGE1_PASS and STAGE2_PASS else 'FAILS_AT_STAGE_1_OR_2')
    ),
}
summary_path = os.path.join(OUT_DIR, 'tier2_t24_summary.json')
with open(summary_path, 'w') as f:
    json.dump(final, f, indent=2, default=str)
log(f'saved {summary_path}')
print('\\n=== T2.4 FINAL ===')
for k, v in final.items():
    print(f'  {k}: {v}')

try:
    from google.colab import files
    for p in [QB_PATH,
              os.path.join(OUT_DIR, 'tier2_t24_decisive.json'),
              os.path.join(OUT_DIR, 'tier2_t24_ablations.csv'),
              summary_path]:
        if os.path.exists(p):
            files.download(p)
except Exception:
    pass"""))

    save('tier2_t24_self_14b.ipynb', cells)


if __name__ == '__main__':
    build_t21()
    build_t22()
    build_t23()
    build_t24()
    print('\nall 4 notebooks written')
