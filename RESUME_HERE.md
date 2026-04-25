# Resume here — for next session

**Last context end:** 2026-04-25 ~13:58 UTC, mid-TinyLlama Colab run.
**Latest commit on main:** `e0693a5` (TinyLlama torch.load idempotent patch).

## What's running right now

**TinyLlama-1.1B Colab notebook** — pre-reg `7523931`. At session end it was at step-10k checkpoint, 140/176 ablations done, ~14 s/head, baseline loss 3.02. ETA for full sweep (7 checkpoints) ≈ 3 hours from notebook start.

**Output destination:** `/content/drive/MyDrive/DFE_research/tier2_tinyllama/`
Files when done:
- `tier2_tinyllama_ablations.csv`
- `tier2_tinyllama_spectral.csv`
- `tier2_tinyllama_verdict.json`
- `wt103_batches.pt`

## What to do in next session

### Step 1 — read state
1. Read `PROJECT_MASTER_INDEX.md` (single source of truth across project)
2. Read this file (RESUME_HERE.md)
3. Check git log: `cd "paper" && git log --oneline -10`

### Step 2 — check TinyLlama status
- If user's Colab still running or finished: download the three files via @path or Drive zip
- Place them at `paper/data/tier2_tinyllama_*.csv`
- Run verdict analysis (script below)

### Step 3 — verdict computation script

For TinyLlama, run analysis identical to OLMo (commit `bea4ffc`, file `analyses/tier2_olmo_verdict.json`). Quick recipe:

```python
import pandas as pd, numpy as np
from scipy import stats as sp

abl = pd.read_csv('data/tier2_tinyllama_ablations.csv')
spec = pd.read_csv('data/tier2_tinyllama_spectral.csv')
m = spec.merge(abl, on=['checkpoint_k','revision','layer_idx','head_idx'], how='outer')
m['abs_delta'] = m['delta'].abs()

# PRIMARY: rho(OV_PR @ step-10k, |Delta| @ step-10k)
d10k = m[m.checkpoint_k == 10]
rho, p = sp.spearmanr(d10k['OV_PR'], d10k['abs_delta'])
print(f'Primary: rho={rho:+.4f}, p={p:.2e}, n={d10k.dropna(subset=["OV_PR","abs_delta"]).shape[0]}')

# Methodology null gate
rng = np.random.default_rng(20260425)
clean = d10k.dropna(subset=['OV_PR','abs_delta'])
nulls = [sp.spearmanr(clean['OV_PR'], rng.permutation(clean['abs_delta'])).statistic for _ in range(200)]
ci = (np.percentile(nulls, 2.5), np.percentile(nulls, 97.5))
print(f'Null span: [{ci[0]:+.3f}, {ci[1]:+.3f}], gate {"PASS" if max(abs(ci[0]),abs(ci[1])) <= 0.10 else "CAUTION" if max(abs(ci[0]),abs(ci[1])) <= 0.15 else "FAIL"}')
```

### Step 4 — apply pre-reg v6 decision rule

Pre-reg `7523931` thresholds:
- PASS: |ρ| ≥ 0.30, neg, p<0.01, gate PASS
- PARTIAL: 0.20–0.30, neg, gate PASS
- WEAK: 0.10–0.20, neg, gate ≥ CAUTION
- FAIL_WRONG_DIRECTION: positive AND |ρ| ≥ 0.10
- NULL: |ρ| < 0.10
- HOLD: gate FAIL regardless

Reference window: ρ ∈ [−0.55, −0.42] from prior 4 scales.

### Step 5 — commit verdict

Same pattern as OLMo (`bea4ffc`):
```
verdict_data = {
    'pre_registration_commit': '7523931',
    'pre_registration_file': 'invariants_preregistration_v6_tinyllama.md',
    'model': 'TinyLlama/tinyLlama-intermediate-checkpoints',
    'primary_result': {...},
    'primary_verdict': '...',
    'methodology_null': {...},
    'cross_scale_comparison': {
        'Pythia_160M': -0.416,
        'Pythia_410M': -0.555,
        'Pythia_1.4B': -0.484,
        'OLMo2_1B': -0.4868,
        'TinyLlama_1.1B': <new_value>,
    },
}
```

### Step 6 — decide writeup direction

If TinyLlama PASS (likely given 4/4 prior PASS):
- 5 scales × 3 architecture families × 3 teams locked
- Begin Paper outline (already drafted in `PROJECT_MASTER_INDEX.md` §7)

If TinyLlama FAIL_WRONG_DIRECTION or HOLD:
- Honest scope downgrade. Paper covers Pythia + OLMo only.
- New pre-reg required for any rescue attempt.

## Pre-registration discipline still binding

**Six rules locked since session 2026-04-24:**
1. Direction NEGATIVE absolute. Positive = FAIL.
2. Single primary test per pre-reg.
3. Numeric thresholds locked.
4. Null is legitimate.
5. Post-hoc reformulation prohibited (new finding = new pre-reg).
6. Pre-reg commit hash in verdict JSON.

## What NOT to do (without explicit user approval)

- Do not start new biology tests.
- Do not write paper text.
- Do not modify locked pre-registration documents.
- Do not run new ML model tests beyond TinyLlama without locking new pre-reg.

## Key file paths (quick reference)

```
paper/PROJECT_MASTER_INDEX.md            ← navigation
paper/SESSION_STATE_2026_04_24.md        ← ML deep history
paper/SESSION_STATE_BIOLOGY_2026_04_24.md ← biology deep history
paper/analyses/phase1a_findings_framing.md ← canonical language for writeup
paper/analyses/invariants_preregistration_v{1,2,3,4,5,6}*.md ← all pre-regs
paper/analyses/tier2_olmo_verdict.json   ← template for tinyllama_verdict
paper/data/tier2_pre1p4b_*.csv           ← Pythia 1.4B raw
paper/data/tier2_olmo_*.csv              ← OLMo raw
paper/tier2_tinyllama_validation.ipynb   ← currently-running notebook
```

---

*Resume by reading this file first, then PROJECT_MASTER_INDEX.md, then git log.*
