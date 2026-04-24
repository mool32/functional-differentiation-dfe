# Biology pre-registration v1 — Schiebinger MEF → iPSC primary test

**Locked:** 2026-04-24, **before any data computation**
**Context:** Cross-substrate validation of ML Phase 1 spectral invariant finding (rho = −0.42/−0.55/−0.48 across 160M/410M/1.4B Pythia). Tests whether the same operational measurement (participation ratio on internal representation at early specification timepoint) predicts perturbation sensitivity at maturity in cellular differentiation.

**Based on:** `biology_validation_plan.md` (commit `pre-locked`) from user 2026-04-24.

This document binds analysis decisions. Deviations will be explicitly labeled post-hoc in results.

---

## 1. Data access assertion

Data location: `/Users/teo/Desktop/research/perceptual_modules/paper6/data/schiebinger2019/` (raw zip) and `/Users/teo/Desktop/research/perceptual_modules/paper6/results/schiebinger_scored.h5ad` (processed, 236,285 cells × 19,089 genes, day 0 to day 18, 39 timepoints).

**No analysis or inspection of any correlation or outcome variable has occurred prior to locking this document.** Only: (a) counts of cells per day, (b) listing of h5ad columns and uns keys, (c) listing of files in the data zip. Pre-commitment verified by absence of any ρ, p-value, or scatter-plot operation in session history up to commit of this document.

## 2. Operational definitions (locked)

### 2.1 Participation ratio per cell

```python
def participation_ratio(x):
    """Per-cell participation ratio across genes.
    x: non-negative expression vector (log-normalized, top-HVG slice)
    Returns scalar in (0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    s1 = x.sum()
    s2 = (x ** 2).sum()
    if s2 <= 0:
        return np.nan
    return (s1 ** 2) / (n * s2)
```

This is the formula from biology plan §6 step 2. Note: this uses `x` not `x²` (differs from the σ-based ML formula but is the plan's explicit definition). Uniform x → PR = 1; fully concentrated → PR = 1/n. LOW PR = concentrated, HIGH PR = uniform.

Applied on: top 2000 highly-variable genes (if `var['highly_variable']` has fewer, use all marked HV; if more, select top 2000 by mean variance).

### 2.2 Fate proxy per day-2-3 cell

The plan pre-registered "success score at day 18 traced back via lineage or cluster-of-origin". Direct lineage tracing is not in Schiebinger dataset. We use **nearest-neighbor-based fate inference** on PCA space:

```python
def fate_proxy_by_nn(adata_early, adata_day18, ipsc_score, k=10, n_pcs=50):
    """
    For each cell in adata_early, find k nearest neighbors in PCA space of
    adata_day18 (concatenated; PCA fit on day-18 cells, projected on early).
    Fate proxy = mean ipsc_score of those k day-18 neighbors.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    # Fit PCA on day 18 (mature) space so embedding is defined by mature manifold
    X18 = _dense(adata_day18.X)  # on HVG slice
    X_early = _dense(adata_early.X)
    pca = PCA(n_components=n_pcs, random_state=20260424).fit(X18)
    E18 = pca.transform(X18)
    E_early = pca.transform(X_early)

    nn = NearestNeighbors(n_neighbors=k).fit(E18)
    dists, idx = nn.kneighbors(E_early)

    return ipsc_score[idx].mean(axis=1), dists.mean(axis=1)
```

`ipsc_score` is per-cell pluripotency gene-set score, defined below.

### 2.3 Pluripotency score per cell

Primary definition: simple z-scored mean expression of canonical pluripotency markers.

```python
PLURIPOTENCY_MARKERS = ['Nanog', 'Pou5f1', 'Sox2', 'Zfp42', 'Klf4',
                        'Esrrb', 'Tfcp2l1', 'Tbx3']  # canonical naive iPSC markers
def ipsc_score(adata):
    """Mean z-scored expression of pluripotency markers.
    Missing genes ignored. Returns np.ndarray of length n_cells.
    """
    markers = [g for g in PLURIPOTENCY_MARKERS if g in adata.var_names]
    sub = adata[:, markers].X
    sub = sub.toarray() if hasattr(sub, 'toarray') else sub
    # z-score per gene across all cells in this subset
    z = (sub - sub.mean(axis=0)) / (sub.std(axis=0) + 1e-9)
    return z.mean(axis=1).A1 if hasattr(z.mean(axis=1), 'A1') else z.mean(axis=1)
```

## 3. Pre-registered tests

### 3.1 Primary test (H1, headline α = 0.05)

**Test:** ρ(PR_day_2_3, fate_proxy_from_day18_NN) across cells present at day 2 or day 3 of Schiebinger reprogramming assay.

**Pre-registered direction:** **NEGATIVE** — cells with more concentrated expression at day 2-3 (lower PR) are more likely to receive high pluripotency scores from their day-18 neighbors, i.e. commit successfully to iPSC fate. Rationale restated from plan §5: cells that start distributed "remain plastic enough to reach iPSC" — distributed expression (high PR) → higher iPSC fate; concentrated (low PR) → stalled intermediates. Inverse correlation between PR and fate proxy.

**Wait — rereading plan §5 H1 carefully:**

> "Cells with more concentrated expression at day 2-3 (lower participation ratio, fewer dominant gene programs) more likely to commit successfully. Cells with more diffuse expression (higher participation ratio) more likely to stall in intermediate states."

So plan predicts: LOW PR → HIGH iPSC success. That is **NEGATIVE** ρ(PR, iPSC_success).

Confirmed predicted direction: **NEGATIVE**.

**Four-tier decision rule (matching ML pre-registration style):**

- **PASS:** |ρ| ≥ 0.20 AND direction negative AND bootstrap 95% CI excludes 0
- **PARTIAL:** |ρ| ∈ [0.10, 0.20] AND direction negative
- **WEAK:** |ρ| ∈ [0.05, 0.10] AND direction negative
- **FAIL:** wrong sign OR |ρ| < 0.05

### 3.2 Null control (H0)

**Test:** Repeat Primary on cells at day 0 (uninduced MEFs).

**Decision rule:**
- Required for Primary to count as early-window-specific: H0 |ρ| < 0.15
- If H0 |ρ| ≥ 0.15 in same direction as Primary: Primary is downgraded to "early-cell-intrinsic property" (not specifically early-window); interpretation of cross-substrate invariant is weaker.

### 3.3 Phase transition (H3, exploratory)

**Test:** Compute ρ(PR_t, fate_proxy_NN_from_day18) for t ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 9, 12, 15, 18}.

**Exploratory — no pre-registered threshold.** Reported as qualitative plot.

### 3.4 Sensitivity reruns

The following are pre-registered robustness checks. Will be run if Primary passes or partials:
- **HVG count:** rerun at N=500, 1000, 4000 HVGs; verify |ρ| doesn't jump by >0.10 with choice
- **NN k:** rerun at k=5, 20, 50; verify |ρ| stable
- **PCA components:** rerun at n_pcs=20, 100; verify |ρ| stable

## 4. Decision logic (joint interpretation table)

| H1 | H0 | H3 | Interpretation |
|----|----|----|--------------------|
| PASS | PASS | sign flip | **Cross-substrate invariant confirmed.** Proceed to Bastidas-Ponce replication. |
| PASS | FAIL | — | Signal broader than early-window. Reframe: "developmentally-persistent invariant". |
| PARTIAL | PASS | — | Weak but directionally correct. Run Bastidas-Ponce before deciding narrative. |
| PARTIAL | FAIL | — | Weak + baseline artifact. Probably not a real early-window effect. |
| WEAK | — | — | Below paper threshold. Document as exploratory null. |
| FAIL | — | — | Phenomenon is not reducible to this operational definition in reprogramming data. Try alternative measurements (module entropy secondary) before final negative claim. |

## 5. Deviations from user's plan

Two clarifications introduced in this operationalization that should be noted in any write-up:

1. **Fate proxy via NN in day-18 PCA space** rather than lineage tracing (user's plan says "cluster-of-origin if direct lineage unavailable"; NN is an implementation of that).
2. **Pluripotency score from canonical markers** rather than full-trajectory success label; user's plan says "binary or graded success score"; canonical marker mean is a graded continuous version.

Both are faithful to plan intent. Flagged here for transparency.

## 6. Compute budget

Entirely CPU. Single-cell operations on log-normalized HVG slice of Schiebinger scored h5ad. Estimated:
- Preprocessing + HVG selection: 2 min
- PR computation (~20k cells at day 2-3): 1 min
- PCA on day 18 (~7k cells × 2000 HVGs): 30 sec
- NN search: 30 sec
- Bootstrap CI (10k resamples of Spearman): 2-5 min
- Total Primary + H0 + H3: ~30 min CPU

## 7. Artifacts to produce

- `tables/biology_schiebinger_primary.csv` — per-cell rows: cell_id, day, PR, pluripotency_score, fate_proxy_nn
- `analyses/biology_schiebinger_summary.json` — ρ values, CIs, p, verdicts for H1 + H0 + H3
- `figures/biology_schiebinger_primary_scatter.pdf` — PR vs fate_proxy with direction indicator
- `figures/biology_schiebinger_phase_transition.pdf` — ρ(PR_t, fate_proxy) vs t

## 8. Post-commit, pre-run checklist

Before running the analysis script:
1. This file committed ✓ (on git push)
2. Analysis script committed with exact function definitions
3. Only THEN: run analysis

Hashes of pre-registration and code recorded in `biology_schiebinger_summary.json`.

---

*Locked 2026-04-24 before Schiebinger data has been correlated with any outcome.*
