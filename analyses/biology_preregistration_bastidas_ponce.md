# Biology pre-registration — Bastidas-Ponce pancreatic endocrinogenesis

**Locked:** 2026-04-24, before any correlation between V4a predictor and fate outcome has been computed on Bastidas-Ponce data.

**Purpose:** one-shot independent test of cross-substrate invariant claim. Replication test on biological system independent of Schiebinger reprogramming. After this test, **no additional methodology variants will be tried on Bastidas-Ponce data without a new pre-registration commit.**

**Status of Schiebinger findings (locked as preliminary):**
- Gene-PR: marginal (C2 z=-1.8, p=0.04), not robust to HVG choice
- V4a (module PC1 signed): at methodology noise floor (C2 z=-0.86, p=0.22), K_NN-dependent
- Interpretation: Schiebinger provides weak or null evidence; signal cannot be distinguished from methodology variance at current measurement framework

**This pre-registration governs Bastidas-Ponce as independent clean test.**

---

## 1. Data access assertion

Data: `/Users/teo/Desktop/research/perceptual_modules/paper6/results/bastidas_ponce_scored.h5ad` (36,351 mouse cells, 17,327 genes, 4 timepoints E12.5–E15.5). Already in pipeline (Paper 6). Has `obsm['module_scores']` (43 modules, same as Schiebinger) and `clusters_fig6_broad_final` with endocrine subtype annotations.

**I have previously inspected:**
- cell counts per day (E12.5: 10790, E13.5: 5042, E14.5: 9633, E15.5: 10886)
- cluster column names
- cluster value counts in `clusters_fig6_broad_final`

**I have NOT inspected:**
- Any correlation between any V4a-like predictor and any fate outcome
- Any PCA on module scores
- Any NN-based outcome derivation

## 2. Operational definitions (LOCKED — no tweaking after data seen)

### 2.1 Predictor: V4a signed PC1 of module activity

```python
def compute_V4a(module_scores, orientation_anchor):
    """
    module_scores: (n_cells, 43) non-negative matrix from obsm['module_scores']
    orientation_anchor: (n_anchor,) boolean mask of cells used to fix sign
        (True = cells expected to have LOW V4a under correct orientation)
    Returns: (n_cells,) signed V4a score.
    """
    pca = PCA(n_components=1, random_state=20260424).fit(module_scores)
    pc1 = pca.transform(module_scores).flatten()
    # Orientation rule: flip sign so that anchor-group mean is NEGATIVE (committed = low V4a).
    # If anchor group mean is already negative, keep as is.
    if pc1[orientation_anchor].mean() > 0:
        pc1 = -pc1
    return pc1
```

**Orientation anchor:** E15.5 cells with endocrine-lineage cluster labels (see §2.2). This fixes the sign convention: committed endocrine cells have LOW V4a, non-endocrine / progenitor cells have HIGH V4a.

### 2.2 Outcome: NN-propagated endocrine fate fraction

```python
ENDOCRINE_CLUSTERS = {'Alpha', 'Beta', 'Delta', 'Epsilon', 'Fev+',
                      'Ngn3 high EP', 'Ngn3 low EP'}

def fate_endocrine_fraction(
    E15_cells_labels,        # 1D bool: is endocrine
    E15_cells_hvg_expr,      # (n_E15, n_HVG) log-normalized
    E12_cells_hvg_expr,      # (n_E12, n_HVG)
    k=10, n_pcs=50):
    """Per-E12.5 cell: fraction of k nearest E15.5 neighbors that are endocrine."""
    pca = PCA(n_components=n_pcs, random_state=20260424).fit(E15_cells_hvg_expr)
    E_15 = pca.transform(E15_cells_hvg_expr)
    E_12 = pca.transform(E12_cells_hvg_expr)
    nn = NearestNeighbors(n_neighbors=k).fit(E_15)
    _, idx = nn.kneighbors(E_12)
    return E15_cells_labels.astype(float)[idx].mean(axis=1)
```

### 2.3 Parameters (LOCKED)

- HVG count for PCA: **N_HVG = 2000** (top 2000 by variance if `highly_variable` column insufficient)
- PCA components for NN: **N_PCS = 50**
- NN k: **K_NN = 10**
- Random seed everywhere: **20260424**

**K_NN is locked at 10.** We do NOT sweep K_NN in this test. Schiebinger K_NN-sweep was a post-hoc exercise. Here we commit to K=10 as the methodology.

## 3. Pre-registered hypotheses

### 3.1 H1 Primary (four-tier)

**Test:** Spearman ρ(V4a_E12.5, endocrine_fraction_NN) across all E12.5 cells with non-missing V4a.

**Predicted direction:** NEGATIVE.

**Decision tiers (in rank order — first satisfied wins):**

- **PASS (strong)**
  - `|ρ| ≥ 0.25`
  - Direction: NEGATIVE
  - `p < 0.01`
  - Methodology-null gate passed (§3.2)

- **PARTIAL**
  - `|ρ| ≥ 0.15`
  - Direction: NEGATIVE
  - Methodology-null gate passed

- **WEAK**
  - `|ρ| ≥ 0.10`
  - Direction: NEGATIVE
  - Methodology-null gate passed OR not (flagged)

- **FAIL_WRONG_DIRECTION**
  - Sign POSITIVE and `|ρ| ≥ 0.10`

- **NULL**
  - `|ρ| < 0.10`

### 3.2 Methodology-null gate (MANDATORY, baked in)

Run before interpreting observed ρ. Shuffle endocrine labels among E15.5 cells. Recompute fate fraction. Correlate with V4a at E12.5 (unchanged). Repeat n = 200 shuffles.

**Gate criteria:**

- If null 95% CI is **within [−0.10, +0.10]** → gate PASS, methodology noise floor is below partial threshold. Observed ρ interpretable by H1 rules.
- If null 95% CI is **[−0.15, +0.15]** → gate CAUTION. Observed ρ only counted if |ρ| ≥ 0.20.
- If null 95% CI **exceeds ±0.15** → gate FAIL. Verdict = HOLD regardless of observed magnitude. Methodology cannot distinguish signal from variance.

### 3.3 Secondary pre-registered

**S1 — H0-analog null.** Because Bastidas-Ponce has no pre-induction uninduced timepoint (E12.5 IS the early timepoint), we use different control: compute V4a and fate fraction for a scrambled-cell-identity control (random E15.5 cells as "fate targets"). Reported as sanity.

**S2 — Phase trajectory.** Repeat the ρ(V4a_t, endocrine_fraction_NN) at each available timepoint t ∈ {E12.5, E13.5, E14.5, E15.5}. Report as descriptive plot (no decision).

## 4. What is NOT pre-registered (and will not be run as post-hoc rescue)

- Alternative V4a variants (V2/V3/V4b) — ruled out by Schiebinger test
- Different K_NN sweeps
- Different N_HVG or N_PCS
- Module subsets other than all-43
- Classifier-based fate proxy
- WOT trajectory inference
- Any outcome definition except endocrine-fraction-NN

**If the pre-registered test yields NULL, interpretation is "biology cross-substrate claim not supported at current methodology framework on this independent dataset." Not "try another variant."**

## 5. Decision tree (joint over H1 verdict + methodology gate)

| H1 | methodology gate | meaning |
|----|-------------------|---------|
| PASS | gate PASS | **Clean independent replication of ML invariant in biology.** Cross-substrate claim supported at partial/strong magnitude. |
| PARTIAL | gate PASS | Weak but genuine replication. Publishable as partial finding. |
| WEAK | gate PASS | Signal exists but below robust threshold. Report as suggestive; do not claim cross-substrate universal. |
| any | gate CAUTION | Interpret at 0.20 threshold; PASS becomes partial, PARTIAL becomes null. |
| any | gate FAIL | HOLD verdict. Methodology is not informative on this dataset. |
| FAIL_WRONG_DIRECTION | any | Opposite-direction signal — either signal exists but our mechanistic framing is wrong, or dataset-specific effect. Reported honestly as refuting pre-registered hypothesis. |
| NULL | any | Independent confirmation that biology cross-substrate signal is not detectable by current methodology. ML finding stands on its own. Biology validation → future work. |

## 6. Run procedure (LOCKED)

1. Lock this doc + analysis script in same git commit.
2. Run script once.
3. Record verdict in `biology_bastidas_ponce_summary.json`.
4. **No re-runs, no parameter tweaks, no alternative methodology.**
5. Commit result.
6. Accept verdict.

## 7. Expected output artifacts

- `tables/biology_bastidas_ponce_primary.csv` — per E12.5 cell: V4a, endocrine_fraction, fate NN distances
- `analyses/biology_bastidas_ponce_summary.json` — all verdicts + methodology-null distribution
- `figures/biology_bastidas_ponce_scatter.pdf`
- `figures/biology_bastidas_ponce_null.pdf`
- `figures/biology_bastidas_ponce_trajectory.pdf`

---

*Locked 2026-04-24. Pre-registration commit hash recorded in summary JSON. One-shot test; no post-hoc rescue permitted.*
