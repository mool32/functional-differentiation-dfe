# Pre-registration v4 — SARS-CoV-2 RBD per-residue cross-substrate test

**Locked:** 2026-04-24, before any predictor–outcome correlation has been computed on RBD data.

**Binding under the six hard rules accepted this session:**
1. Direction pre-registered absolute. Positive result = FAIL, no reformulation.
2. Single primary pre-registered test. One measurement, one correlation, drives the verdict.
3. Numeric decision rule pre-locked below.
4. Null is legitimate outcome. "Invariant exists in ML, fails at viral level" is publishable.
5. Post-hoc reformulation prohibited.
6. Pre-registration commit hash baked into verdict JSON.

Previous biology attempts (Schiebinger / Bastidas-Ponce) are locked as either null / HOLD or as independent finding distinct from ML claim. This v4 is a clean one-shot test on a simpler substrate (viral protein, no cellular machinery).

---

## 1. Conceptual framing

**ML test (established):** per-head within one model. OV_PR at step 1000 → |Δ_head| at step 1000. Negative correlation, ρ ≈ −0.48 across three Pythia scales + OLMo pending.

**Biology analog (this test):** per-residue within one viral protein. Residue-level structural concentration → per-residue fitness impact. Negative correlation predicted.

**Unit of analysis match:** head ↔ residue, model ↔ protein. Directly analogous to ML test. Not cell ↔ model (that framing failed in prior phase).

## 2. Data sources (fixed)

- **Structural data:** RCSB PDB entry `6M0J` — crystal structure of SARS-CoV-2 RBD bound to human ACE2. Chain **E** (RBD residues 333–526, approximately 194 residues).
- **DMS fitness data:** `https://raw.githubusercontent.com/jbloomlab/SARS-CoV-2-RBD_DMS/master/results/summary/single_mut_effects.csv` — Starr, Greaney, Bloom et al. 2020 *Cell*, per-mutant effects on yeast-display ACE2 binding and folding (expression).
  - If the file path has moved, fall back to `final_variant_scores.csv` at the same repository.
  - Primary assay: **bind_avg** (ACE2 binding, mean across barcodes).
  - Filter: Class `missense` only; exclude stop codons.

## 3. Operational definitions (LOCKED — no tweaking after data inspection)

### 3.1 Structural predictor (per residue)

```python
def per_residue_contact_PR(ca_coords, lambda_angstrom=8.0):
    """
    ca_coords: (N, 3) array of C-alpha coordinates for N residues in the protein.
    lambda_angstrom: decay length for continuous contact strength.

    Returns: (N,) array of per-residue participation ratio values.

    c_ij = exp(-d_ij / lambda)   for i != j, else 0 (self excluded)
    PR_i = (Σ_j c_ij)^2 / (N * Σ_j c_ij^2)
    Low PR_i  = concentrated: residue i has few strong specific contacts.
    High PR_i = distributed:   residue i has many weak contacts.
    """
    from scipy.spatial.distance import cdist
    d = cdist(ca_coords, ca_coords)
    np.fill_diagonal(d, np.inf)             # zero out self-contact
    c = np.exp(-d / lambda_angstrom)
    s1 = c.sum(axis=1)
    s2 = (c ** 2).sum(axis=1)
    n = len(ca_coords)
    return (s1 ** 2) / (n * s2)
```

**Locked parameters:** `lambda_angstrom = 8.0` (standard C-alpha contact scale covering secondary-structure interactions).

**Locked PDB entry:** `6M0J`, chain `E`. If RBD chain labeled differently in mmCIF, use the chain containing residues 333-526 of SARS-CoV-2 spike.

### 3.2 Outcome (per residue)

Per-residue fitness impact:
```python
fitness_per_residue[i] = mean over all missense substitutions at position i
                         of |bind_avg - bind_wildtype|
```

Interpretation: average magnitude of binding perturbation when you mutate residue i. Higher = residue more critical for ACE2 binding = more perturbation-sensitive (analog of |Δ_head|).

### 3.3 Alignment

- PDB residue numbering (from 6M0J chain E) and DMS residue numbering (Bloom lab spike numbering) both use SARS-CoV-2 spike reference numbering. They should align directly.
- Positions included in analysis: intersection of residues present in both PDB (with C-alpha coordinates) AND DMS data.
- Pre-registered exclusion: residues with fewer than 10 measured missense substitutions in DMS (incomplete coverage).

## 4. Primary pre-registered test

**Statistic:** Spearman ρ between `PR_i` and `fitness_per_residue_i`, across all residues passing the coverage filter.

**Predicted direction:** **NEGATIVE**. Low PR (specialized) → large fitness impact.

**Four-tier decision rule:**
- **PASS:** |ρ| ≥ 0.30 AND direction negative AND p < 0.01 AND methodology-null gate PASS.
- **PARTIAL:** 0.20 ≤ |ρ| < 0.30 AND direction negative AND methodology-null gate PASS.
- **WEAK:** 0.10 ≤ |ρ| < 0.20 AND direction negative AND methodology-null gate at least CAUTION.
- **FAIL:** direction positive AND |ρ| ≥ 0.10, OR |ρ| < 0.10.

Rule 1 enforcement: if ρ is POSITIVE with |ρ| ≥ 0.10, verdict is `FAIL_WRONG_DIRECTION`. No re-interpretation as "deeper invariant" permitted. The positive finding, if it occurs, is reported as a distinct observation requiring separate pre-registration.

## 5. Methodology-null gate

Before interpreting primary ρ, run 1000 shuffles of `fitness_per_residue` among residues. Compute null distribution of ρ.

- **Gate PASS** if null 95 % CI within `[-0.10, +0.10]` → primary test interpretable at the locked thresholds.
- **Gate CAUTION** if null 95 % CI within `[-0.15, +0.15]` → primary thresholds raised by +0.05 (PASS requires |ρ| ≥ 0.35, PARTIAL 0.25, WEAK 0.15).
- **Gate FAIL** if null 95 % CI exceeds ±0.15 → `HOLD` verdict regardless of observed ρ.

## 6. Secondary exploratory (reported, not decision-defining)

- Sensitivity check: rerun primary with `lambda_angstrom ∈ {6.0, 10.0, 12.0}`. Reported as descriptive range.
- Alternative outcome: repeat correlation using `expression_avg` (folding/expression DMS assay) instead of `bind_avg`. Reported separately. Primary decision stays based on binding.
- Region stratification: recompute correlation restricted to RBM (receptor binding motif, residues ≈ 437-507) vs non-RBM. Reported descriptively.

All secondaries flagged **EXPLORATORY** in any output.

## 7. What is NOT pre-registered

- Any structural measurement other than the PR formula in §3.1 with λ=8 Å.
- Any outcome other than `mean |Δbind_avg|` per residue with the coverage filter.
- Any DMS dataset other than the Bloom lab RBD DMS (2020 canonical).
- Any PDB entry other than 6M0J.

If this test gives NULL or WRONG_DIRECTION or HOLD, no rescue attempts. The honest outcome is reported.

## 8. Compute budget

Entirely CPU. Estimated:
- Download PDB 6M0J and DMS CSV: ~2 minutes
- Compute contact-PR per residue: ~5 seconds (n ≈ 194)
- Compute per-residue DMS aggregate: ~1 second
- Primary correlation + bootstrap + methodology null: ~2 minutes
- Secondary sensitivity: ~1 minute

**Total: ~5 minutes CPU.**

## 9. Artifacts

- `tables/biology_rbd_per_residue.csv` — per-residue: residue_id, PR, mean_abs_fitness, n_mutants
- `analyses/biology_rbd_summary.json` — pre-reg commit hash, primary ρ with CI, methodology-null gate, verdict, secondary sensitivity results
- `figures/biology_rbd_scatter.pdf` — primary scatter + methodology null histogram + RBM stratification

## 10. Pre-commit gate

This document + the analysis script are committed together in the same git commit **before** any per-residue correlation is computed. Commit hash is recorded in the verdict JSON as `pre_registration_commit`.

No data has been downloaded at the time of this commit. Download happens only after commit hash is finalized.

---

*Locked 2026-04-24. No post-hoc rescue permitted. One-shot test under six binding rules.*
