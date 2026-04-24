# Pre-registration v5 — per-protein viral DMS test on ProteinGym

**Locked:** 2026-04-25, before any protein-level correlation between structural predictor and DMS outcome has been computed.

**Relation to v4 (RBD per-residue FAIL_WRONG_DIRECTION, commit `fdef01e`):** v4 was a per-residue test. It failed under Rule 1 with strong positive direction opposite to prediction. v4 verdict stays locked.

**v5 is a NEW hypothesis under Rule 5:** different unit of analysis (per-protein, not per-residue). The hypothesis is that the ML per-head invariant has a proper biological analog at the per-protein scale — head ↔ protein, not head ↔ residue. RBD per-residue failure does not constrain this new test in either direction. The test is scientifically independent.

**Binding rules (same as v3/v4):**
1. Direction absolute, NEGATIVE. Positive with |ρ|>0.10 = FAIL_WRONG_DIRECTION.
2. Single primary test.
3. Numeric decision rule pre-locked.
4. Null legitimate.
5. No post-hoc rescue. If v5 FAILS too, biology cross-substrate claim is closed under the tested unit choices.
6. Pre-reg commit hash baked into verdict JSON.

---

## 1. Substrate + data sources

- **Substrate:** ProteinGym v1 DMS substitution assays, filtered to Virus taxon (n = 31 viral proteins per reference file `DMS_substitutions.csv`).
- **Fitness data:** ProteinGym v1 DMS substitutions parquet files (5 chunks) from `https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-0000{0..4}-of-00005.parquet`. Format per row: `mutant`, `mutated_sequence`, `target_seq`, `DMS_score`, `DMS_score_bin`, `DMS_id`.
- **Structural data:** AlphaFold DB prediction for each viral protein's UniProt ID, URL template `https://alphafold.ebi.ac.uk/files/AF-{UniProt_ID}-F1-model_v4.pdb`.
  - If AlphaFold structure not available for a UniProt ID, protein excluded from analysis and noted in output.
  - AlphaFold-only; no mixing with experimental PDB. Consistency of measurement across proteins > quality of any individual structure.
- **Reference file:** `DMS_substitutions.csv` from ProteinGym GitHub `main/reference_files/` (already downloaded, pre-commit). Used only for viral taxon filter and UniProt_ID lookup. No correlation computations on it.

## 2. Operational definitions (LOCKED)

### 2.1 Predictor per protein

Per-protein **contact-matrix spectral participation ratio**.

```python
def protein_contact_PR(ca_coords, lambda_angstrom=8.0):
    """
    ca_coords: (N, 3) C-alpha coordinates for the protein.
    Build continuous contact matrix C_ij = exp(-d_ij / lambda), diagonal = 0.
    Compute SVD singular values of C. Return PR of those singular values:
        PR = (sum sigma_i^2)^2 / sum sigma_i^4
    Low PR = concentrated spectral structure (few dominant modes).
    High PR = distributed (spread across many modes).
    """
    from scipy.spatial.distance import cdist
    d = cdist(ca_coords, ca_coords)
    np.fill_diagonal(d, np.inf)
    C = np.exp(-d / lambda_angstrom)
    sigma = np.linalg.svd(C, compute_uv=False)
    s2 = sigma ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())
```

**Locked parameters:** `lambda_angstrom = 8.0`.

Note: this is the **matrix SVD PR** (direct analog of ML OV_PR on a head's weight matrix), not per-row PR. The v4 test used per-row PR and failed. v5 uses the full-matrix SVD, which is the correct analog of OV_PR methodology.

### 2.2 Outcome per protein

```python
fitness_sensitivity_per_protein = mean(|DMS_score|) across all single-substitution mutations in that protein's DMS assay
```

Rationale: large `|DMS_score|` = large perturbation from WT fitness. Per-protein mean captures "how sensitive this protein is to random mutation" — direct analog of ML per-head mean ablation effect.

**Filter:** protein included iff its DMS assay has ≥ 100 single-mutation measurements.

### 2.3 Matching

- Each viral protein's DMS assay has `target_seq`. We compute contact-PR on AlphaFold structure of the corresponding UniProt ID.
- If structure residue count differs from `seq_len` in reference, use PDB coordinates as-is (AlphaFold covers full UniProt sequence, length should match).
- If AlphaFold structure not retrievable, exclude protein. Final n reported after exclusions.

## 3. Primary pre-registered test

**Statistic:** Spearman ρ between per-protein `contact_PR` and per-protein `mean |DMS_score|`, across all viral proteins passing inclusion filters.

**Predicted direction:** **NEGATIVE**. Low contact-PR (concentrated spectral structure) → large fitness sensitivity. Same sign as ML pre-registration.

**Four-tier decision rule:**
- **PASS:** |ρ| ≥ 0.30 AND direction negative AND p < 0.01 AND methodology-null gate PASS.
- **PARTIAL:** 0.20 ≤ |ρ| < 0.30 AND direction negative AND gate PASS.
- **WEAK:** 0.10 ≤ |ρ| < 0.20 AND direction negative AND gate at least CAUTION.
- **FAIL:** direction positive AND |ρ| ≥ 0.10 → `FAIL_WRONG_DIRECTION`; or |ρ| < 0.10 → `NULL`.

Expected n ≈ 25–31 after AlphaFold availability + DMS coverage filters. Statistical power for ρ = 0.30 at n = 30 is ~0.5, modest. Partial result at n ≈ 30 is informative but not overwhelming. This is honest: we do not claim high power; any partial/pass must survive methodology null.

## 4. Methodology null gate

1000 shuffles of `mean |DMS_score|` across proteins. Compute null ρ distribution.

- Gate **PASS** if null 95% CI within [−0.10, +0.10] → primary test interpretable.
- Gate **CAUTION** if within [−0.15, +0.15] → thresholds raised by +0.05.
- Gate **FAIL** if exceeds ±0.15 → HOLD regardless of observed ρ.

## 5. Secondary exploratory (reported, not decision-defining)

- Sensitivity: repeat primary with `lambda_angstrom ∈ {6.0, 10.0, 12.0}`.
- Split by DMS `coarse_selection_type` (Activity / OrganismalFitness / Binding). Reported.
- Binary outcome: use `DMS_score_bin == 0` fraction (fraction unfit mutations) instead of `|DMS_score|`. Reported.

All flagged **EXPLORATORY**. Primary decision based only on primary statistic.

## 6. What is NOT pre-registered

- Non-viral ProteinGym subsets.
- Alternative structural measurements (e.g., per-row PR as in v4, residue-level aggregates).
- Aggregation of DMS scores other than `mean |DMS_score|` (secondary exceptions listed).
- Any comparison re-interpreting v4 RBD failure or prior biology tests.

If v5 FAILS, no rescue. Biology cross-substrate claim is closed under all five tested operationalizations (Schiebinger PR, Pancreas V4a, RBD per-residue contact, ProteinGym per-protein spectral).

## 7. Compute

Entirely CPU. Estimated ~10 minutes total:
- Download 5 parquet files (~110 MB): 1–2 min
- Filter viral, download AlphaFold structures (31 × ~500 KB): 3 min
- Compute per-protein PR: 2 min
- Per-protein DMS aggregate: 1 min
- Bootstrap + methodology null: 2 min

## 8. Artifacts

- `tables/biology_proteingym_viral.csv` — per-protein: DMS_id, UniProt_ID, n_mutants, mean_abs_DMS, contact_PR, structure_source
- `analyses/biology_proteingym_viral_summary.json` — pre-reg commit hash, per-tier decision, primary ρ, null gate, secondary results
- `figures/biology_proteingym_viral_scatter.pdf` — scatter + null histogram + secondary sensitivity

## 9. Pre-commit gate

This document and the analysis script are committed together in the same git commit. No per-protein contact-PR × DMS correlation has been computed at commit time.

Script will record `pre_registration_commit` (this commit's hash) in output JSON per Rule 6.

---

*Locked 2026-04-25. Unit of analysis: protein. Rules 1–6 binding. No rescue.*
