# Pre-registration: static head-level invariants as predictors of DFE tail weight

**Locked:** 2026-04-24, before any Phase 1a computation
**Author:** Theodor Spiro (theospirin@gmail.com)
**Context:** Paper 2 follow-up — upgrade categorical "emergent/born" framing to measurement-based invariant framing; aiming for substrate-independent predictor of DFE tail structure.

This document is binding on analysis decisions. Results reported deviating
from these choices will be explicitly flagged.

---

## 1. Primary quantity and outcome

**Model under study:** Pythia 410M-deduped, revision `step143000` (final checkpoint used in Paper 2).

**Heads in scope:** the 144 heads sampled in Paper 2 (layers 0–23, heads [0, 3, 6, 9, 12, 15] per layer). Matches the ablation sweep data in `data/all_ablations.csv`.

**Primary per-head outcome:** `|Δ_h|` at `step143000`, where `Δ_h = -(L_ablated - L_baseline) / |L_baseline|` is the ablation effect on wikitext-103 validation loss from the Paper 2 sweep. Direct, continuous, unambiguous per-head scalar. Heads in the heavy tail of the DFE are by construction those with large `|Δ_h|`.

**Secondary per-head outcome:** `class_Δ_AIC_h`, the class-level
`Δ_AIC(Student-t vs Normal)` value for the class into which head `h` is classified (strict Paper 2 criteria, crit = 5e-4, born_low = 1e-4). Used only for the absorb-the-label test (§3.2). Not used as primary outcome because it is mechanically tied to class label and introduces partial circularity.

## 2. Candidate invariants and hierarchical registration

### 2.1 Primary test (one, threshold |ρ|>0.4 medium / |ρ|>0.6 strong)

**OV_participation_ratio_h**: participation ratio of the singular values of the OV circuit for head `h`.

Construction:
- Extract `W_V` slice for head `h` from fused `query_key_value.weight`: rows `[2·d_model + h·d_head : 2·d_model + (h+1)·d_head]`, columns `[:]`. Shape `(d_head, d_model) = (64, 1024)`.
- Extract `W_O` slice from `dense.weight`: columns `[h·d_head : (h+1)·d_head]`. Shape `(d_model, d_head) = (1024, 64)`.
- Form OV product: `M_OV = W_O @ W_V`. Shape `(1024, 1024)`, rank ≤ 64.
- Singular values `σ₁ ≥ σ₂ ≥ ... ≥ σ₆₄ ≥ 0`.
- Participation ratio: `PR = (Σ σᵢ²)² / Σ σᵢ⁴`. Bounded in `[1, 64]`; small PR = concentrated, large PR = spread.

### 2.2 Secondary exploratory (three, threshold |ρ|>0.5 for nominal significance)

- **QK_participation_ratio_h**: same formula on `M_QK = W_Q @ W_K^T`, shape (1024, 1024), rank ≤ 64.
- **OV_spectral_entropy_h**: `-Σ pᵢ log pᵢ` with `pᵢ = σᵢ² / Σ σⱼ²`, singular values of `M_OV`.
- **QK_spectral_entropy_h**: same for `M_QK`.

### 2.3 Control

- **random_control_h**: a single draw from `N(0,1)` per head (144 values total), fixed seed `20260424`. Passed through the full Phase 2 regression pipeline unchanged. Gives empirical null ρ distribution at n=144 — the "what ρ would random noise produce?" floor.

## 3. Phase 2 statistical tests

### 3.1 Univariate

Spearman ρ between each invariant and `|Δ_h|`, 10 000 bootstrap resamples of heads for 95% CI of ρ. Report ρ point estimate, 95% CI, two-sided p-value.

### 3.2 Absorb-the-label test

OLS regression:

```
|Δ_h| ~ β_0 + β_label · emergent_dummy_h + β_inv · z(invariant_h) + ε_h
```

where `emergent_dummy_h = 1` if head's class ∈ {emergent, growing} else 0 (these are the two heavy-tail classes from robustness analysis; merged because they were not pairwise distinguishable in 410M), and `z()` is z-score standardization.

Fit (a) with label only, (b) with invariant only, (c) joint. Report:
- R² for each fit
- Coefficient reduction: `(β_label_a - β_label_c) / β_label_a` (fractional)
- Significance of `β_inv` in joint fit

Decision rules (pre-registered):
- If `β_label` coefficient drops by ≥ 50 % when invariant enters AND `β_inv` remains significant: **invariant is a cause, label is observation.** Strong result.
- If both `β_label` and `β_inv` significant in joint fit with similar magnitudes: **invariant carries additional information beyond label.** Moderate result.
- If `β_inv` becomes insignificant in joint fit: **invariant tracks label but does not explain more.** Null result for invariant.

### 3.3 Random control

Repeat §3.1 and §3.2 using `random_control_h` in place of each invariant. Report resulting ρ and regression statistics. If Spearman ρ for random control at n=144 is e.g. 0.15, any invariant with |ρ| < 0.15 + 0.05 is within noise floor.

## 4. Direction predictions (pre-registered BEFORE running)

For all four spectral invariants:

**Predicted direction:** NEGATIVE Spearman ρ between invariant and `|Δ_h|`.

Equivalently: **low participation ratio (concentrated computation) → large |Δ| (heavy-tail contribution).**

**Mechanistic rationale:** A head with low OV participation ratio performs a concentrated, low-rank transformation — a specialized function. Ablating such a head removes one specific capability; tokens / contexts that rely on that capability take a disproportionate loss hit; tokens that do not rely on it are unaffected. This produces input-dependent ablation effect with high variance across the validation set, i.e. a large `|Δ|` averaged over the 25 × 4 × 2048 tokens in the Paper 2 eval, driven by the subset of tokens where the capability mattered. Consistent with canonical findings in mechanistic interpretability literature where specialized heads (induction, successor, duplicate-token) are disproportionately "critical" in ablation studies (Olsson et al. 2022, Wang et al. 2023).

**Alternate hypothesis tested:** NONE pre-registered. If the observed ρ has the opposite (positive) sign with |ρ| > 0.4, we report this as a finding *contrary* to the pre-registered direction, which is itself publishable information.

## 5. Multiple-comparison correction

Hierarchical pre-registration. Only the primary test (OV_participation_ratio, §2.1) carries the headline significance threshold. Secondary exploratory tests (§2.2) reported with the stricter |ρ| > 0.5 threshold.

Bonferroni-equivalent on primary alone: α_primary = 0.05. No further correction needed on the primary.

## 6. Checkpoint policy

- Phase 1a uses weights from `step143000` (final) only. All four spectral invariants computed once.
- Phase 3B (training-stage invariance) will later recompute the primary invariant at `step16000` and `step64000` and test whether its predictive power on the *final-checkpoint* `|Δ_h|` is stable across these snapshots. Pre-registered threshold for "stable": correlation between invariant at step16000 and at step143000 across 144 heads is |ρ| > 0.7, and neither step’s invariant predicts `|Δ_h|` with ρ more than 0.15 further from the step143000 value. Large deviations indicate transient property, not invariant.

## 7. Implementation constants

- Working precision: float32 on CPU (no GPU available as of 2026-04-24; weights from HF; total RAM ≈ 1.6 GB for 410M).
- SVD routine: `numpy.linalg.svd` on the explicit `(1024, 1024)` product matrices. Rank-64 truncation handled via sorting and keeping all non-negligible singular values (none are truncated; we keep all 1024 values, of which ≤ 64 are nonzero).
- Random seed for control draw: `20260424`.
- Bootstrap for ρ CIs: `N_BOOT = 10_000` using `numpy.random.default_rng(seed=20260424)`.

## 8. Outputs

- `/paper/tables/phase1a_spectral_invariants.csv` — per-head table: `layer, head, class, |Δ|_step143000, OV_PR, QK_PR, OV_entropy, QK_entropy, random_control`.
- `/paper/figures/phase1a_scatter_primary.pdf` — primary scatter OV_PR vs |Δ| with class colour.
- `/paper/figures/phase1a_scatter_all_invariants.pdf` — 2×3 grid (4 invariants + random control + emergent-label) vs |Δ|.
- `/paper/analyses/phase1a_summary.json` — ρ, CIs, regression coefs for all tests including the random control and absorb-label results.

---

*Locked and committed before Phase 1a runs.*
