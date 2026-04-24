# Biology Validation Plan: Early-Window Invariant in Cellular Differentiation

**Author:** T. Spirin
**Date:** 2026-04-24
**Status:** Pre-experiment plan, before any data analysis
**Context:** Following ML invariant finding (OV_PR at step 1000 predicts |Δ| at three scales: 160M, 410M, 1.4B Pythia, ρ = -0.42 to -0.55, all pre-registered)

---

## 1. The hypothesis being tested

In Pythia transformers across three scales, an early structural measurement (OV participation ratio at training step 1000, less than 1% of training) predicts which attention heads will be critical at the end of training (correlation with ablation effect magnitude |Δ|, ρ ≈ -0.5).

**Substrate-independent claim under test:**

> In any adaptive system undergoing differentiation from initially-similar units to functionally-specialized roles, an early structural measurement predicts later perturbation sensitivity. This early-window predictability is a universal feature of evolving systems, not a transformer-specific phenomenon.

The biology test: in cellular development from progenitor to differentiated cell types, an early structural measurement on individual cells should predict which cells become functionally critical (most sensitive to perturbation) at maturity.

**If confirmed:** First substrate-independent measurement of early specification window. Programme moves from speculation to empirical universality.

**If refuted:** Phenomenon is transformer-specific. ML finding stands on its own. Programme refines its substrate-independence claim.

---

## 2. Why cellular development specifically

Three reasons.

**Conceptual fit.** Cell differentiation has documented early specification windows analogous to ML early training. Embryonic cells start similar, commit to fates over hours-days, and end up with specialized roles. This is the closest biological analog to attention heads starting from similar initialization and differentiating into specialized roles during training.

**Data availability.** Time-resolved scRNA-seq datasets exist for multiple developmental systems with both early timepoints AND adult-cell perturbation data. We can directly test the analog of "early measurement predicts adult perturbation sensitivity".

**Cost.** Existing public datasets, Python infrastructure already built for perceptome work, no wet-lab requirement.

---

## 3. The exact mapping

Need explicit one-to-one correspondence between ML and biology variables. Define before any data work.

| ML (already established) | Biology (to test) |
|---|---|
| Attention head | Cell |
| Initialization (step 0) | Progenitor cell, undifferentiated |
| Training step 1000 (≈0.7% of training) | Early specification timepoint, ~10-20% through differentiation |
| Final model (step 143000) | Mature/differentiated cell type |
| OV participation ratio per head | Spectral concentration measure per cell |
| |Δ| from ablation sweep | Cell-type sensitivity to perturbation |
| ρ(OV_PR_step1000, |Δ|_step143k) | ρ(early_spectral_measure, mature_perturbation_sensitivity) |
| Pre-registered direction: negative | Pre-registered direction: TBD before data |

**Key methodological decision before start:** define "spectral concentration per cell" precisely. Three candidates ranked by closeness to ML measurement:

(a) **PCA participation ratio** of single cell's expression vector across genes (direct analog of OV_PR computed on weight matrix). Compute per cell at early timepoint. This is the closest mathematical analog.

(b) **Module activity entropy** using existing 43-module perceptome. For each cell, compute Shannon entropy of distribution of module activities. Lower entropy = more concentrated activity in few modules. This connects to your perceptome programme.

(c) **Gene covariance dimensionality** locally per cell using neighborhood. Computationally heavier but most biologically interpretable.

Pre-register choice: **(a) primary, (b) secondary**. (c) excluded due to computational cost and methodological complexity for first test.

---

## 4. Datasets and why each

Three datasets, ranked by suitability. Run on first; if positive, validate on second and third.

**Primary: Schiebinger 2019 (MEF → iPSC reprogramming)**
- 236,285 mouse cells, 39 timepoints over 18 days
- Already in your pipeline (used in Paper 6)
- Critical advantage: clear initial state (uniform fibroblasts), clear endpoint state (iPSC vs failed-reprogramming intermediates)
- Early timepoint candidate: day 2-3 (≈10-15% through reprogramming)
- Late perturbation sensitivity proxy: which cells reach iPSC vs which stuck in intermediates
- Limitation: "perturbation sensitivity" here is endogenous (success/failure of reprogramming) rather than experimental knockout

**Secondary: Bastidas-Ponce 2019 (mouse pancreas endocrinogenesis)**
- 36,351 mouse cells, 4 embryonic timepoints E12.5-E15.5
- Already in your pipeline (used in Paper 6)
- Critical advantage: known lineage tree from progenitor to endocrine subtypes (alpha, beta, delta, epsilon)
- Early timepoint: E12.5 multipotent progenitors
- Late timepoint: E15.5 differentiated subtypes
- Perturbation data: DepMap / pancreatic CRISPRi datasets exist for adult cells of these lineages
- Limitation: only 4 timepoints, less granular than Schiebinger

**Tertiary: Mouse Organogenesis Cell Atlas (MOCA, Cao 2019)**
- ~2 million cells, multiple developmental timepoints, multiple tissues
- Broad scale, multiple lineages tested simultaneously
- Limitation: large compute requirement, run only if primary + secondary positive

For **perturbation sensitivity ground truth** (the |Δ| analog), three options:
- DepMap Achilles essentiality scores (per gene per cell line) - cleanest mapping
- Replogle 2022 K562 Perturb-seq (already in your pipeline) - within-cell-type variation
- Endogenous fate failure rates from lineage tracing - implicit perturbation

Pre-register choice: **DepMap essentiality scores** as primary perturbation ground truth. Most analogous to ablation effect (gene knockout vs head ablation, both produce per-unit fitness effects).

---

## 5. Pre-registered hypotheses

**Primary hypothesis (H1):**

For mouse fibroblast → iPSC reprogramming (Schiebinger), at day 2-3 of reprogramming, per-cell PCA participation ratio across gene expression correlates with successful-iPSC-fate-attainment at day 18.

**Direction:** TBD before data. State direction now to commit.

Predicted direction: **negative correlation**. Cells with more concentrated expression at day 2-3 (lower participation ratio, fewer dominant gene programs) more likely to commit successfully. Cells with more diffuse expression (higher participation ratio) more likely to stall in intermediate states.

Reasoning: in ML, low OV_PR (concentrated) heads were the ones that became dead/never-critical, high OV_PR (distributed) became active/important. Translating: cells starting concentrated get "stuck" in early role (analog of dead heads), cells starting distributed remain plastic enough to reach iPSC (analog of active heads).

Magnitude prediction: |ρ| ≥ 0.20 (more conservative than ML; biology has more confounds).

**Secondary hypothesis (H2):**

For pancreatic endocrinogenesis (Bastidas-Ponce), at E12.5, per-cell PCA participation ratio across gene expression correlates with adult cell type DepMap essentiality.

**Direction:** Same as H1, negative correlation.

Magnitude: |ρ| ≥ 0.15.

**Secondary hypothesis (H3) - phase transition:**

In both datasets, the relationship between early measurement and late outcome should change sign during differentiation, analogous to the ML phase transition between step 2000 and step 4000.

**Test:** Compute ρ(participation_ratio at timepoint t, final outcome) for each timepoint t. Look for sign flip.

This is exploratory; we have no theoretical prediction for when in biological time the flip occurs.

**Null control (H0):**

For each cell, compute participation ratio at the earliest available timepoint (analog of ML step 0). This should NOT correlate with final outcome.

If H0 fails (early random correlation present), this suggests confounding (technical artifact, batch effect) rather than real signal.

---

## 6. Analysis pipeline

Step-by-step. Each step has clear input, output, and decision rule.

**Step 1: Data acquisition and preprocessing**

Load Schiebinger Day 2-3 cells. Filter to high-quality cells (standard QC: min counts 1000, max mitochondrial 10%). Normalize and log-transform. Output: cell × gene matrix at early timepoint, with annotations of final-day fate.

**Step 2: Per-cell participation ratio computation**

For each cell at early timepoint:
- Take its gene expression vector
- Compute participation ratio: PR = (Σ x_i)² / (n × Σ x_i²) where x_i are normalized expression values
- This gives one number per cell

Alternative: compute on top-N highly variable genes only (N=2000) to reduce noise. Pre-register N=2000.

Output: per-cell PR vector at day 2-3.

**Step 3: Late outcome measurement**

For Schiebinger: classify each cell's final fate at day 18 using existing annotations (iPSC, intermediate, failed reprogramming).

Map back to early-timepoint cells via lineage tracing or via cluster-of-origin if direct lineage unavailable.

Output: per-cell binary or graded "success score" at day 18 for cells present at day 2-3.

**Step 4: Primary correlation test**

Compute Spearman ρ(PR_day2_3, success_score_day18) across all cells with both measurements.

Compute 95% bootstrap CI.

Compare against pre-registered threshold (|ρ| ≥ 0.20, negative direction).

Decision rule:
- **Pass:** |ρ| ≥ 0.20, negative direction, bootstrap CI excludes 0
- **Partial:** |ρ| in [0.10, 0.20] in pre-registered direction
- **Fail:** sign wrong or |ρ| < 0.10

**Step 5: Null control**

Repeat Step 4 using day 0 cells (uninduced fibroblasts) instead of day 2-3.

If correlation present at day 0, signal is not specific to early-window phenomenon - it's a baseline cellular property. Pre-register that day 0 |ρ| must be < 0.15 for primary finding to count as early-window-specific.

**Step 6: Phase transition analysis (exploratory)**

Compute ρ(PR_t, success_score_day18) for each timepoint t in {day 0.5, 1, 2, 3, 4, 6, 9, 12}.

Plot trajectory. Look for sign flip analogous to ML.

This is exploratory - no pre-registered prediction about specific timepoint.

**Step 7: Replication on Bastidas-Ponce**

Repeat Steps 1-6 on Bastidas-Ponce dataset. Use E12.5 as early timepoint, mature cell type at E15.5 as late state, DepMap essentiality of mature lineage as perturbation ground truth.

Modified analysis: instead of "success score", use "essentiality score of mature lineage cell descended from this progenitor".

---

## 7. Decision tree based on results

**Outcome A: Primary passes on Schiebinger AND replicates on Bastidas-Ponce**

This is two-substrate confirmation. Major result.

Action: Begin paper draft connecting ML and biology findings under unified framework. Target venue: Nature Communications or PNAS. Cross-substrate measurement of early specification window is novel claim worth significant venue.

**Outcome B: Primary passes on Schiebinger but not Bastidas-Ponce**

Mixed result. Phenomenon may be reprogramming-specific rather than developmental-general.

Action: Investigate why second dataset fails. Possible reasons: timepoint mismatch (E12.5 may not be the right "step 1000" analog for endocrinogenesis), perturbation ground truth mismatch (DepMap may not capture relevant sensitivity), or genuine phenomenon limitation. Run third dataset (MOCA) before deciding final framing.

**Outcome C: Primary partial on Schiebinger**

Direction correct but weak signal. Suggests phenomenon real but more confounded in biology than ML.

Action: Run on Bastidas-Ponce anyway. If second dataset also partial in same direction, two weak signals together are publishable as exploratory cross-substrate observation. Frame conservatively as "preliminary biological evidence consistent with ML invariant; further work needed".

**Outcome D: Primary fails on Schiebinger**

Phenomenon does not transfer cleanly to biology.

Action: Important negative result. Three possibilities to investigate before final negative claim:
- Wrong measurement choice: try secondary measurement (module entropy via perceptome)
- Wrong timepoint: scan multiple early timepoints
- Wrong dataset: try Bastidas-Ponce as primary on grounds it has cleaner trajectory

If all three fail, finding is ML-specific. Honest negative result for substrate-independence claim. ML paper still strong on its own.

---

## 8. Resource and timeline estimate

**Compute:** All work CPU-feasible. Schiebinger dataset already loaded. Estimated ~2 hours per dataset for full pipeline (preprocessing + PR computation + correlation analysis).

**Time:** 1-2 days for primary test on Schiebinger. 2-3 days for Bastidas-Ponce replication. 1 week for full pipeline including null controls and phase transition analysis.

**Dependencies:** None new. All datasets and tools already in existing pipeline.

---

## 9. What to commit before starting

To ensure scientific honesty:

1. This document, locked, before any data analysis
2. Specific code for participation ratio computation, locked before running
3. Pre-registered direction and magnitude thresholds, in this document
4. Pre-registered datasets and analysis order: Schiebinger first, Bastidas-Ponce second
5. Pre-registered decision rules above

Commit to Git with timestamp before any analysis begins. Hash recorded in subsequent results files.

---

## 10. What this validates if successful

If H1 + H2 both pass: substrate-independent invariant established with two independent biological systems plus ML.

This means: there exists a measurable property of adaptive systems undergoing differentiation - participation ratio of internal structure at early specification window - that predicts which units become functionally critical at maturity. This property works in transformers, in cellular reprogramming, and in pancreatic development. The mathematical form of the measurement is the same across substrates.

This is qualitatively stronger than:
- Watson's evolutionary connectionism (formal isomorphism without measurement)
- Vanchurin et al. (theoretical framework without cross-substrate test)
- Levin's TAME (conceptual framework without quantitative invariant)

It would be the first substrate-independent quantitative invariant of evolving adaptive systems, validated across at least two substrates.

This is the contribution that defines the next 5 years of the programme.

---

## 11. Honest limitations to state in any subsequent paper

Even if positive:

- Two substrates is suggestive, not conclusive. Universal claim requires more substrates over time.
- Correlation, not causation. We measure association, not mechanism. Mechanism follows.
- The mathematical analog (participation ratio of expression vector vs participation ratio of weight matrix) is structural, not exact. Future work needs to formalize the mapping.
- Biological "perturbation sensitivity" via DepMap is one operationalization. Other operationalizations may give different results.
- Schiebinger and Bastidas-Ponce are mouse systems. Human validation pending.

These limitations should be in any paper Section explicitly, not hidden.

---

End of plan.
