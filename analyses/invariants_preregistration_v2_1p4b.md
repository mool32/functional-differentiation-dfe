# Pre-registration v2 — Pythia 1.4B validation of early-training spectral signal

**Locked:** 2026-04-24
**Author:** Theodor Spiro (theospirin@gmail.com)
**Context:** Exploratory findings from Phase 1b on 410M and 160M require independent-scale validation. This document pre-registers the 1.4B tests **before** any 1.4B invariant or ablation data is computed. No GPU work on 1.4B has occurred at the time of locking.

---

## What this is and is not

This is **not** a rescue of pre-registration v1. Phase 1a primary test at step 143000 was null and remains null. The findings below are **new exploratory observations** from the dense scan, now being pre-registered *as new predictions* for a third, independent scale. The distinction is kept explicit.

This pre-registration commits to:
- specific per-prediction thresholds in advance
- decision rules for pass / partial / fail
- a single primary test that carries the headline significance
- ordering of secondary tests
- explicit statement of which v1 claims are being upgraded vs retained as null

Anything found on 1.4B that is *not* in this document is exploratory, not pre-registered.

## Summary of locked v1 status

| finding | phase | locked status |
|---------|-------|---------------|
| Pre-reg A: OV_PR @ step143k → |Δ|_step143k, \|ρ\|>0.4 neg. | H | **NULL (ρ = −0.126, p = 0.13)** |
| Finding B: QK active-vs-dead gap | E→H | replicates 410M + 160M, training-driven (step 0 gap ≈ 0) |
| Finding C: label breakdown in 160M | E | two scales, tentative |
| Finding D: step-1000 OV_PR signal | E | replicates 410M + 160M; within-class strong; lottery-ticket rejected |
| Finding E: ρ sign-flip between 1000 and 4000 | E | two scales, tentative |

## Experimental design on 1.4B

Pythia 1.4B-deduped revision `step{1000, 2000, 4000, 8000, 16000, 64000, 143000}` plus `step0`.

**Head sample:** 144 heads, one per (layer, head) slot on the 24-layer × 16-head architecture at heads [0, 3, 6, 9, 12, 15] per layer, matching Paper 2 and 410M sampling.

**Ablation sweep:** single-head attention-output ablation at each of the 7 non-zero checkpoints plus step 0 baseline. wikitext-103 25 × 4 × 2048 tokens evaluation, float32 + TF32 matmul, SHA-256 save/restore verification, CSV append-resume. This is the methodology used for 410M and 160M.

**Spectral invariants computed per head per checkpoint:** OV_PR, QK_PR (identical extraction as 410M script).

## Primary pre-registered test (one, carries headline α = 0.05)

**Claim:** On Pythia 1.4B at training step 1000, the per-head OV participation ratio correlates negatively with the per-head ablation effect |Δ| at the same checkpoint.

**Statistic:** Spearman ρ between `OV_PR_h` at step1000 and `|Δ_h|` at step1000, across all 144 sampled heads.

**Four-tier criterion (amended 2026-04-24 before any 1.4B data):**

- **Pass:** |ρ| ≥ 0.30 AND direction is negative AND p < 0.01.
- **Partial:** direction negative AND 0.20 ≤ |ρ| < 0.30.
- **Weak:** direction negative AND 0.10 ≤ |ρ| < 0.20. Reported as "signal directional but decaying with scale" — itself an interpretable outcome, not a moving goalpost (amendment made prior to data collection).
- **Fail:** direction positive OR |ρ| < 0.10.

*(Reference: 410M ρ = −0.555, 160M ρ = −0.416. If signal magnitude holds → easy pass. If it decays linearly with log-scale → would land near the Weak boundary at 1.4B, still interpretable as scaling-relevant observation.)*

## Secondary pre-registered tests (three, reported with stricter |ρ| threshold)

### S-1. Intra-class independence at step 1000

**Claim:** OV_PR at step 1000 predicts `|Δ|` at step 1000 within at least two of the three active classes (emergent, growing, born) — or, if born has n < 10, within emergent and growing alone.

**Statistic:** within-class Spearman ρ with sign negative and |ρ| ≥ 0.30.

**Pass:** at least two active classes meet the threshold.

**Fail:** fewer than two active classes meet the threshold; in that case the primary-test signal is absorbed by class label.

### S-2. Lottery-ticket control

**Claim:** OV_PR at step 0 (Pythia 1.4B actual init) does *not* correlate with `|Δ|` at step 1000 or step 143000.

**Statistic:** Spearman ρ on both correlations.

**Pass:** both |ρ| < 0.20 (consistent with 410M's +0.10 and +0.09).

**Fail:** either |ρ| > 0.20 — in that case the step-1000 signal is re-interpreted as lottery-ticket / init-driven, and Finding D is reframed.

### S-3. Two-phase sign flip (Finding E)

**Claim:** ρ(QK_PR, |Δ|) at same-checkpoint flips from negative to positive between step 1000 and step 8000, and reaches |ρ| ≥ 0.30 in positive direction by step 143000.

**Statistics:** Spearman ρ at steps 1000, 2000, 4000, 8000, 143000.

**Pass:** ρ at step 1000 negative (ρ < −0.15), ρ at step 143000 positive (ρ > +0.30), monotonic or near-monotonic transition between 1000 and 4000.

**Partial:** sign flip observed but not at pre-registered steps, OR final-step magnitude 0.10–0.30.

**Fail:** no sign flip observed at any scanned checkpoint.

### S-4. QK active-vs-dead gap at final checkpoint (Finding B replication)

**Claim:** At step 143000, mean QK_PR of never-critical heads is lower than mean QK_PR of any of the three active classes by at least 15 units.

**Statistic:** gap = mean(QK_PR | active) − mean(QK_PR | never).

**Pass:** gap ≥ +15. (410M: +20.31, 160M: +17.12.)

**Fail:** gap < +10 — Finding B claimed universal cross-scale, would need revision.

## Ordering and pre-registered interpretation

The primary test is run first on arrival of 1.4B data. Secondary tests are run in the order S-1, S-2, S-3, S-4.

### Joint interpretation table

| primary | S-1 | S-2 | S-3 | S-4 | interpretation |
|---------|-----|-----|-----|-----|----------------|
| PASS | PASS | PASS | PASS | PASS | **All findings replicate at 1.4B — publishable at NeurIPS-level.** Early specialization signal in OV_PR at step 1000 is a cross-scale universal; the two-phase structure of ρ is real; QK discrimination is a universal architectural consequence of training. |
| PASS | PASS | PASS | FAIL | PASS | Primary claim (step-1000 OV_PR predictive) confirmed; two-phase sign flip may be small-model phenomenon. Paper reframes E as 160M/410M-specific; core claim about early specialization is still publishable. |
| PASS | FAIL | — | — | — | Primary signal present but absorbed by class label at 1.4B. Finding D reduces to Finding B-like discrimination. Publishable result but weaker claim. |
| PASS | — | FAIL | — | — | Step-0 correlation detected → lottery-ticket-like at 1.4B even though not at 410M/160M. Reinterpret as scale-dependent initialization effect. |
| FAIL | — | — | — | PASS | Step-1000 signal is scale-specific to ≤410M; only Finding B replicates. Publishable as "spectral discriminator for head activity is universal, heavy-tail predictor is scale-dependent." |
| FAIL | — | — | — | FAIL | All findings scale-specific. Major re-scope: paper becomes about phase transition between 410M and 1.4B — possibly interesting but different framing. |

## Compute budget on 1.4B

Estimated from 410M scaling, with 1.4B being ~3.4× the parameter count:

- Ablation sweep at each of 7 checkpoints: ~35 min A100 80GB per checkpoint → ~4 h total. 144 heads × 7 checkpoints = 1,008 ablations, plus the 25 eval batches per ablation.
- Spectral invariants loading: 8 checkpoint loads × ~5 min each on A100 = ~40 min if model already on device (zero GPU cost for the SVD itself, CPU only).
- Step 0 baseline: ablation sweep not needed (baseline |Δ| is known per-head). Spectral load + SVD only.

**Total: ~5 h A100 80GB.** Budget: ~$8-10 on RunPod, ~$12-15 on Lambda.

## Non-pre-registered exploratory analyses (flagged as such in any write-up)

The following will be computed on 1.4B data as exploratory, not pre-registered, and reported with clear exploratory labels:

- 1.4B inversion phase pattern (is the initial negative QK_PR gap also present?)
- 1.4B dense curve shape and location of sign flip
- 1.4B cross-correlation structure of OV_PR and QK_PR at each checkpoint
- Robustness of results to classification threshold and to head sampling choice

## Deliverables required at 1.4B completion

- CSV: per-head per-checkpoint table (layer, head, OV_PR, QK_PR, OV_entropy, QK_entropy, |Δ|).
- JSON: per-test pass/fail/partial verdict with exact rho values.
- Report: one-page integrating pass/fail/partial outcomes with the joint-interpretation table above.

## What this document does NOT pre-register

- No prediction about Hessian diagonal or any candidate that was not computed on 410M/160M.
- No prediction about pruning behavior (that is a separate GPU experiment, Block 2 of the plan).
- No prediction about inverse meta-heads (that is Tier 2 T2.3, separate track).

---

*Locked 2026-04-24 before Pythia 1.4B data exists. Any deviation from this document in analysis must be flagged as post-hoc.*
