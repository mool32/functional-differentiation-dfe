# Phase 1 findings — locked framing

**Do not retrospectively reformulate.** The findings below are distinct phenomena and must be reported separately in any write-up. This document supersedes the earlier version but keeps every previously-locked claim; new material from the dense scan is added, earlier material is clarified, nothing is softened.

---

## Finding A — pre-registered primary test result (NULL)

OV circuit participation ratio of Pythia 410M head weights at `step143000` does **not** predict per-head `|Δ|` at that same checkpoint.

- Spearman ρ(OV_PR, |Δ|) = −0.126, 95 % CI [−0.30, +0.06], p = 0.13.
- Direction matches pre-registered prediction (negative), magnitude does not (pre-registered threshold was |ρ| > 0.4 for medium).
- Random N(0,1) control at n = 144 gave ρ = +0.113 — same noise floor, opposite sign.

**The pre-registered hypothesis as stated failed.** Any later mention of OV_PR predictive power at other checkpoints is a *new* exploratory finding, not a rescue of the original pre-registration. The step 143000 null is not overturned by observing stronger correlations at other training steps.

## Finding B — active-vs-dead QK discrimination (significant, different phenomenon)

QK participation ratio discriminates never-critical heads (mean QK_PR = 20 in 410M, 16 in 160M) from active-critical heads (mean QK_PR = 38–46 in 410M, 26–36 in 160M). Within any active class, QK_PR does not grade tail weight.

- Cross-scale: gap active-minus-dead QK_PR is +20.31 (410M) and +17.12 (160M). Universal direction, similar magnitude.
- Training-driven: at step 0 (actual Pythia random init) on 160M, gap = +0.04. The discrimination emerges during training.
- Non-monotonic buildup at smaller scale: 160M oscillates (gap goes −2.86 → +3.56 → +4.15 → −0.49 → +17.12 across training). 410M is cleaner but also starts inverted (−1.06 at step 1000) before consolidating.

**This is an active-vs-dead discriminator, not a heavy-tail predictor. The two claims remain separate.**

Outstanding: GPU-based prune-and-evaluate test required to validate discrimination via behavior rather than single-head `|Δ|` correlation.

## Finding C — scale-dependent category breakdown

In 160M, the categorical `emergent` label does not predict `|Δ|` (Spearman ρ = +0.075, p = 0.38). In 410M it does (ρ = +0.70, p = 10⁻²²). Meanwhile QK_PR remains predictive in both (+0.552 and +0.390). Spectral invariant is scale-invariant where the categorical label is scale-dependent.

- In 160M the `emergent` class has QK_PR = 25.9, lower than `born` (36.4) and `growing` (34.5). In 410M `emergent` sits between `born` and `growing`. QK-space reproduces the inversion pattern found in the AIC-structure comparison.

Scope: two scales. Claim status is suggestive pending 1.4B replication.

## Finding D — training-stage-dependent OV_PR correlation (EXPLORATORY)

**This is not a rescue of Finding A.** Finding A is the pre-registered test at step 143000 and it is null. Finding D is a separate exploratory observation obtained after the pre-registration had resolved.

From the dense checkpoint scan, ρ(OV_PR, |Δ|) at the same checkpoint varies across training in both models:

| step | 410M  | 160M |
|------|-------|------|
|  512 | −0.36 | −0.42 |
| 1000 | **−0.56** | **−0.42** |
| 2000 | −0.45 | −0.16 |
| 4000 | −0.36 | +0.07 |
| 8000 | −0.21 | +0.01 |
|16000 | −0.17 | −0.04 |
|64000 | −0.32 | −0.22 |
|143000| −0.13 | −0.17 |

The magnitude peaks at step 1000 in both models with correct pre-registered direction (negative) and |ρ| between 0.42 and 0.56. This is an exploratory post-pre-registration observation; it does not convert Finding A from null to pass.

**Three candidate mechanistic interpretations, all testable:**

1. **Lottery ticket.** Heads that will become critical have specific OV structure at initialization; step 1000 measurement catches residual of init. Ruled in by: high ρ between OV_PR at step 0 (actual init) and |Δ| at step 143000.
2. **Early specialization.** Heads quick-commit to roles during steps ~0–2000; commitment imprints in OV_PR transiently. Ruled in by: step 0 shows no correlation, step 1000 shows strong, later checkpoints fade.
3. **Noise window.** Early chaotic dynamics produce random correlations. Ruled in by: variation across different random inits gives the same |ρ| magnitude.

## Finding E — two-phase sign-flip of ρ during training (EXPLORATORY, TENTATIVE)

Both models exhibit ρ(QK_PR, |Δ|) initially negative (−0.64 at 410M step 1000, −0.55 at 160M step 1000), flipping to positive by step 4000 (410M +0.25, 160M +0.13). Replicates across two scales. Interpretation deferred until 1.4B replication.

**Scope limit:** two models, no mechanistic story yet, no third-scale confirmation. Status is *observation of a replicated pattern*, not *finding with publishable claim*.

---

## Canonical language for any write-up

> "A pre-registered primary test — OV participation ratio as predictor of per-head tail contribution at the final checkpoint — yielded a null result (ρ = −0.126, p = 0.13, direction consistent with prediction, magnitude at the noise floor). Exploratory post-pre-registration analysis revealed that the correlation between OV participation ratio and `|Δ|`, measured at the same checkpoint, varies substantially across training and peaks at approximately step 1000 in both Pythia 410M and 160M (|ρ| ≈ 0.4–0.6). This is a separate finding from the pre-registered hypothesis. A secondary invariant on the QK circuit discriminates never-critical heads from active heads (gap ≈ +20 QK_PR units in 410M, +17 in 160M); within active classes it does not grade tail weight. The mechanistic cause of the step-1000 OV signal and the direction flip during training remain undetermined pending a 1.4B replication and an init-structure control."

## What this does NOT say

- It does **not** say Finding A was validated. It was null. Dense scan produced a new finding, not a rescue.
- It does **not** say that Finding E is robust enough for submission. Two observations on two scales is replication of pattern, not publishable claim.
- It does **not** fold B, C, D, E into one narrative. Each has separate evidence base and separate exposure to 1.4B validation.

---

*Locked 2026-04-24, before any 1.4B work.*
