# Phase 1a findings — locked framing

**Do not retrospectively reformulate.** The two findings below are distinct phenomena and must be reported separately in any write-up.

---

## Finding A — pre-registered primary test result (null)

OV circuit participation ratio of Pythia 410M head weights at `step143000` does **not** predict per-head `|Δ|` (ablation effect on wikitext-103 validation loss at the same checkpoint).

- Spearman ρ(OV_PR, |Δ|) = −0.126, 95 % CI [−0.30, +0.06], p = 0.13.
- Direction matches pre-registered prediction (negative), magnitude does not (threshold was |ρ| > 0.4 for medium).
- Random N(0,1) control at n = 144 gave ρ = +0.113 — same noise floor, opposite sign.

Interpretation: the mechanistic hypothesis "low-rank (concentrated) OV computation generates heavy-tailed ablation effects" is not supported at this checkpoint.

## Finding B — secondary result (significant, opposite direction, different phenomenon)

QK circuit participation ratio and QK spectral entropy show significant positive association with `log |Δ|`, but this association is driven entirely by between-class discrimination (never-critical vs the three active classes), not by within-active-class gradation of tail weight.

- Between-class: mean QK_PR is 20 for never-critical heads, 38–46 for any of {growing, emergent, born}.
- Within active classes: QK_PR Spearman ρ with `log |Δ|` is 0.09–0.26 (all p > 0.1) — not significant.
- In the absorb-the-label regression, QK_PR and QK_entropy reduce the emergent-dummy coefficient by 14 % and 17 % respectively; both remain significant in the joint model (p = 0.001, p = 8 × 10⁻⁵).

**This is an active-vs-dead discriminator, not a heavy-tail predictor.** The two claims are separate.

## Language to use in any paper/write-up

The following four sentences are canonical and should be used verbatim (or equivalent) whenever these results appear:

> "A pre-registered primary test — OV participation ratio as predictor of per-head tail contribution — yielded a null result (ρ = −0.126, p = 0.13, direction consistent with prediction, magnitude at the noise floor). Secondary exploratory invariants on the QK circuit showed significant positive association with head activity (ρ ≈ +0.39, p < 10⁻⁶, direction opposite to the pre-registered heavy-tail hypothesis). The QK effect is driven entirely by discrimination between never-critical heads and the three active classes, not by gradation within active classes. The emergent-critical label is therefore not reducible to any of the four spectral properties tested at the final checkpoint; heavy-tail generation and active-vs-dead status are empirically distinct phenomena in Pythia 410M."

## What this does NOT say

- It does **not** say that QK_PR is a heavy-tail predictor. It is not.
- It does **not** say that spectral concentration has been ruled out as a universal mechanism; only at step 143000 on 410M for OV and QK measurements as defined.
- It does **not** fold the active-vs-dead finding into the main narrative as a "predictor" — that is a separate result deserving its own treatment.

## What remains testable (Phase 1b and later)

- Dynamic head properties (candidates 2–6) might predict heavy tails where static weight properties fail.
- Scale-dependent behaviour of QK discrimination across 160M / 410M / 1.4B (Phase 3 C).
- Training-stage-dependent behaviour of QK discrimination (Phase 3 B, planned now).

---

*Locked 2026-04-24, before Phase 3 B/C computation.*
