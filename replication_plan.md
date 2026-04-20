# Replication and Strengthening Plan: Paper 2 + Paper 3

**Status:** locked 2026-04-20
**Scope:** concrete experiments, compute estimates, expected outcomes, reviewer-objection mapping

The plan is organized into three tiers by urgency and impact on peer review:

- **Tier 1** — must-do before Paper 2 arXiv submission (days, not weeks)
- **Tier 2** — strongly recommended before peer-reviewed venue submission (weeks)
- **Tier 3** — programmatic strengthening for Paper 3 and unified story (months)

Each entry lists: reviewer objection addressed, experimental protocol, compute cost, expected outcome, integration point in paper, and priority.

---

## Tier 1 — Before Paper 2 arXiv submission

### T1.1 — Second seed replication on Pythia 410M

- **Reviewer objection addressed:** "Single ablation sweep; numbers may be artifacts of specific runs."
- **Protocol:** rerun main pilot (1,584 ablations) with alternate Type A PRNG seeds (base 12345 instead of 42000). Head and layer ablations are deterministic; only Type A needs reseeding. Also evaluate on a different batch of validation tokens (same wikitext-103 source, different 25-batch slice).
- **Cost:** ~1 h A100, $2–3.
- **Expected outcome:** numbers shift by <5 %, qualitative findings stable. Shift >10 % = investigate.
- **Integration:** appendix table "Replication with alternate seeds and eval batches." One sentence in Methods.
- **Priority:** **critical.** Not doing this is reviewer-2 low-hanging fruit.

### T1.2 — L8H9 sanity check on separate eval sets

- **Reviewer objection addressed:** "L8H9 finding is wikitext-specific, not real."
- **Protocol:** at step 143 000 only, re-measure L8H9 ablation effect on 5 eval sets: wikitext-103 (current), Pile validation subset, C4 subset, OpenWebText subset, BookCorpus subset. 100 batches each.
- **Cost:** ~30 min A100, $1.
- **Expected outcome:** |Δ| consistent in range 0.1–0.2 across datasets. If disappears on one — important to know before submission.
- **Integration:** one sentence in Section 3.5. Supplementary table.
- **Priority:** **high.** Cheap, strengthens central case study.

### T1.3 — Bootstrap iterations increased for β CI

- **Reviewer objection addressed:** "CIs based on 2,000 resamples may be unstable."
- **Protocol:** on existing data, recompute β bootstrap CI with 10,000 resamples (from 2,000). Zero model compute.
- **Cost:** minutes CPU.
- **Expected outcome:** CIs tighten by 5–10 %. Point estimates unchanged.
- **Integration:** update numbers in Methods and Fig 4.
- **Priority:** medium-high. Trivial cost, closes potential objection.

### T1.4 — Save/restore drift check on longer sequence

- **Reviewer objection addressed:** "Drift monitor every 30 ablations; what if drift accumulates between checks?"
- **Protocol:** run 500 consecutive ablation-restore cycles on one head, measure drift each cycle. Verify <10⁻⁶ throughout.
- **Cost:** ~20 min A100, $1.
- **Expected outcome:** drift stays within float32 precision, no accumulation.
- **Integration:** one sentence in Methods 2.4.
- **Priority:** medium. Low effort, closes specific objection.

**Tier 1 totals:** ~2–3 h A100, ~$7, elapsed time < 1 day of focused work. Required before arXiv v1.

---

## Tier 2 — Before peer-reviewed venue submission

### T2.1 — Scaling replication on Pythia 160M and 1.4B

- **Reviewer objection addressed:** "Single model size; findings may be Pythia 410M specific."
- **Protocol:**
  - *Pythia 160M-deduped:* 6 layers × 12 heads = 72 heads total, ablate all exhaustively; 6 layer ablations; 30 Type A; 8 checkpoints. ~850 ablations total.
  - *Pythia 1.4B-deduped:* 24 layers × 16 heads = 384 heads; sample 144 systematically (same as 410M); 24 layer ablations; 30 Type A; 8 checkpoints. ~1,584 ablations total.
- **Cost:** 160M ~40 min A100 ($2); 1.4B ~4 h A100 80GB ($8). **Total ~$10, ~5 h compute.**
- **Expected outcome:** qualitative findings replicate (dual differentiation, emergent dominance, iterable hierarchy). Specific numerical values differ 10–30 %. L8H9 equivalent may or may not exist at other sizes — informative either way.
- **Integration:** new supplementary figures S5–S7 (analogs of main Fig 1–4 for other sizes). Main text: one paragraph in Section 4 "Replication across model sizes."
- **Priority:** **critical for venue submission.** Required for NeurIPS/ICLR; arXiv can go without.

### T2.2 — Cross-dataset validation

- **Reviewer objection addressed:** "wikitext-103 specific; what about Pile validation (training distribution)?"
- **Protocol:** at step 143 000 on Pythia 410M, rerun full 1,584-ablation sweep with Pile validation split as eval.
- **Cost:** ~45 min A100, $3.
- **Expected outcome:** qualitative structure same. Specific Δ values may shift because Pile is closer to training distribution. Key question: does dual differentiation signature (count up, effective N down, Gini stable) replicate?
- **Integration:** supplementary section "Eval dataset sensitivity" with figures analogous to main text.
- **Priority:** **high** for venue submission.

### T2.3 — Inverse meta-heads analysis for Paper 3

- **Reviewer objection addressed:** "20 inverse meta-heads observed but not discussed — why?"
- **Protocol:** on existing Paper 3 data, analyze 20 inverse meta-heads:
  1. Paper 2 class distribution (concentrated where?)
  2. Layer distribution
  3. Disagreement-pattern analysis: on which questions does ablation help?
- **Test specific hypothesis** "inverse meta-heads are de-biasing components": ablate them and measure whether answer variance across sampling seeds *decreases* (pattern-locking). 10 repeated generations × per head, same question.
- **Cost:** ~15 min A100, $1.
- **Expected outcome:** either confirm de-biasing hypothesis (add to paper as independent finding) or not confirm (mention briefly as open question).
- **Integration:** new subsection in Paper 3 (~1 page). Concrete additional finding.
- **Priority:** **high.** Major addition to Paper 3.

### T2.4 — Self-modeling replication on Pythia 1.4B

- **Reviewer objection addressed:** "Paper 3 on single model; same single-model critique as Paper 2."
- **Protocol:**
  1. Build question bank for Pythia 1.4B (may differ from 410M because model makes different mistakes).
  2. Decisive control with Pythia 6.9B as "other model."
  3. If signal confirmed, ablation sweep on 144 fixed heads.
- **Cost:** ~3 h A100 80GB, $15.
- **Expected outcome:** signal either replicates (meta-heads in 1.4B) or not (scope limited). Null result still valuable.
- **Priority:** **high** for Paper 3 venue submission.

### T2.5 — Formal power analysis

- **Reviewer objection addressed:** "n=144 heads, n=29 questions; what's the statistical power?"
- **Protocol:** compute formal power analysis for main claims.
  - Paper 2 dual differentiation: effect size detectable at α = 0.05 with current n?
  - Paper 3 p < 10⁻⁸: power curve.
- **Cost:** analytical, few hours of work.
- **Integration:** appendix "Statistical power analysis" showing tests had adequate power.
- **Priority:** medium. Good practice, defensive.

**Tier 2 totals:** ~15–20 h A100, ~$50, elapsed time 2–3 weeks. Required before NeurIPS/ICLR.

---

## Tier 3 — Programmatic strengthening

### T3.1 — Cross-architecture validation

- **Protocol:** apply methodology to one non-Pythia architecture:
  - OLMo-1B (open, checkpoints available, different training data); or
  - Qwen-1.5B base (different architecture details); or
  - GPT-Neo-1.3B (older, different training).
  Repeat main pilot + self-modeling pre-flight + (if passed) ablation sweep.
- **Cost:** ~10–15 h A100, $30–50.
- **Priority:** medium-term. Makes claim truly substrate-independent.

### T3.2 — Instruction-tuned comparison

- **Protocol:** compare same-size base vs instruction-tuned: OLMo-1B vs OLMo-1B-Instruct, or similar pair.
- **Specific question:** do meta-heads' count/strength differ systematically? RLHF tunes on preference signals that require self-modeling.
- **Cost:** ~5 h A100 80GB per pair, $20–30.
- **Priority:** medium-term. Scientifically important follow-up; possible standalone short paper.

### T3.3 — N-head ablation protocol

- **Protocol:** on Pythia 410M step 143 000, ablate random triples of 3 never-critical heads simultaneously. Test whether joint effect > sum of individual effects (parallel redundancy detection). Sample 50 random triples from 49 never-critical heads.
- **Cost:** ~1 h A100, $3.
- **Priority:** medium. Closes "parallel redundancy vs truly null" question flagged in Paper 2 Section 3.1 and 4.5.

### T3.4 — L8H9 attention-pattern analysis

- **Protocol:** extract L8H9 attention patterns at checkpoints 4 000, 8 000, 16 000, 143 000. Visualize what tokens it attends to. Induction-head-style probe (query-key matching patterns).
- **Cost:** 2–3 h manual + agent analysis; minimal model compute.
- **Expected outcome:** hint at specific function. Likely candidate: induction head or specific syntactic role.
- **Priority:** medium. Self-contains the L8H9 case study, gives concrete target for mech-interp community.

**Tier 3 totals:** ~30 h A100, ~$100, elapsed time 2–4 months. Follow-up work between Paper 2 and Paper 3 submissions, or post-submission as Paper 3.1.

---

## Execution order (recommended)

1. **Now → arXiv v1 (Paper 2):** T1.1 → T1.3 → T1.4 → T1.2 → submit.
2. **arXiv v1 → venue submission (Paper 2):** T2.1 → T2.2 → T2.5 → revise → submit.
3. **In parallel with Paper 2 venue track:** T2.3 → draft Paper 3 with inverse-meta section.
4. **After Paper 2 venue submission:** T2.4 → finalize Paper 3 → arXiv Paper 3.
5. **Between Paper 3 arXiv and venue:** T3.3 → T3.4 → include in venue version.
6. **Post-Paper 3:** T3.1 → T3.2 as Paper 3.1 standalone.

## Total budget

| Tier | Compute | Cash | Elapsed time |
|------|---------|------|--------------|
| 1 | ~2–3 h A100 | ~$7 | <1 day |
| 2 | ~15–20 h A100 | ~$50 | 2–3 weeks |
| 3 | ~30 h A100 | ~$100 | 2–4 months |
| **Total** | **~50 h A100** | **~$160** | **~4 months to full program** |

This is a small research budget for two papers. Bottleneck is elapsed time, not money.

## Decision gates

- **T1.1 shows numbers shift >10 %:** stop; investigate before submission. Possible causes: single-seed variance is high, or there is a bug in original run.
- **T2.1 scaling does not replicate qualitatively:** findings become Pythia-410M-specific. Paper 2 becomes weaker; reframe abstract to acknowledge scope.
- **T2.3 inverse-meta de-biasing hypothesis fails:** drop from Paper 3 main text, relegate to "Open questions."
- **T2.4 self-modeling does not replicate on 1.4B:** Paper 3 scope is explicitly Pythia 410M base. Paper still publishable but narrower. Needs explicit limitation statement.
- **T3.1 cross-architecture fails:** substrate-independence claim downgraded. Paper 2 becomes "universality in one architecture family" rather than substrate-independent generalization.

## Integration with Paper 2 LaTeX

- Tier 1 results → update existing text (not new sections), Section 2 Methods additions, Table S2–S3 in appendix.
- Tier 2 results → new subsection in Section 4 "Replication across model sizes and datasets," new Figs S5–S7, appendix power-analysis section.
- Tier 3 results → cross-referenced in Discussion as "follow-up work in preparation," or become standalone papers.

## Integration with Paper 3 LaTeX (to be drafted)

- T2.3 inverse meta-heads → primary 3rd finding in Results.
- T2.4 1.4B replication → critical result; if passes, cross-size finding in Results.
- T3.2 instruction-tuned comparison → fourth paper in sequence (Paper 3.1).

---

*End of plan. Locked 2026-04-20.*
