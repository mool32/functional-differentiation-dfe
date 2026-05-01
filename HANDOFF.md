# DFE Research — Session Handoff

Read this FIRST when resuming.

## Current state (2026-04-20 end of day)

**GitHub:** https://github.com/mool32/functional-differentiation-dfe
**Latest commit:** `b69cfbf` (Tier 1 integrated into Paper 2)
**Author:** Theodor Spiro, tspiro@vaika.org

## Two papers in flight

### Paper 2: Functional Differentiation Generates Universal Fitness-Effect Distributions in Neural Networks
- **Status:** Tier 1 replication complete, integrated into `paper/main.tex`. Submission-ready pending Overleaf compile-read.
- **Location:** `paper/main.tex` (520 lines), `paper/references.bib`, figures in `paper/figures/`
- **Four findings:** (1) emergent dominates born-critical 10:1, (2) dual differentiation conjecture (Gini invariant under count↑ + eff_N↓), (3) DFE shape emerges first 1-1.5% training, (4) iterable scale hierarchy (Gaussian/t/outlier-plus-base)
- **Tier 1 verdicts all PASS or interpretable-PASS:**
  - T1.1 seed replication: std ratios 0.93-1.05, max dev 6.5%
  - T1.2 L8H9 cross-dataset: 0.154/0.065/0.070 on wikitext/C4/OWT (honestly reported, no speculative interpretation)
  - T1.3 bootstrap 10k: max median β drift 0.007
  - T1.4 drift 500 cycles: zero drift, 500/500 hash preserved

### Paper 3: Self-Specific Attention Heads: Localization of Procedural Self-Modeling in a Pretrained Language Model
- **Status:** outline locked in `paper3/abstract_and_outline.md`. Prose not started.
- **Empirical foundation complete:**
  - Pre-flight: split profile (C1 fail 0.55, C4 pass 0.70, C7 chance 0.58)
  - Decisive controls (Pythia 1.4B): 9/10 self-match ratio on disagreement, p=0.011
  - Main sweep: 144 heads, 29 meta-heads (all Δ_self > Δ_cross, p<10⁻⁸), 8 pure meta-heads
  - Pattern A: 4 emergent + 3 growing out of top 8 pure, 0 born-critical
  - Secondary: 20 inverse-meta heads, L8H9 dual-role
- **Data on GitHub:** `data/micropilot/question_bank.json`, `data/micropilot_summary.json` available via notebooks
- **5 open questions for Teo** at end of `paper3/abstract_and_outline.md`:
  1. Title approval
  2. "Procedural vs recognitional" framing sense-check
  3. Pattern A enrichment tone (small n per class, supportive-not-conclusive?)
  4. Related literature to cite/distinguish (meta-heads, self-specific heads precedents)
  5. Venue: NeurIPS 2026 (May, companion to Paper 2) vs ICLR 2027 (Sept, after Tier 2)

## Immediate next steps (when session resumes)

### For Paper 2 (parallel track)
1. **Compile-read** main.tex on Overleaf (upload `paper/` folder + `neurips_2024.sty`)
2. Fix any compile errors / broken refs
3. arXiv submission: cs.LG primary + stat.ML + q-bio.NC secondary
4. External reviewer outreach in parallel (Teo's network, 48-72h turnaround)

### For Paper 3 (main track)
1. **Teo answers 5 outline questions** in `paper3/abstract_and_outline.md`
2. **Generate 7 figures** from existing data (`data/micropilot/*`). Script should produce:
   - fig1 C1/C4/C7 baselines
   - fig2 decisive test disagreement subset
   - fig3 Δ_self × Δ_task scatter (CENTRAL, color by Paper 2 class)
   - fig4 Pattern A enrichment bars
   - fig5 Δ_self distribution with meta + inverse-meta tails
   - fig6 L8H9 dual-role quadrant
   - fig7 layer distribution of meta-heads
3. **Write Results prose** starting with 3.3 (meta-head localization) and 3.4 (Pattern A) — strongest two
4. Then Methods → Intro → Discussion (same order as Paper 2)
5. Bibliography: extend Paper 2 .bib with alignment/self-modeling citations (hubinger2024sleeper, ngo2023alignment, panickssery2024llm, laine2024memoranda — exact cite keys tbd)
6. Assemble main.tex paper 3 in `paper3/main.tex`

## Replication plan

`paper/replication_plan.md` — three tiers:
- **Tier 1:** DONE (integrated into Paper 2)
- **Tier 2:** pending (scaling 160M+1.4B, cross-dataset Pile, inverse-meta analysis for P3, 1.4B self-modeling rep for P3, power analysis)
- **Tier 3:** long-term (cross-architecture OLMo/Qwen, instruction-tuned comparison, N-head ablation, L8H9 attention patterns)

Total compute for full program: ~50h A100, ~$160.

## Key discipline learned (do not violate)

1. **Gating before scaling.** Every expensive experiment preceded by minimal pilot.
2. **Float32 + TF32 matmul.** Float16 noise floor destroys early-checkpoint signal.
3. **Mandatory Drive mount on Colab.** Local filesystem ephemeral on disconnect.
4. **SHA-256 bitwise save/restore verification.** Reviewer-proof reproducibility.
5. **Threshold-free primary metrics (Gini, Eff N).** Threshold choice is arbitrary.
6. **Bootstrap CIs everywhere.** No point estimates without quantified uncertainty.
7. **Explicit scope disclaimers.** "What we do not claim" pre-empts reviewer 1.
8. **Negative findings as content, not setback.** Type A → Gaussian (CLT), OLMo degenerate → Pythia 1.4B control, n=11 cache bug → n=29 rerun. Each failure was additional data.

## Author's working mode

- Russian-English mixed discourse
- Prefers concrete next action over open-ended brainstorming
- Strong methodological discipline; will catch speculative interpretation and push for tighter claims
- Reviews drafts in point-by-point technical mode (accept/reject/revise with reasons)
- Three-node workflow: Teo (conceptual framing) + Claude session (implementation) + spawn agent (long compute tasks)

## Files to read (in order) to resume fully

1. This file (HANDOFF.md)
2. `paper3/abstract_and_outline.md` (Paper 3 plan + 5 open questions)
3. `paper/replication_plan.md` (what's done, what's next)
4. `paper/submission_checklist.md` (Paper 2 status)
5. `paper/main.tex` (Paper 2 current state)
6. Memory: `~/.claude/projects/-Users-teo/memory/project_dfe_pilot.md`

## Do NOT redo

- Do not re-run Paper 2 main pilot — data in `data/all_ablations.csv`
- Do not rebuild question bank — locked at 29 questions in `data/micropilot/question_bank.json`
- Do not re-run micro-pilot sweep — results available, only prose/figure generation remains
- Do not re-verify Tier 1 — integrated and committed
