# ArXiv / NeurIPS Submission Checklist

Living document. Accumulate items as they arise during writing.

## Supplementary tables to include

- [x] **Table S1** — Top-10 most critical heads at step 143,000 (L, H, |Δ|, ratio to top-1). GENERATED: `paper/tables/tableS1_top10_heads.md`. Closes reviewer challenges on ratio claims in Sec 3.5.

## Supplementary figures to generate

- [ ] **Fig S1** — Lorenz curves across 8 checkpoints (DONE, saved to `paper/figures/figS1_lorenz_curves.pdf`)
- [ ] **Fig S2** — Differentiation count robustness to threshold: reproduce per-checkpoint counts at threshold values $\{10^{-4}, 5\times10^{-4}, 10^{-3}\}$
- [ ] **Fig S3** — *Central supplementary figure.* Establishes the regime classifications of Fig. 6 through sensitivity analysis. Three panels: (a) head sensitivity (df stable: 2.34 → 2.34 → 2.33 as top-1, top-3, top-5 extremes removed); (b) layer sensitivity (df drifts dramatically: 0.85 → 1.74 → ∞ as L0 and L5 removed); (c) overlay of empirical distributions before and after trimming, showing population-level heavy tail for heads vs outlier-plus-base for layers. This figure is not just "diagnostics" — it establishes the claim that the three granularities are qualitatively distinct regimes.
- [ ] **Fig S5** — Class count sensitivity: 4.2% born-critical / 38.9% emergent / etc. at threshold values in $[10^{-4}, 10^{-3}]$; shift ±2 percentage points

## Methods / Statistics to document

- [ ] Threshold $5\times10^{-4}$ justification paragraph (in Methods Section 2.5)
- [ ] Sentence that class counts shift by ±2pp across threshold range, qualitative conclusions preserved
- [ ] $N$-head ablation caveat in Sec 3.1 and Limitations
- [ ] Sensitivity check for layer df=0.85 in Sec 3.4 prose (DONE, integrated in revised draft)
- [ ] Sensitivity check for head df=2.34 stability (ADD to Sec 3.4 supplementary)

## Discussion content

- [ ] "Gifts for other communities" framing — three explicit follow-up directions:
  - L8H9 phase transition (4k→8k) → mech interp community (activation patching)
  - Born-critical heads (4.2% class) → lottery ticket / pruning community (what structural feature of init predicts criticality?)
  - Never-critical heads (34% class) → functional redundancy research (truly null or parallel-redundant?)
- [ ] "Upper bracket" honesty — acknowledge that layer-granularity as upper bracket of universality regime is empirically suggestive, not firmly established
- [ ] "Functional differentiation is simultaneously a distributional pattern and a nameable event" — good closer line for Discussion, pulled from 3.5

## Narrative / framing

- [ ] Explicit "iterable scale hierarchy" definition in Intro or start of 3.4 (DONE in 3.4 prose)
- [ ] Refinement of Paper 1 formalized as "substrate-independence conditional on discrete modularity"
- [ ] Conservation conjecture (not law) framing consistent throughout
- [ ] Single-model scope in abstract (DONE)
- [ ] In Discussion: acknowledge that "upper bracket" of universality regime (very-coarse outlier-dominated) is empirically suggestive, not firmly established; would require measurements at intermediate granularities (multi-head circuits, single-MLP vs attention vs full-block) to solidify. Closes reviewer 2 objection to "bracketed on both sides" claim in Sec 3.4.

## Infrastructure

- [ ] GitHub repo `functional-differentiation-dfe` created
- [ ] Raw CSV `all_ablations.csv` pushed
- [ ] Colab notebook `main_pilot_colab.ipynb` pushed with reproduction instructions
- [ ] README.md with quick-start
- [ ] License (MIT or CC-BY for data)

## Acknowledgements / disclosures

- [ ] "Analysis conducted with extensive use of Claude (Anthropic) as thinking and implementation partner"
- [ ] Hardware: Google Colab Pro A100, runtime 82 min, total cost ≈ $4
- [ ] Data source: Pythia 410M-deduped (EleutherAI), wikitext-103 (HuggingFace)

## ArXiv submission details

- [ ] Primary category: cs.LG
- [ ] Secondary: stat.ML, q-bio.NC
- [ ] License: CC-BY 4.0
- [ ] Abstract formatted per arXiv rules (no LaTeX macros, specific character limits)

## Cross-reference hygiene

- [ ] All `\ref{}` and `\citep{}` resolve
- [ ] All figures labeled and referenced in text
- [ ] Forward references consistent with final section numbering

## Reviewer 2 stress test — pre-emptive defense

**The single strongest plausible attack:** "Your central claim (dual differentiation) is framed as a conservation conjecture from n=8 checkpoints in one model. You defend against 'trivial explanation' via simulation, but the simulation is itself parametric — you take Student's t fit parameters as ground truth. Without replication across models, scales, or architectures, this is at best suggestive, and the paper's claims are disproportionate to the evidence."

**Prepared response (for response letter if attack appears):**

> We address this directly. The dual differentiation conjecture is framed throughout as a conjecture — not an established principle — with three explicitly falsifiable predictions (healthy training: Gini ≈ stable; collapsed: Gini → 1; undertrained: Gini → 0) and a clear negative-result criterion.
>
> The Gini triviality simulation in Fig. S4 is narrower in scope than the reviewer suggests. It does not attempt to prove the conjecture. It rules out one specific class of trivial explanation — that invariance is a mathematical consequence of Student's t fit parameters alone — by showing the empirical trajectory diverges from the fit-predicted trajectory in direction, magnitude, and cross-checkpoint correlation (r = −0.04). This test is parametric in its null hypothesis (Student's t alone) precisely because that is the trivial-explanation null we want to falsify.
>
> Replication across models, scales, and architectures is the essential next step. Section 4.6 flags this as the natural follow-up, and the falsifiable framing in Section 3.2 makes clear that a failed replication would constrain rather than refute the conjecture.
>
> We believe the paper's contributions are not disproportionate to the evidence:
> - (a) the first measurement of DFE along a full training trajectory with identity-tracked components is a methodological contribution independent of the conjecture's ultimate generality;
> - (b) the observed four-way emergence classification (Sec. 3.1) and the three-regime perturbation hierarchy (Sec. 3.4) are empirical findings that stand on their own;
> - (c) the conservation-conjecture framing is offered explicitly as a hypothesis for others to test, not as a claim the present data establishes.
>
> We thank the reviewer for the pressure to clarify; we have revised [Section X] to make the scope of each claim more explicit.

**Where to pre-emptively strengthen the paper against this attack:**
- [ ] Add one sentence at the end of 4.6 making the above scope-limitation fully explicit
- [ ] Consider adding abstract language that foregrounds "we measure" over "we show" for the conjecture specifically
- [ ] Ensure Section 3.2 says "conjecture" not "finding" or "principle" at every reference

## Final pre-submission checks

- [ ] Read entire paper in one sitting, note any awkward transitions
- [ ] Check all numerical claims against `all_ablations.csv` directly (no stale numbers from early drafts)
- [ ] Verify all four main findings from abstract appear verbatim in results
- [ ] Limitations section covers: single model, single task, 144-head sampling, $N$-head caveat, threshold sensitivity, n=24 for layers
