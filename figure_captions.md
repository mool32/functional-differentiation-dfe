# Figure captions — draft for review

## Figure 1 — Functional differentiation of attention heads (PRIMARY)

**Each line represents one attention head tracked across 8 training checkpoints.** We ablate the same 144 fixed heads (heads {0, 3, 6, 9, 12, 15} at each of 24 layers) at steps 512, 1 000, 2 000, 4 000, 8 000, 16 000, 64 000, and 143 000 of Pythia 410M training. The y-axis shows |Δ|, the fractional change in validation loss caused by ablating that head. Heads are classified by their trajectory: **born-critical** (4.2 %, n = 6; important at step 512 and step 143 000), **emergent** (38.9 %, n = 56; from noise floor to |Δ| > 5×10⁻⁴ by end of training), **growing** (22.9 %, n = 33; small initial effect amplified during training), and **never-critical** (34.0 %, n = 49; below threshold throughout). The dominance of the emergent class over the born-critical class demonstrates that most function is not latent at initialization but develops during training.

---

## Figure 2 — Dual structure of functional differentiation

Three complementary metrics of head importance distribution across training, computed from the 144-head sample at each checkpoint. **(a)** Gini coefficient of |Δ| — inequality of head importance. Approximately invariant across training (Spearman ρ = +0.12; 95 % CI overlap at all checkpoints). **(b)** Effective number of important heads, 2^H where H is the Shannon entropy of normalized |Δ| — biological analog of effective species diversity. Monotonically decreasing (ρ = −0.88), indicating concentration of responsibility in fewer heads. **(c)** Count of heads with |Δ| > 5×10⁻⁴ — intuitive threshold-based measure of functional differentiation. Monotonically increasing (ρ = +1.00), indicating broadening of functional base. The combination — more heads become critical (c), yet critical importance concentrates in the top few (b), with inequality constant (a) — defines a **dual differentiation** regime: training simultaneously broadens and concentrates the function landscape.

---

## Figure 3 — Emergence of DFE shape during training

Distribution of fitness effects for 144 head ablations at four representative checkpoints. Histograms (bars), fitted Student's t (black), fitted Normal (red dashed). **At step 512**, the distribution is dominated by a narrow peak at zero with two outliers; the Student's t AIC advantage (ΔAIC = +140) reflects outlier capture, not distribution shape. **By step 2 000**, a continuous heavy-tailed distribution has emerged. **At step 16 000 and step 143 000**, the Student's t form with strong negative skew matches the universal DFE shape reported across biological and engineered adaptive systems [Paper 1]. The gamma shape β of the deleterious tail evolves from β = 1.77 [95 % CI: 1.23, 2.84] at step 512 into the biological range (E. coli β ≈ 0.5, yeast β ≈ 0.7) by step 1 000 and remains there. Titles report n_diff (heads above threshold 5×10⁻⁴), Gini, skewness, and kurtosis.

---

## Figure 4 — Quantitative distribution fit across training

**(a)** ΔAIC of Student's t and Laplace distributions relative to Normal, at each checkpoint. Positive values indicate better fit than Normal. Student's t decisively wins at every checkpoint (ΔAIC ≫ 10 throughout), confirming non-Gaussian shape. **(b)** Gamma shape parameter β of the deleterious tail (|Δ| > 10⁻⁴ to exclude numerical noise), with 2 000-sample bootstrap 95 % CI. β = 1.77 at step 512 is in the light-tailed regime; the drop to β ≈ 0.8 at step 1 000 is significant (non-overlapping CIs). Subsequent evolution to β ≈ 0.6 at step 143 000 matches the biological range marked with dashed lines.

---

## Figure 5 — Layer-level ablation landscape

Heatmap of |Δ| for all 24 layer ablations across 8 checkpoints (log color scale). **Layer 0 is the dominant critical component at every checkpoint** — removing it grows in impact from |Δ| = 0.28 (step 512) to |Δ| = 2.76 (step 143 000). Layer 5 is second in importance, growing from 0.018 to 0.71; the L0/L5 ratio shrinks from 15.9 to 3.9, indicating L5 catches up to L0 in relative criticality. Middle layers (L9–L15) remain the least critical throughout. The monotonic darkening at every layer confirms that total dependency on every discrete unit increases during training — a layer-level analog of the head-level differentiation.

---

## Figure 6 — Three DFE regimes across perturbation granularities

Ablation effect distributions at step 143,000 for three perturbation granularities: distributed parametric noise (α = 1·σ applied to all parameters of one block, n = 30), single attention head ablation (n = 144), single transformer layer ablation (n = 24). **(a)** Survival function P(|Δ| > x) on log-log axes. **(b)** Magnitude density on log |Δ| axis. Three qualitatively distinct regimes are observed, not points on a smooth heaviness gradient:

| Perturbation | Student's t df | ΔAIC vs Normal | Regime |
|---|---|---|---|
| Distributed noise | → ∞ | −2 | Gaussian (CLT limit) |
| Head ablation | 2.34 | +589 | **Heavy-tailed (population-level)** |
| Layer ablation | 0.85 | +82 | Outlier-plus-Gaussian-base |

Head-level ablation is the primary finding: a robustly heavy-tailed Student's t distribution matching biological DFE form (β ≈ 0.6, within the *E. coli*–yeast range from Paper 1). The heavy-tailedness is stable under outlier removal (df remains 2.34 after dropping the top-5 extreme observations), indicating a population-level property across many contributing heads. Layer-level heavy-tailedness at n = 24 is by contrast fragile: removing Layer 0 and Layer 5 collapses the fit to Gaussian, revealing that the apparent heavy tail at this granularity is driven by two structural outliers on an otherwise Gaussian base. Distributed noise obeys the CLT into Gaussian regardless of landscape structure. See Fig. S3 for the full sensitivity analysis establishing these regime classifications. **The universal heavy-tailed DFE form of Paper 1 applies at intermediate discrete granularity, bracketed on both sides: distributed perturbations Gaussianize via CLT, very-coarse discrete ablations become dominated by a few structural outliers.**

---

## Figure 7 — The L8H9 crystallization

Absolute effect of ablating Layer 8 Head 9 across training (orange, heavy), overlaid on trajectories of other born-critical and emergent heads (gray background). Between step 4 000 and step 8 000, L8H9's ablation effect jumps by more than 20 × (from 2×10⁻³ to 5.3×10⁻²) and continues growing to 1.54×10⁻¹ by step 143 000 — roughly 10 × the second-most-critical head. This single emergence dominates the kurtosis ≈ 127 signal observed at step 8 000+ (Fig 4a). The step-4 000-to-8 000 phase-transition-like emergence is a concrete target for mechanistic-interpretability follow-up: identifying the specific computation that crystallizes in L8H9 during this interval would link our component-level quantitative measurement to a circuit-level qualitative understanding.
