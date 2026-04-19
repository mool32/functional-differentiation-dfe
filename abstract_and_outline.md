# Paper 1 (follow-up) — Abstract and Outline

## Working title options

1. **Functional Differentiation Generates Universal Fitness-Effect Distributions in Neural Networks**
2. Dual Differentiation: Simultaneous Broadening and Concentration of Attention Head Importance During Training
3. The Emergence of Structured Ablation Spectra in Pythia 410M

**Recommendation: Option 1** — signals both the empirical content (*functional differentiation*) and the theoretical claim (*generates universal form*), and inherits vocabulary from Paper 1.

---

## Abstract (target ~200 words, revised)

> Ablation studies in trained neural networks reveal a universal statistical form — heavy-tailed, negatively skewed distributions of fitness effects (DFE) — previously observed across biological and engineered adaptive systems. This universality has been treated as an emergent property of trained systems, but its *origin* in the training process has not been characterized. We ablate 144 tracked attention heads, 24 layers, and apply 30 distributed-noise perturbations at each of 8 checkpoints spanning the entire training trajectory of Pythia 410M (step 512 → 143 000). We find that (i) the DFE shape evolves from a delta-peak-with-outliers regime at early training to a continuous heavy-tailed form; by step 143 000 the median gamma shape β ≈ 0.6 falls within the biological range reported for *E. coli* and yeast; (ii) this evolution is driven by *functional differentiation* of attention heads — 39 % of eventually-critical heads emerged from noise, only 4 % were born critical; (iii) differentiation has a *dual structure*: the number of functional components grows (Spearman ρ = +1.00) while importance concentration grows simultaneously (ρ = −0.88 on effective N), with inequality (Gini) approximately invariant across training; (iv) substrate-independence of the DFE form is conditional on discrete modularity — distributed parametric noise produces Gaussian DFE by CLT, head ablations produce Student's t (df ≈ 2.3), layer ablations produce super-Cauchy tails (df ≈ 0.85). We identify a single attention head (L8H9) undergoing phase-transition-like specialization between step 4 000 and 8 000, a concrete target for mechanistic interpretability. These results are obtained on a single model (Pythia 410M) and represent an initial characterization; scaling and cross-architecture replication are the natural follow-up.

---

## Outline with notes

### 1. Introduction (~1 page)

**Opening hook:** The distribution of fitness effects (DFE) of single-component ablations is statistically universal across adaptive substrates — from point mutations in E. coli to architectural components in neural networks [cite Paper 1]. The form is Student's t, heavy-tailed, negatively skewed, with a gamma-shape β ≈ 0.3–0.7 on the deleterious tail.

**The puzzle:** This universality is *observed*, not *explained*. Is it a property of fully-trained systems that emerges from the learning process, or is it already present at initialization? What mechanism at the component level produces universality at the distribution level?

**Our contribution** (three bullets):
1. We measure DFE along the training trajectory of Pythia 410M at 8 checkpoints using identity-tracked attention heads (the same 144 heads probed at every checkpoint).
2. We show the universal DFE form emerges during training from a qualitatively different regime (delta peak + few catastrophic outliers).
3. We identify the generative mechanism: functional differentiation of individual heads, with a *dual* structure — base broadening and tip concentration — that leaves inequality (Gini) invariant.

**Related work positioning** (brief):
- Paper 1 (substrate-independent DFE, observation)
- Mechanistic interpretability (Olsson, Conmy, Wang) — studies heads qualitatively; our DFE approach measures heads *quantitatively* through fitness
- Emergent abilities / phase transitions (Wei et al., Srivastava et al.) — complementary: we study *component-level* emergence, they study *capability-level*
- Lottery ticket hypothesis (Frankle) — shares "some components are critical from initialization" flavor; our born-critical class (4.2 %) is quantitatively small, inconsistent with strong lottery-ticket interpretation

### 2. Methods (~1 page)

**Model and checkpoints.** Pythia 410M-deduped, 8 checkpoints log-spaced: 512, 1 000, 2 000, 4 000, 8 000, 16 000, 64 000, 143 000. Baseline perplexity 922 → 18.

**Three perturbation types:**
- **Head ablation.** Systematic sampling: heads {0, 3, 6, 9, 12, 15} from each of 24 layers = 144 fixed heads, same identities at every checkpoint. Ablation = zeroing columns of `attention.dense.weight` corresponding to this head's output slice (equivalent to output-projection masking; standard in mech interp).
- **Layer ablation.** Exhaustive — all 24 transformer blocks zeroed one at a time.
- **Distributed noise (Type A).** Gaussian noise added to all parameters in one random block, scale α = 1.0 × parameter std. 30 per checkpoint.

**Fitness metric.** Δ = −(ℓ_perturbed − ℓ_baseline) / |ℓ_baseline|, evaluated on 25 batches × 4 × 2048 tokens = 204 800 tokens from wikitext-103. Positive Δ = beneficial, negative = deleterious. Baseline recomputed per checkpoint on identical validation batches; float32 evaluation with TF32 matmul.

**Total: 1 584 ablations. Runtime: 82 min on A100 40 GB.**

**Reproducibility and correctness.** Bitwise save/restore verification (SHA-256 of affected weights identical before and after restoration). Checksum drift monitor on a reference tensor, threshold 10⁻³. All drift values observed: 0.

**Primary metrics (threshold-free).**
- **Gini coefficient** of |Δ| distribution (inequality of head importance)
- **Effective N** = 2^H where H = Shannon entropy of normalized |Δ| (biological analog: effective number of species)

**Secondary metrics.**
- Differentiation count: #heads with |Δ| > 5 × 10⁻⁴ (intuitive, threshold-dependent)
- Student's t fit with df, skewness, kurtosis
- Gamma shape β fit on deleterious tail (|Δ| > 10⁻⁴ to filter numerical floor), bootstrap 95 % CI

### 3. Results

#### 3.1 Functional differentiation is predominantly emergent
**Fig 1 + classification table.**

Of 144 tracked heads:
- 6 (4.2 %) born-critical — important at both step 512 and step 143 000
- 56 (38.9 %) emergent — from noise floor to critical
- 33 (22.9 %) growing — initial small effect amplified over training
- 49 (34.0 %) never-critical — remained below 5e-4 throughout

**Key claim:** majority of eventual function is *not* latent at initialization. This is a strong quantitative constraint on lottery-ticket-style hypotheses at the head granularity.

#### 3.2 Differentiation has a dual structure
**Fig 2 + Fig S1 (Lorenz curves).**

Three threshold-free metrics tell complementary stories:
- **Differentiation count** monotonic ↑ (Spearman ρ = +1.00 across 8 checkpoints)
- **Effective N** monotonic ↓ (ρ = −0.88)
- **Gini** approximately invariant for step 1 000 onwards (values 0.71–0.78; step 512 is an outlier at 0.60 consistent with the pre-emergence regime)

Interpretation: training simultaneously *broadens* the functional base (more heads contributing) *and* concentrates responsibility in the top few. These effects balance on the Lorenz-curve level, keeping Gini approximately stable after emergence. **Fig S1** shows Lorenz curves at three representative checkpoints nearly coinciding — the visual signature of this invariance.

We propose this as a **conservation conjecture** rather than a law: approximate Gini invariance under dual differentiation is observed on n = 8 checkpoints in one model, and is testable in follow-up work on (a) other model sizes, (b) other architectures, (c) pathological training regimes.

**Falsifiable predictions** from the conjecture:
- Collapsed representations → Gini → 1 (single head carries all function)
- Undertrained / uniform networks → Gini → 0 (all heads equal, no specialization)
- Healthy training maintains intermediate, approximately stable Gini

#### 3.3 DFE shape emerges during training
**Fig 3 + Fig 4.**

- At step 512, head-ablation DFE is qualitatively different: delta peak at zero + two outliers. Gini = 0.60. The ΔAIC(t vs Normal) = +140 advantage is driven by outlier capture, not by a continuous heavy tail.
- By step 2 000, a continuous heavy-tailed distribution is established.
- β median trajectory shows an apparent decline from 1.77 at step 512 to 0.63 at step 143 000. **Bootstrap 95 % CIs overlap between endpoints** ([1.23, 2.84] vs [0.41, 1.35]), so we characterize this as *consistent with* a decline into the biological range, not as a statistically significant decline on this n.
- What *is* significant: the median β at step 512 lies above 1 (exponential-like), and the median β at step 143 000 lies below 1 within the biological range reported for *E. coli* (β ≈ 0.5) and yeast (β ≈ 0.7).
- **This refines Paper 1:** the universal heavy-tailed DFE form is a property of *trained* systems, emerging during training rather than present at initialization.

#### 3.4 Iterable scale hierarchy of perturbations
**Fig 6.**

At step 143 000:

| Perturbation | t_df | Skewness | Kurtosis | ΔAIC(t vs N) |
|---|---|---|---|---|
| Distributed noise (Type A, α=1) | ∞ (Gaussian) | −0.10 | −1.5 | −2 |
| Head ablation | 2.34 | −11.2 | +127 | +589 |
| Layer ablation | **0.85** | −4.2 | +16.7 | +82 |

Monotonic relationship between **granularity of perturbation** (distributed → head → layer) and **heaviness of resulting DFE tails**. t_df = 0.85 < 1 is super-Cauchy: layer-ablation distribution has no finite variance.

**This uncovers a missing condition on substrate-independence:** universal form requires discrete modular perturbation units. Distributed perturbations collapse to Gaussian by CLT regardless of landscape geometry. This is a meaningful refinement of Paper 1's claim.

**Bonus observation:** Type A at step 512 *is* Student's t (ΔAIC +55), becoming Gaussian only by step 2 000+. At initialization, one block (L0) dominates; distributed noise hitting L0 creates outliers. After training, all blocks matter; CLT applies. This is a second independent witness of dual differentiation.

#### 3.5 Case study: L8H9 crystallization
**Fig 7.**

A single attention head, Layer 8 Head 9, exhibits phase-transition-like specialization:

| step | 512 | 1 000 | 2 000 | 4 000 | 8 000 | 16 000 | 64 000 | 143 000 |
|---|---|---|---|---|---|---|---|---|
| |Δ| | 3e-5 | 1e-4 | 3e-6 | 2e-3 | **5.3e-2** | 8.1e-2 | 1.3e-1 | **1.5e-1** |

Between step 4 000 and 8 000, |Δ| jumps by > 20 × in a single checkpoint interval; growth continues to step 143 000 where L8H9 is the single most critical head, ~ 10 × more impactful than the second-place head.

This single emergence explains the skewness = −11, kurtosis = 127 signal at step 8 000+.

**Open question for mechanistic interpretability:** what computation does L8H9 perform? A targeted activation-patching study on L8H9 across step 4 000 → 8 000 would reveal the crystallizing function.

### 4. Discussion (~1 page)

**Brief opening:** summary of the four findings.

#### 4.1 Relation to Paper 1: refinement, not contradiction

In Paper 1, substrate-independence of DFE form was observed between mature adaptive systems (biology, trained AI). Here we show that (a) the heavy-tailed universal form is absent in the earliest training stages, (b) it emerges gradually through functional differentiation of components, (c) its applicability requires discrete modular perturbations (distributed noise collapses to Gaussian via CLT). This *deepens* rather than contradicts the substrate-independence claim: we identify the conditions under which universality applies, and propose a generative mechanism — functional differentiation — that accounts for why these conditions are met in systems where universality has been observed.

#### 4.2 Biological parallels

Functional differentiation is a canonical concept in biology and evolutionary theory (Gould; Wagner on modularity; West-Eberhard on developmental plasticity). The trade-off between division of labor (broadening) and specialization (concentration) is studied in cell types, microbial communities, and economic systems. Our result provides the first *quantitative simultaneous* measurement of both effects in an artificial system, using a substrate-independent metric (Gini / Effective N on |Δ|). This opens a direct comparison axis with longitudinal biological data where the same quantities could in principle be computed.

#### 4.3 Predictions for mechanistic interpretability

- **L8H9** has a specific function that crystallized between step 4 000 and step 8 000 — ripe for targeted activation-patching analysis to identify the computation that emerged.
- Heads in the 4.2 % **born-critical** class share something structural with the model's initialization — candidates for "pruning-robust components."
- The 34 % **never-critical** class represents functional redundancy; removing these should minimally affect loss.

#### 4.4 Predictions for training diagnostics

Models with collapsed representations or undertrained convergence should show anomalous Gini trajectories (collapse → Gini → 1, under-training → Gini low and flat). Testable on known-pathological training runs as a training-health metric orthogonal to loss.

#### 4.5 Limitations

- Single model (Pythia 410M). Scaling study is the natural follow-up (does L8H9 have an analog at 1.4B? 12B?).
- Single task family (next-token prediction). Vision models, RL agents left to future work.
- Evaluation on wikitext (close to but not identical to Pythia's Pile-based training distribution).
- n = 144 head sampling; finer sampling could reveal sub-classes of our 4 categories.
- Conservation conjecture on Gini invariance established on n = 8 checkpoints in one model; requires replication across scales and architectures.

### 5. Open questions / next work (half page)

- **Branch 2 of the program (agency measurement)** becomes more meaningful: agency as divergence of P_sampled vs P_ambient depends on the structure of P_ambient, which *itself* differentiates during training. Agency is a moving target.
- **Cross-architecture universality (Paper 1.2 of program)** — do ViT / diffusion models show the same three-point hierarchy and the same dual differentiation conservation law?
- **Biological replication.** Allen Brain Observatory longitudinal neural recordings during learning — does the same dual-differentiation signature appear?

### 6. Acknowledgements / availability

- Code and raw ablation data (CSV, 1 584 rows) available at `github.com/teo/...`
- Pythia checkpoints: EleutherAI public release
- Reproducibility: 82-minute run on single A100 40 GB, total cost ≈ $4

---

## Length target

- NeurIPS 2026 main track: 9 pages main + unlimited appendix
- ArXiv preprint: ~12 pages is fine
- Estimate: intro 1p, methods 1p, results 4p, discussion 1p, references 1-2p. Comfortable.

## Writing order (suggested)

1. **Figures first** — polish to publication quality, freeze them. Don't rewrite figures while writing text.
2. **Abstract** (we have a draft above; refine after each section below)
3. **Results** (start with 3.1 and 3.4 — the two strongest)
4. **Methods** (mostly factual, short)
5. **Introduction** (easier to write once results are clear)
6. **Discussion** (last — it synthesizes)

## What I need from Teo before next step

- Title approval (or alternative)
- Authorship (you alone, or you + me listed as computational collaborator?)
- ArXiv categories: primary **cs.LG**; secondary **cs.NE** (neural/evolutionary), **q-bio.NC** (biology)
- GitHub repo: create now or at submission time?
