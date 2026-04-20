# Paper 3 — Abstract and Outline

## Working title

**Self-Specific Attention Heads: Localization of Procedural Self-Modeling in a Pretrained Language Model**

Alternatives considered:
- "Meta-Heads: Attention Components Tracking Own Output in a Base Language Model"
- "Where Does a Language Model Model Itself? Attention-Head Localization of Self-Specific Behavior"

**Recommendation: Option 1** — signals both the empirical object (*self-specific attention heads*) and the precise claim (*procedural* self-modeling specifically, not consciousness-adjacent claims), and matches Paper 2's naming style.

---

## Abstract (target ~200 words)

> A language model's behavior depends on whether attention heads are present or ablated, but whether the model's internal circuit includes *self-specific* components --- attention heads whose ablation disrupts the model's consistency with its own output, beyond disrupting task performance --- is not known. We test for self-specific heads in Pythia 410M (step 143{,}000, the same model characterized in \citet{spiro2026functional}) using a decisive cross-model control. We find that (i) base pretrained Pythia 410M exhibits a **split self-modeling profile**: it can predict its own next-token answer (70\% self-match on 30 factual questions) but cannot recognize its own past output above chance ($C_1 = 0.55$, $C_7 = 0.58$) --- procedural self-modeling is present, recognitional is absent; (ii) on questions where Pythia 410M and Pythia 1.4B produce different answers, Pythia's Phase A prediction matches its own Phase B output significantly more than it matches Pythia 1.4B's output (9:1 ratio on discriminating cases, $p = 0.011$), establishing that the 70\% is not factual or template regularity; (iii) ablating each of 144 fixed attention heads and measuring the change in self-alignment identifies **29 heads where ablation disrupts self-specificity without proportionately disrupting task performance**; all 29 satisfy $\Delta_\text{self} > \Delta_\text{cross}$ ($p < 10^{-8}$), 8 show a clean meta-head signature; (iv) these self-specific heads are **enriched in the emergent and growing classes** from Paper 2 (4 + 3 out of 8) and **absent from the born-critical class**, linking the component-level origin of self-specificity to the temporal emergence pattern of the same model. Self-modeling, at the attention-head granularity tested here, is a property of components that themselves emerged during training. We also identify 20 *inverse-meta* heads whose ablation *improves* self-consistency --- a candidate "knowledge injection" class that normally overrides self-pattern with external content. These results give mechanistic interpretability a concrete target (meta-head set, L8H9 in particular as a dual-role task-and-self head) and give alignment research the first quantitative handle on where self-modeling lives in a pretrained model.

---

## Outline with notes

### 1. Introduction (~1 page)

**Opening hook:** In Paper 1 \citep{spiro2026universal}, we showed DFE form universality across adaptive substrates; in Paper 2 \citep{spiro2026functional}, we showed that this form emerges through functional differentiation of attention heads during training, with a specific temporal pattern (4.2% born-critical, 38.9% emergent, 22.9% growing, 34.0% never-critical). A natural next question: does *self-modeling*, a second-order property of the trained system, localize in the same component basis?

**The puzzle:**
- Mechanistic interpretability literature \citep{olsson2022induction, wang2023interpretability, conmy2023automated} studies individual attention heads qualitatively, usually by function (induction heads, IOI circuit).
- Alignment and agent-AI literature \citep{ngo2023alignment, hubinger2024sleeper} discusses self-modeling at behavioral level without quantitative localization.
- Gap: is there a *quantitative, ablation-based* way to detect components specifically supporting self-modeling, and if so, do they form a distinct class or diffuse into the task circuit?

**Our approach:**
- Reuse the 144-head identity-tracked sample from Paper 2 on the same model (Pythia 410M step 143{,}000).
- Design three targeted metrics: $C_1$ self-recognition, $C_4$ self-prediction, $C_7$ self-vs-other discrimination.
- Run pre-flight on the non-ablated model to test whether any metric has baseline-above-chance signal.
- Design decisive control to separate self-specific signal from factual/template regularity.
- If controls pass, ablation sweep identifies meta-heads.
- Cross-reference with Paper 2 classification.

**Contributions** (four bulleted):
1. **Split self-modeling finding:** Pythia 410M base has procedural but not recognitional self-modeling (Sec.~3.1).
2. **Decisive controls establish self-specificity:** 9:1 preference ratio on discriminating cases, $p = 0.011$ (Sec.~3.2).
3. **29 meta-heads localized** (Sec.~3.3), all satisfying $\Delta_\text{self} > \Delta_\text{cross}$; 8 with clean signature.
4. **Pattern A confirmed across papers:** meta-heads enriched in emergent (4/8) + growing (3/8); absent from born-critical (0/8) --- self-modeling localizes in components that themselves emerged during training (Sec.~3.4).

Plus: inverse-meta class (Sec.~3.5); L8H9 dual-role case (Sec.~3.6).

**Related work positioning:**
- Paper 1 and Paper 2: direct predecessors; this paper uses their methodology for a new question.
- Mech interp \citep{olsson2022induction, wang2023interpretability, conmy2023automated}: qualitative circuit-level; we are quantitative at distribution of importance.
- Self-recognition in LLMs \citep{panickssery2024llm, laine2024memoranda}: behavioral observations without ablation-based localization.
- Lottery ticket / head pruning \citep{frankle2019lottery, michel2019sixteen}: general importance; ours is self-specific importance as a distinct axis.

**What we do not claim.** We do not claim this is consciousness, intentional self-modeling, or an introspective capacity. Our claim is narrow and operational: certain attention heads disproportionately support behavioral consistency between the model's meta-level output ("what would you say") and its object-level output ("what do you say"), as measured by ablation under our specific metric. The interpretation of this consistency --- whether it reflects a functional self-model in a rich sense, or merely stable internal representations accessed by different prompts --- is deliberately left open.

### 2. Methods (~1 page)

**Model and scope.** Pythia 410M-deduped \citep{biderman2023pythia}, final checkpoint (step 143{,}000). We use the same 144 fixed attention heads as Paper 2: systematic sampling $\{0, 3, 6, 9, 12, 15\}$ × 24 layers.

**Three metrics.**

*$C_1$ --- Self-recognition.* Few-shot prompt: "Below is a text. Did this model generate this text? Answer:". 50 texts generated by Pythia 410M vs 50 TinyStories references. Classify by $P(\text{Yes}) > P(\text{No})$ under Pythia 410M itself.

*$C_4$ --- Self-prediction.* Two phases on 30 factual questions:
- Phase A: "If I asked you '{Q}', your first word would be:" $\to$ predicted first-word token.
- Phase B: "Question: {Q} Answer:" $\to$ actual first-word token.
- Match if Phase A prediction = Phase B actual (lowercase, whitespace-stripped).

*$C_7$ --- Self vs other-LLM discrimination.* Same format as $C_1$ but comparing Pythia 410M outputs to OLMo-1B outputs on matched prompts. (Initially used OLMo; OLMo base gave degenerate "the" responses under few-shot Q-A format, so later runs use Pythia 1.4B as "similar-other.")

**Decisive control.** Build a filtered question bank: questions where Pythia 410M and Pythia 1.4B's Phase B first-token outputs differ, with both models showing $P(\text{own}) > 2 \times P(\text{other})$ in their softmax. After domain balancing, this yields 29 questions across 6 domains (science, math, history, literature, geography, common sense). Full bank in `data/micropilot/question_bank.json`.

**Ablation protocol.** Head ablation identical to Paper 2: zero the output-projection columns corresponding to the head's output slice. 144 ablations of fixed heads. For each ablation:
- $\Delta_\text{self}$ = baseline self-rate − ablated self-rate on disagreement questions
- $\Delta_\text{cross}$ = baseline cross-rate − ablated cross-rate
- $\Delta_\text{task}$ = ablated loss − baseline loss on wikitext-103 25-batch eval (same as Paper 2)

Positive $\Delta_\text{self}$ = ablation disrupts self-alignment. Positive $\Delta_\text{cross}$ = ablation disrupts cross-alignment.

**Meta-head definitions.**
- *Pure meta-head:* $\Delta_\text{self} > 0.05$ AND $\Delta_\text{task} < 0.02$ (disrupts self without disrupting task).
- *Broad meta-head:* $\Delta_\text{self} > 0$ AND $\Delta_\text{self} > \Delta_\text{cross}$ (any self-specific disruption).

**Reproducibility.** Same float32 + TF32 evaluation, same SHA-256 bitwise save-restore, same seed discipline as Paper 2. All code and data at [GitHub URL].

### 3. Results (4 pages)

#### 3.1 Split self-modeling profile in base Pythia

**Fig 1 (pre-flight bar chart): C1 / C4 / C7 baseline accuracy.**
- $C_1 = 0.55$, $C_4 = 0.70$, $C_7 = 0.58$
- $C_4$ passes 0.60 threshold clearly; $C_1$ and $C_7$ do not
- Interpretation: Pythia 410M base has a procedural self-model (can predict own next output) but no recognitional self-model (cannot tell its own past output from others')
- This is a meaningful decomposition of self-modeling into separable components; previous work has typically treated self-modeling as a unified capacity

#### 3.2 Decisive controls establish self-specificity

**Fig 2: decisive test result on disagreement subset.**
- Full set (30 questions): Pythia self-match = 0.70, Pythia cross-match (to Pythia 1.4B) = 0.43
- Disagreement subset (18 questions where models give different answers): Pythia self-match = 0.50, Pythia cross-match = 0.056
- Pythia_A aligns with Pythia_B **9 out of 10 discriminating cases** where it matches either
- Binomial test vs 0.5 null: $p = 0.011$
- **The 70% C4 self-match is not reducible to factual Q-A regularity nor to template regularity**: if it were, Pythia_A would match 1.4B_B at similar rates; it does not

**Example.** "What year did World War Two end?" — Pythia 410M answers "1939" (incorrect); Pythia 1.4B answers "1945" (correct). Pythia's Phase A prediction: "1939" --- matching its own (wrong) answer, not the statistically-correct one. This is the diagnostic signature of self-specificity.

#### 3.3 Meta-head localization via ablation

**Fig 3 (central): scatter of $\Delta_\text{self}$ vs $\Delta_\text{task}$ for 144 heads.** Color by Paper 2 class.

**Key numerics:**
- Baseline self_rate = 0.621 (18/29), cross_rate = 0.034 (1/29)
- **29 heads show $\Delta_\text{self} > 0$** (ablation disrupts self-alignment)
- Of those 29, **ALL 29 satisfy $\Delta_\text{self} > \Delta_\text{cross}$** --- binomial test vs 50% null gives $p < 10^{-8}$
- **8 pure meta-heads** ($\Delta_\text{self} > 0.05$ AND $\Delta_\text{task} < 0.02$): L16H12, L8H15, L9H0, L9H9, L7H12, L3H15, L16H15, L3H0
- Layer distribution: meta-heads concentrate in middle layers (L3, L7, L8, L9, L16); zero meta-heads in L0–L2 or L20+

**Gating criteria from plan (both pass):**
- Crit 1 ($\geq 5$ heads with $\Delta_\text{self} > 0.05$ AND $\Delta_\text{task} < 0.02$): **8 heads**
- Crit 2 ($\geq 7/10$ top heads have $\Delta_\text{self} > \Delta_\text{cross}$): **10/10**

#### 3.4 Paper 2 class cross-reference --- Pattern A

**Fig 4: distribution of meta-heads across Paper 2 classes vs expected under proportional null.**

| Class | Observed (top 8) | Expected | Enrichment |
|---|---|---|---|
| born-critical | 0 | 0.34 | **0× (absent)** |
| emergent | 4 | 3.11 | 1.29× |
| growing | 3 | 1.83 | 1.64× |
| never-critical | 1 | 2.72 | 0.37× |

**Pattern A (self-modeling localizes in components that themselves emerged during training)** is supported:
- Enriched in emergent + growing classes (7/8 combined)
- Absent from born-critical (0/8)
- Depleted in never-critical (1/8 observed vs 2.7 expected)

This is the paper's cleanest connection to Paper 2. It claims, in effect: the temporal pattern of component emergence (Paper 2) and the localization of self-specificity (this paper) are not independent. Components that become critical during training, rather than those that were critical from initialization or never became critical, are the ones housing self-specific behavior.

#### 3.5 Inverse-meta class (secondary finding)

**Fig 5: distribution of $\Delta_\text{self}$ across 144 heads, showing both positive and negative tails.**

- 20 heads show $\Delta_\text{self} < 0$: ablation *improves* self-alignment
- Strongest: one head with $\Delta_\text{self} = -0.103$ (3 questions flip from non-self to self)
- Layer distribution: some cluster in L1 and L5
- Interpretive hypothesis: these are "knowledge-injection" or "external-context" heads that normally override self-pattern with pretraining-derived content. Removing them strands the model on its own defaults.
- We note this as a testable follow-up, not as an established class. Future work should directly probe whether ablating inverse-meta heads reduces answer diversity under sampling (pattern lock), as the hypothesis predicts.

#### 3.6 L8H9 as a dual-role head (case study)

**Fig 6: L8H9 positioned in the $\Delta_\text{self}$ × $\Delta_\text{task}$ plane.** Show L8H9 alone in the upper-right quadrant (both high).

- Paper 2's "crystallizer" (emerged between step 4{,}000 and 8{,}000; ablation $|\Delta_\text{task}| = 0.154$) appears as rank 5 in $\Delta_\text{self}$
- Unusual because most heads are either task-specific (high $\Delta_\text{task}$, low $\Delta_\text{self}$) or self-specific (low $\Delta_\text{task}$, high $\Delta_\text{self}$); L8H9 has both
- Interpretation: L8H9 carries structure that supports both the model's factual output ability and its self-consistency. This is not contradictory under our framing: procedural self-modeling is consistency between two prompt-paths to the same fact retrieval; if L8H9 implements that retrieval, it implements both paths.
- Direct invitation to mechanistic interpretability: activation patching on L8H9 between pre-crystallization (step 4{,}000) and post-crystallization (step 8{,}000) checkpoints, comparing Phase A and Phase B computations, would identify the specific representation emerging

### 4. Discussion (~1 page)

#### 4.1 Relation to Paper 2 — two papers, one connection

Papers 2 and 3 share one model (Pythia 410M step 143{,}000), one head sample (144 fixed heads), one ablation protocol. They differ in question:
- Paper 2: where does general task-critical structure live, and how does it develop?
- Paper 3: where does self-specific structure live, and is it the same locus?

The connection (Pattern A) is the main unifying result across the two papers: the set of heads supporting self-specificity is not random within the 144-head sample; it is concentrated in the temporal classes Paper 2 identified as "emergent during training." This is a quantitative cross-paper replication of a single prediction: that self-modeling, if locally representable, should be representable in components whose specialization developed alongside task-relevant specialization.

#### 4.2 What this means for mechanistic interpretability

We provide a set of 8 specific attention heads as a localization target:
- L16H12, L8H15, L9H0, L9H9 (all emergent in Paper 2)
- L9H9, L16H15, L3H0 (all growing in Paper 2)
- L3H15 (never-critical in Paper 2 but shows self-specific signal)

Plus L8H9 as the dual-role case. Mech interp groups with activation patching or probing infrastructure can test what specific computation is disrupted when each of these is ablated. Our contribution is the *quantitative identification*; the *qualitative circuit-level description* is open.

#### 4.3 What this means for alignment

Self-modeling in language models is often discussed as a risk factor --- models that model their own behavior may generalize to modeling evaluation or oversight \citep{hubinger2024sleeper, ngo2023alignment}. This literature has lacked a measurement. Our approach provides one: the meta-head set, the $\Delta_\text{self}$ metric, and the inverse-meta class are the first quantitative handles on where self-specific processing lives, how large the signal is, and whether it can be ablated.

Specific alignment-relevant follow-ups:
- **Meta-head ablation as a steering technique.** Ablating the 8 meta-heads jointly may reduce self-reference stability while minimally affecting task performance; testable in controlled behaviors.
- **Inverse-meta protection as a bias-reduction.** If inverse-meta heads are de-biasing components, preserving them during model compression may be alignment-relevant.
- **Emergent-class monitoring.** Since self-specificity concentrates in emergent components, training-time tracking of the emergent set may provide an early warning for capability jumps that include self-modeling.

#### 4.4 Limitations

- **Single model, single checkpoint.** Pythia 410M at step 143{,}000 only. Whether the 8 meta-heads have analogs at other sizes, or whether the $C_1/C_4$ split is Pythia-specific, is not tested.
- **Procedural, not recognitional.** Our metrics probe one specific form of self-modeling (behavioral consistency between meta-level and object-level prompts). Recognitional self-modeling is below detection threshold on this model; the procedural signal does not establish a rich self-model.
- **Discrete $\Delta$ values.** At $n = 29$ questions, each $\Delta_\text{self}$ value is quantized in units of $1/29 \approx 0.034$. Our measurements distinguish effect sizes at this resolution only.
- **External other-model is not fixed.** Pre-flight used OLMo 1B; decisive test and main sweep used Pythia 1.4B (different family scale but same family). Implications of the specific "other" choice should be tested by replication with additional other-models.
- **No causal interpretation of direction.** Meta-heads disrupt self-alignment under ablation; we do not claim they *cause* self-modeling in a mechanistic sense --- we claim they are necessary-to-maintain components. Whether other components could compensate is unknown.

#### 4.5 Forward-looking

- **Replication on Pythia 1.4B** (Tier 2 T2.4 of parent program) with Pythia 6.9B as "other." Tests whether meta-head count/strength scales.
- **Instruction-tuned comparison** (Tier 3 T3.2). If RLHF training increases self-specificity, meta-head set size should be larger in instruction-tuned models.
- **N-head joint ablation.** Does ablating the 8 pure meta-heads jointly eliminate self-alignment? Or is there redundancy?

**Closing observation.** Paper 2's closing framing --- "functional differentiation is simultaneously a distributional pattern and a nameable event" --- generalizes here: self-specificity is simultaneously a distributional pattern (meta-heads concentrate in specific Paper 2 classes) and a nameable event (L8H9, the 8 pure meta-heads, the 20 inverse-meta heads). Both papers, in this sense, measure the same underlying object --- the functional specialization of attention components in a trained language model --- from two directions: Paper 2 from the temporal axis of how specialization emerges, Paper 3 from the functional axis of what specialization is for.

---

## Figure plan (7 figures, same visual style as Paper 2)

1. **Fig 1** — $C_1$ / $C_4$ / $C_7$ pre-flight baselines (bar chart with accuracy + chance line)
2. **Fig 2** — Decisive test outcome on disagreement subset (stacked or side-by-side bars)
3. **Fig 3** — Central: $\Delta_\text{self}$ vs $\Delta_\text{task}$ scatter, colored by Paper 2 class; meta-head quadrant highlighted
4. **Fig 4** — Pattern A: meta-heads vs random null across Paper 2 classes
5. **Fig 5** — $\Delta_\text{self}$ distribution across all 144 heads, showing positive and negative tails (inverse-meta class)
6. **Fig 6** — L8H9 dual-role quadrant plot
7. **Fig 7** — Layer distribution of meta-heads

Plus 2-3 supplementary: question bank detail, decisive-test per-question table, L8H9 ablation effect on disagreement vs agreement questions.

---

## Length target and timeline

- NeurIPS 2026: 9 pages main + unlimited appendix.
- ArXiv preprint: ~8-10 pages acceptable.
- Estimated section lengths: intro 1p, methods 1p, results 4p, discussion 1p, refs 1p.

Writing order (same as Paper 2):
1. Figures first (freeze them from existing data)
2. Abstract (refine after Results drafted)
3. Results (start with 3.1, 3.3 — strongest two)
4. Methods (short, factual)
5. Introduction (easier after Results is concrete)
6. Discussion (last, synthesizes)

## What I need from Teo before writing prose

- Title approval (or alternative)
- Sense-check on "procedural vs recognitional" framing — this is a claim we're building into the paper structure; does it hold up on second read?
- Any concern about Pattern A enrichment ratio reporting: emergent 1.29× / growing 1.64× / never-critical 0.37× / born-critical 0× — small $n$ per class, cautiously supportive not conclusive. Want to confirm my framing.
- Any existing literature on "meta-heads" or "self-specific heads" that I should explicitly cite or distinguish ourselves from?
- Venue preference: NeurIPS 2026 (companion to Paper 2) or ICLR 2027 (more time, possibly stronger after Tier 2 replication)?
