# Pre-registration v3 — Cross-architecture replication on OLMo-2 1B

**Locked:** 2026-04-24, before any OLMo analysis runs.
**Purpose:** Cross-architecture replication test of the ML early-window invariant (established across three scales on Pythia 160M/410M/1.4B in commits `b69cfbf` through `a23562a`).
**Substrate:** `allenai/OLMo-2-0425-1B-early-training` — transformer from a different training project (AllenAI, not EleutherAI), different training data (Dolma, not Pile), different architecture variant (OLMo-2, not GPT-NeoX).

Binding under the five hard rules accepted before the drafting of this document:

1. **Direction pre-registered absolute.** Predicted: NEGATIVE ρ. Positive result = FAIL.
2. **Single primary pre-registered test.** One measurement, one correlation, drives the verdict.
3. **Numeric decision rule pre-locked.** Thresholds specified below.
4. **Null is legitimate.** If OLMo fails, ML finding becomes Pythia-family-specific, not within-ML-universal. Honest outcome.
5. **Post-hoc reformulation prohibited.** Direction mismatch = FAIL, not "different invariant".
6. **Pre-registration commit hash baked into verdict JSON.** Commit of this document recorded in output.

---

## 1. Model + checkpoints

- **Model:** `allenai/OLMo-2-0425-1B-early-training` (HuggingFace)
- **Architecture:** OLMo-2 transformer, separate Q/K/V projections, RMSNorm, rotary embeddings. `Olmo2ForCausalLM` class in transformers. Specs auto-detected from config: hidden_size, num_hidden_layers, num_attention_heads, head_dim.
- **Training data:** Dolma
- **Checkpoint range:** step 0 to step 37000 (every 1000 steps, some may be missing per AllenAI notes)
- **Revision naming:** `stage1-step{N}-tokens{M}B` (actual token counts looked up at runtime via `list_repo_refs`)
- **Final checkpoint for this test:** step 37000

## 2. Sampling + methodology

- **Heads:** ALL heads at all layers. If `num_attention_heads × num_hidden_layers ≤ 300`, ablate all. If larger, sample 6 per layer at head indices `[0, 3, 6, 9, 12, 15]` (if num_heads ≥ 16), matching Pythia methodology.
- **Precision:** float32 + TF32 matmul (same as Pythia tests).
- **Eval data:** wikitext-103 train stream, 25 × 4 × 2048 tokens (same as Pythia tests).
- **Ablation:** zero out `o_proj.weight[:, h·d_head:(h+1)·d_head]` for head h in layer L. SHA-256 save/restore verification before sweep starts.
- **OV_PR computation:** for head h in layer L,
  `W_V = v_proj.weight[h·d_head:(h+1)·d_head, :]`  (shape d_head × hidden),
  `W_O = o_proj.weight[:, h·d_head:(h+1)·d_head]`  (shape hidden × d_head),
  `M_OV = W_O @ W_V`,
  `PR = (Σ σᵢ²)² / Σ σᵢ⁴` on singular values of `M_OV`.

## 3. Primary pre-registered test

**Statistic:** Spearman ρ between `OV_PR_h` at step 1000 and `|Δ_h|` at step 1000, across all sampled heads.

**Predicted direction:** NEGATIVE (same as Pythia cross-scale, no reinterpretation permitted).

**Four-tier decision rule:**
- **PASS:** |ρ| ≥ 0.30 AND direction negative AND p < 0.01 AND methodology-null gate passed.
- **PARTIAL:** 0.20 ≤ |ρ| < 0.30 AND direction negative AND methodology-null gate passed.
- **WEAK:** 0.10 ≤ |ρ| < 0.20 AND direction negative (methodology gate informational).
- **FAIL:** direction positive OR |ρ| < 0.10.

Pythia reference: ρ = −0.416 (160M), −0.555 (410M), −0.484 (1.4B). All well into PASS tier.

**Comparison to Pythia cross-scale window:** if OLMo ρ lands within [−0.70, −0.30], we report "cross-architecture replication within ML". If within [−0.30, −0.10], we report "weak architectural dependence". If positive or null, "phenomenon Pythia-family-specific".

## 4. Methodology-null gate (mandatory pre-flight)

Before interpreting primary ρ, run 200 shuffles of ablation outcome among sampled heads at step 1000. Compute null distribution of ρ.

- Gate **PASS** if null 95% CI within [−0.10, +0.10]: primary test interpretable at pre-registered thresholds.
- Gate **CAUTION** if null 95% CI within [−0.15, +0.15]: primary thresholds raised by +0.05.
- Gate **FAIL** if null 95% CI exceeds ±0.15: HOLD verdict regardless of observed ρ.

## 5. Secondary exploratory (reported, not decision-defining)

- Trajectory ρ(OV_PR_t, |Δ|_t) at step 2000, step 4000, step 8000, step 16000, step 37000
- Step 0 lottery control: ρ(OV_PR_step0, |Δ|_step1000) — expected near null
- QK participation ratio analysis (same as Pythia secondaries S-3, S-4)

All flagged EXPLORATORY. Do not upgrade to primary post-hoc.

## 6. Compute budget

Colab A100 80GB or RunPod A100 80GB. Estimate ~5-8 hours for full sweep. CSV append-resume so safe across disconnects.

Cost estimate: ~$15-25 RunPod pricing.

## 7. Artifacts

- `tier2_olmo_ablations.csv` — 7 checkpoints × all sampled heads
- `tier2_olmo_spectral.csv` — invariants at 8 checkpoints (including step 0)
- `tier2_olmo_verdict.json` — pre-reg commit hash, precision, actual revision names used, primary + secondary results, verdict tier

## 8. What is NOT pre-registered

- Any test on OLMo that differs from the primary + secondaries listed here
- Any parameter sweep beyond the locked configuration
- Any comparison to Pythia that re-interprets Pythia results post-hoc
- Any biology comparison — this is cross-architecture within ML only

If null result, no rescue attempts. Finding either cross-architecture or it is not.

---

*Locked 2026-04-24. Pre-registration commit hash baked into `tier2_olmo_verdict.json` as `pre_registration_commit` field.*
