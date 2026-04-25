# Pre-registration v6 — Cross-team replication on TinyLlama-1.1B

**Locked:** 2026-04-25, before any TinyLlama analysis runs.

**Purpose:** Third independent verification of the ML early-window invariant. Adds a *third independent training team* (Singapore U / TinyLlama Project) using *third independent data* (SlimPajama + StarCoder) and *third independent architecture variant* (Llama-style: RMSNorm, SwiGLU, RoPE).

**Rules 1–6 binding** (locked since v3 OLMo):
1. Direction NEGATIVE absolute. Positive = FAIL.
2. Single primary test, single decision.
3. Numeric thresholds pre-locked.
4. Null is legitimate.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

**Locked findings to date that this test extends, not modifies:**
- Pythia 160M / 410M / 1.4B (GPT-NeoX, Pile): ρ = −0.42 / −0.55 / −0.48. Pre-reg PASS.
- OLMo-2 1B-early (OLMo2, Dolma): ρ = −0.49. Pre-reg PASS.
- Range observed: ρ ∈ [−0.55, −0.42]. Window for prediction interpolation.

---

## 1. Model + checkpoints

- **Model:** `TinyLlama/tinyLlama-intermediate-checkpoints` (HuggingFace, 145 intermediate revisions)
- **Architecture:** Llama-style transformer, 22 layers × 32 attention heads, hidden=2048, head_dim=64. Auto-detected from config.
- **Training:** SlimPajama (~3T tokens) + StarCoder. 1431k total steps.
- **Revision naming:** `step-{N}k-token-{M}B` (e.g., `step-10k-token-21B`, `step-100k-token-210B`).
- **Equivalent-fraction-of-training mapping to Pythia:**

| Pythia step | fraction | TinyLlama step (closest available) |
|-------------|----------|-----------------------------------|
| 1000 | 0.7 % | **step-10k** (0.7 %) |
| 2000 | 1.4 % | step-20k (1.4 %) |
| 4000 | 2.8 % | step-40k (2.8 %) |
| 8000 | 5.6 % | step-80k (5.6 %) |
| 16000 | 11.2 % | step-160k (11.2 %) |
| 64000 | 44.8 % | step-640k (44.7 %) |
| 143000 | 100 % | step-1431k (100 %) |

This mapping is locked. Any deviation is reported as a deviation in the verdict JSON.

If a specific revision is missing from the repository at run time, the closest-available 5k-step revision is used, and the substitution is logged.

## 2. Sampling + methodology

- **Heads:** sampled `[0, 4, 8, 12, 16, 20, 24, 28]` × 22 layers = **176 heads** (matches Pythia's "6 per layer × 24 layers = 144" sampling density of ~25 % of total heads).
- **Precision:** float32 + TF32 matmul.
- **Eval data:** wikitext-103 train stream, 25 × 4 × 2048 tokens. (Same as all prior pre-regs.)
- **Ablation:** zero out `o_proj.weight[:, h·d_head:(h+1)·d_head]` for head `h` of layer `L`. SHA-256 save/restore verification before sweep.
- **OV_PR:** participation ratio of singular values of `W_O[:, h·d_head:(h+1)·d_head] @ W_V[h·d_head:(h+1)·d_head, :]`. λ-free.
  - **GQA handling:** if `num_key_value_heads < num_attention_heads`, the V slice for head h is taken from K/V group it belongs to (group index = h // (num_heads/num_kv_heads)). Each Q head in a group uses the same V slice. This is the architecturally correct OV.

## 3. Primary pre-registered test

**Statistic:** Spearman ρ between `OV_PR_h` at step-10k and `|Δ_h|` at step-10k, across all 176 sampled heads.

**Predicted direction:** **NEGATIVE.**

**Four-tier decision:**
- **PASS:** |ρ| ≥ 0.30 AND direction negative AND p < 0.01 AND methodology-null gate PASS.
- **PARTIAL:** 0.20 ≤ |ρ| < 0.30 AND direction negative AND gate PASS.
- **WEAK:** 0.10 ≤ |ρ| < 0.20 AND direction negative AND gate ≥ CAUTION.
- **FAIL_WRONG_DIRECTION:** direction positive AND |ρ| ≥ 0.10.
- **NULL:** |ρ| < 0.10.

Reference: Pythia + OLMo window [−0.55, −0.42]. Easy PASS expected if invariant universal.

## 4. Methodology null gate

200 shuffles of `|Δ|` among heads at step-10k. Compute null ρ distribution.
- Gate **PASS** if null 95 % CI within ±0.10.
- Gate **CAUTION** if within ±0.15. Thresholds raised by +0.05.
- Gate **FAIL** if exceeds ±0.15. HOLD verdict.

## 5. Secondary exploratory (reported, not decision-defining)

- Trajectory ρ at step-20k / step-40k / step-80k / step-160k / step-640k / step-1431k.
- Step-0-equivalent: TinyLlama earliest available checkpoint (likely step-10k itself; if step-1k or step-5k available, use as lottery control). Reported as descriptive.
- QK_PR trajectory (S-3 analog).
- QK active-vs-dead gap at step-1431k (S-4 analog).

All flagged EXPLORATORY. Do not promote to primary post-hoc.

## 6. Compute

Colab A100 80GB or RunPod A100. Estimate:
- 176 heads × 7 checkpoints = 1232 ablations × ~5 sec = 1.7 h pure ablation
- Spectral × 7 checkpoints × ~9 min = 63 min
- Loading × 7 = 5 min
- **Total ≈ 3 h.** Safe within Colab Pro+ session.

Cost: ~$5–10 RunPod, free Colab Pro+.

## 7. Artifacts

- `tier2_tinyllama_ablations.csv`
- `tier2_tinyllama_spectral.csv`
- `tier2_tinyllama_verdict.json` (pre-reg commit hash, primary + secondary results)

## 8. What is NOT pre-registered

- Different architecture variant tests (e.g. ablation on q_proj or k_proj instead of o_proj)
- Different aggregation of |Δ|
- Comparison re-interpreting prior pre-regs

If v6 FAILS, ML cross-architecture claim becomes "Pythia + OLMo (transformer family A & B)" only. TinyLlama-style Llama family would be excluded. Honest result.

---

*Locked 2026-04-25. Pre-registration commit hash recorded in verdict JSON. One-shot. No rescue.*
