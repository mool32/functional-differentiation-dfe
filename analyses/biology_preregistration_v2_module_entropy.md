# Biology pre-registration v2 — module entropy as alternative PR

**Locked:** 2026-04-24, **before any module entropy → fate correlation is computed**
**Reason for amendment:** pre-registration v1 (commit `1b01020`) operational PR definition (`(Σx)²/(n·Σx²)` on top 2000 HVGs) failed robustness test. ρ range across parameter configs was 0.343 (pre-reg threshold <0.10). At pre-registered config it gave ρ=-0.337 but at N_HVG=500 or K_NN=50 the signal vanished to near zero. This amendment tests whether a principled alternative measurement — module entropy on the existing 43-module perceptome decomposition — is more robust and more faithful to the ML analog.

Importantly, the results already obtained with v1 pre-registration (ρ=-0.337 at default config, H3 trajectory, null controls, robustness sweep, binary outcome tests) remain locked as obtained. This v2 is a **separate test with a different measurement**, not a retroactive redefinition of v1.

---

## 1. What is locked at this point (no peek)

The following has NOT been examined for any correlation or fate outcome:
- Per-cell module entropy values from Schiebinger `obsm['module_scores']`
- Any derived quantity combining entropy with fate proxy

The following HAS been examined (earlier session steps):
- Shape and signedness of module_scores array (will be verified before this analysis runs)
- The 43-module names (from `uns['module_names']`)

Pre-commitment: if any inspection of module-entropy ↔ fate correlation has happened prior to this commit, that fact would be flagged. None has.

## 2. Operational definition (locked)

### 2.1 Module activity — handling signed values

Schiebinger `obsm['module_scores']` is a (n_cells, 43) matrix. Values may be signed (z-scored module activities — some modules may be negatively enriched). Two standard options for entropy:

**Option A (softmax):** `p_i = exp(s_i) / Σ exp(s_j)` per cell. Always non-negative, sums to 1.
**Option B (clipped-shift):** `p_i = max(s_i - min_j, 0)` renormalized. Drops negative activities.

**Pre-registered choice:** Option A (softmax). More principled, always well-defined, smoother. Uses all modules.

### 2.2 Module entropy per cell

```python
def module_entropy(module_scores):
    """Shannon entropy of softmax-normalized module activity per cell.
    module_scores: (n_cells, 43) array, possibly signed.
    Returns: (n_cells,) vector, entropy in nats, bounded [0, log(43)≈3.76].
    """
    s = np.asarray(module_scores, dtype=np.float64)
    s_stable = s - s.max(axis=1, keepdims=True)  # numerical stability
    p = np.exp(s_stable)
    p /= p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-20, 1.0)
    return -(p * np.log(p)).sum(axis=1)
```

**Direction of analog:**
- LOW module entropy = concentrated activity in few modules = committed / specialized cell
- HIGH module entropy = distributed activity = uninformed / plastic cell

Matches the intuition behind low-PR in ML (concentrated computation). Direction prediction therefore: LOW entropy ↔ HIGH fate commitment score.

## 3. Tests (parallel to pre-reg v1, same decision rules)

All tests reuse the v1 outcome definitions (fate_proxy_NN via pluripotency + binary fate_proxy_NN via 2i label). Only the per-cell predictor changes.

### 3.1 H1 Primary — module entropy at day 2-3 vs pluripotency-based fate

**Test:** ρ(module_entropy_day_2_3, fate_proxy_NN_pluripotency)
**Predicted direction:** NEGATIVE (low entropy → committed → close to iPSC in day-18 space)
**Four-tier decision:**
- PASS: |ρ| ≥ 0.20 AND direction negative AND CI excludes 0
- PARTIAL: |ρ| ∈ [0.10, 0.20] AND negative
- WEAK: |ρ| ∈ [0.05, 0.10] AND negative
- FAIL: wrong sign OR |ρ| < 0.05

### 3.2 H1' Primary — module entropy at day 2-3 vs binary fate (NN-propagated)

**Test:** ρ(module_entropy_day_2_3, fate_proxy_NN_binary_2i_fraction)
Same thresholds as H1. This is the **cleaner test** — direct label, less methodology choice.

### 3.3 H0 Null — day 0 (uninduced MEF)

Required: |ρ| < 0.15 at day 0 for H1 to count as early-window-specific.

### 3.4 H3 Phase transition (exploratory)

ρ(entropy_t, fate_proxy) per timepoint t. Plot trajectory, locate sign flip if any.

### 3.5 Robustness check — critical because v1 failed it

**The following "robustness" vectors are baked into the method:**
- No HVG count choice (all 43 modules used, no subset)
- No subset of modules chosen (Option A softmax uses all)
- The only free parameter is softmax temperature, which is fixed at 1 in pre-reg

So parameter-sensitivity is reduced to:
- Softmax temperature β: rerun with β ∈ {0.5, 1, 2, 4} as sensitivity check
- Fate proxy K_NN sweep {5, 10, 20, 50} (same as v1, since fate-proxy methodology unchanged)

**Pre-registered threshold:** range of ρ across softmax β and K_NN sweeps must be < 0.15.

### 3.6 Comparison to v1 (PR) — new diagnostic

Report ρ(module_entropy_day2-3, PR_day2-3) across same cells. If entropy and PR are highly correlated (|ρ| > 0.8), they measure the same thing and v2 is just a re-expression of v1. If moderately correlated (0.3-0.7), they capture different aspects. If weak (|ρ| < 0.3), they are largely independent measurements.

## 4. Decision tree

| H1 v2 | H1' v2 | robustness | interpretation |
|---|---|---|---|
| PASS | PASS | STABLE | **Module entropy is the correct biology analog.** Proceed to Bastidas-Ponce with this methodology. |
| PASS | PARTIAL | STABLE | Pluripotency-score aspect real; binary-label aspect weaker. Still a viable biology finding. |
| PARTIAL | PARTIAL | STABLE | Weak consistent signal; consider reformulating cross-substrate claim magnitude-wise. |
| — | — | UNSTABLE | Methodological fragility is inherent to biology operationalization; fundamental concept work needed before continuing. |
| FAIL | FAIL | — | Module entropy doesn't carry early-window signal either. Seriously weakens biology-side claim across all measurement choices. Step back to concept. |

## 5. Bias protection

Same commit discipline: this document + analysis script committed in same git commit BEFORE any module-entropy-vs-fate correlation is computed. Hash recorded in summary output.

---

*Locked 2026-04-24. Module entropy predictive-of-fate correlation not yet measured at time of locking.*
