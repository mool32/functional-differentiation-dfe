# Session state — 2026-04-24

Complete snapshot of the spectral-invariants investigation for easy restoration in a future session. Paired with memory file `~/.claude/projects/-Users-teo/memory/project_dfe_pilot.md`.

**Do not delete this file.** It is the single source of truth for where Phase 1 ended and what remains to do before paper writeup.

---

## 1. One-line state

Phase 1 invariant search on Pythia 160M / 410M / 1.4B is complete. Four of five pre-registered tests on 1.4B passed. Writeup deferred. User will separately draft biological validation plan. No new experiments pending until biology plan arrives or user explicitly opens Block 2 / Block 3 / writeup track.

---

## 2. Commit map (GitHub `mool32/functional-differentiation-dfe`)

| commit | content | function |
|--------|---------|----------|
| `b69cfbf` | Tier 1 integrated into Paper 2 | Paper 2 arXiv-ready state |
| `056913f` | Tier 2 notebooks T2.1–T2.4 | Scaling + Pile + inverse-meta + 1.4B self-modeling |
| `bd0e643` | Block 1.2 per-class DFE + robustness pipeline | Classification AIC story, 410M + 160M |
| `f99769b` | Pre-registration v1 (Phase 1a) | Primary OV_PR at step 143k, four spectral, random control |
| `51d5341` | Phase 1a result: primary NULL, secondary QK findings | Null confirms, finding framing starts |
| `0b4a7b3` | **Locked framing doc** Phase 1a | Prevents post-hoc reformulation |
| `c1d3382` | Phase 1b-B/C: stability + 160M | QK discrimination is training-emergent and universal |
| `d341c99` | **Phase 1b-D dense scan** | Two-phase sign-flip structure discovered on 410M + 160M |
| `10316a4` | **Phase 1b-E + pre-reg v2** | Intra-class independence passes on 410M; lottery rejected; 1.4B pre-reg locked |
| `8be0f11` | Pre-reg v2 amendment (weak tier) | Added 0.10–0.20 weak tier before 1.4B data |
| `3ddf146` | 1.4B Colab notebook | One-click `tier2_pre1p4b_validation.ipynb` |
| `a23562a` | **1.4B pre-registered validation complete** | 4/5 PASS, locked verdict report |

## 3. Key findings (status summary)

| # | Finding | Evidence | Phase | Locked? |
|---|---------|----------|-------|---------|
| A | Pre-reg OV_PR @step143k NULL | ρ = −0.126, p = 0.13 | H | ✓ |
| B | QK active-vs-dead discrimination | gap +17/+20/+41 across 160M/410M/1.4B, step-0 null | H | ✓ |
| C | Category label breaks at 160M | emergent_dummy ρ = +0.07 on 160M vs +0.70 on 410M | E | awaiting biology |
| D | OV_PR @step1000 predicts \|Δ\| | ρ = −0.42/−0.55/−0.48 on 160M/410M/1.4B, lottery ticket rejected, training-driven | H on 3 scales | ✓ |
| E | Two-phase sign flip of ρ(QK_PR, \|Δ\|) | universal across 3 scales, transition between step 2000 and step 4000 | H on 3 scales | ✓ |
| F | QK gap scales with model size | +17 → +20 → +41 | E (first observed) | awaiting follow-up |

### Three-scale magnitude summary (locked)

```
Primary invariant (OV_PR at step 1000 vs |Delta| at step 1000):
  160M  n=144  rho = -0.416
  410M  n=144  rho = -0.555    (peak magnitude)
  1.4B  n=144  rho = -0.484

Two-phase sign flip (rho(QK_PR, |Delta|) at same checkpoint):
  step 1000   all three scales negative (-0.32 / -0.64 / -0.55)
  step 4000   all three scales positive (+0.30 / +0.25 / +0.13)
  step 143k   all three scales positive (+0.46 / +0.39 / +0.55)
  Sign crosses zero between step 2000 and step 4000 universally.

QK active-vs-dead gap at step 143000:
  160M  +17.12
  410M  +20.31
  1.4B  +40.88    (doubles — scaling behavior)
```

## 4. Methodology locked for any future work

- Primary invariant: OV participation ratio = (Σ σᵢ²)² / Σ σᵢ⁴ of singular values of `W_O[:, h·d_head:(h+1)·d_head] @ W_V[h·d_head:(h+1)·d_head, :]`. GPT-NeoX QKV layout: `qkv[3·d_head·h : 3·d_head·(h+1)]` with interleaved Q/K/V per head.
- Secondary invariants: QK participation `(Q_h.T @ K_h)`, OV/QK spectral entropy.
- Classification: strict Paper 2 rule, init threshold 5e-4 for born/never, born_low 1e-4 for emergent.
- Bootstrap: 2 000 resamples for class-level fits (Student-t MLE), 10 000 for invariant Spearman CI.
- Outcome for regression: log(max(|Δ|, 1e-6)) to handle heavy tails that break raw OLS.
- Absorb-the-label test: `log|Δ| ~ class_dummy + z(invariant)`; report fractional reduction in β_label and joint R² vs alone.
- Resume protocol: CSV append per head × checkpoint with SHA-256 save/restore verification.
- Pre-registration amendment rule: amendments allowed before any data collection; four-tier decision rules (Pass / Partial / Weak / Fail) locked before run.

## 5. File map (what each analysis script does)

### In `paper/analyses/`

- `per_class_dfe.py` — Block 1.2 primary per-class Student-t fits with MoM and MLE backups. 410M + 160M.
- `per_class_dfe_robustness.py` — Five-block robustness pipeline: threshold sweep, class-count matching, shuffle null, cross-model AIC-inversion bootstrap, pooled vs time-resolved.
- `per_class_dfe_summary.json` — numeric results of both.
- `per_class_dfe_robustness_summary.json` — ditto.

- `phase1a_spectral.py` — Pre-registration v1 execution. 4 spectral invariants on 410M step143000, random control, absorb-the-label test on raw |Δ| (broken by heavy tails; later fixed).
- `phase1a_spectral_logtransform.py` — Absorb test re-run on log|Δ|.
- `phase1a_summary.json`, `phase1a_logtransform_summary.json` — numeric.
- `phase1a_findings_framing.md` — **CANONICAL LANGUAGE FOR ANY WRITEUP**. Locked 2026-04-24.

- `phase1b_B_stability.py` — QK_PR at Pythia 410M step 16000 and 64000, rank correlation with step 143k, causal ordering tests.
- `phase1b_B_stability_summary.json` — numeric.

- `phase1b_C_160m_invariants.py` — Full Phase 1a pipeline on 160M.
- `phase1b_C_160m_summary.json` — numeric.

- `phase1b_D_dense_scan.py` — 8 Paper 2 checkpoints × OV_PR + QK_PR × (410M, 160M) + step 0 control.
- `phase1b_D_dense_scan_summary.json` — numeric (per-checkpoint ρ for every invariant, both models).

- `phase1b_E_lottery_and_intraclass.py` — Two CPU tests before 1.4B pre-reg. Test 1 intra-class at step 1000 on 410M. Test 2 lottery vs specialization (Pythia step 0 + fresh random init).
- `phase1b_E_lottery_intraclass_summary.json` — numeric.

- `invariants_preregistration.md` — Pre-registration v1 (step 143000 test on 410M). Null result logged separately.
- `invariants_preregistration_v2_1p4b.md` — Pre-registration v2 for 1.4B. Five tests, four-tier primary, joint interpretation table.
- `phase1b_1p4b_verdict_report.md` — **LOCKED 1.4B verdict. Canonical narrative for paper.**

### In `paper/tables/`
- `per_class_dfe_{410m,160m}.csv` — per-class Student-t fit parameters
- `per_class_dfe_pairwise_tests.csv` — bootstrap pair-tests
- `robust_threshold_sweep.csv` — AIC structure across classification thresholds
- `robust_time_resolved.csv` — per-class trajectory
- `robust_class_count_matching.csv` — class-size controlled comparisons
- `phase1a_spectral_invariants.csv` — 410M step 143k per-head table
- `phase1b_B_stability_step{16000,64000}.csv` — 410M earlier checkpoints
- `phase1b_C_160m_invariants.csv` — 160M step 143k per-head
- `phase1b_D_dense_scan.csv` — all checkpoints × heads × invariants × |Δ|

### In `paper/figures/`
- `per_class_dfe_nu_trajectories.pdf` — ν(t) per class, 410M + 160M
- `per_class_dfe_final_kde.pdf` — Student-t density fits
- `robust_threshold_sweep.pdf` — 2×2 Δ_AIC panel
- `robust_time_resolved.pdf` — Δ_AIC(t) per class per model
- `phase1a_scatter_primary.pdf` — OV_PR vs |Δ| (the primary null scatter)
- `phase1a_scatter_all_invariants.pdf` — 2×3 grid, all invariants
- `phase1b_D_dense_scan.pdf` — curves ρ(invariant, |Δ|) across checkpoints

### In `paper/data/`
- `all_ablations.csv` — Paper 2 410M ablation sweep
- `tier2_t21_scaling_160m.csv` — 160M from Tier 2 T2.1
- `tier2_pre1p4b_ablations.csv` — 1.4B 7 checkpoints × 144 heads
- `tier2_pre1p4b_spectral.csv` — 1.4B 8 checkpoints × 144 heads × 4 invariants
- `tier2_pre1p4b_verdict.json` — 1.4B pre-registered verdict

### Colab notebook
- `tier2_pre1p4b_validation.ipynb` — one-click reproducible 1.4B run

## 6. Open tracks (none active, all wait on user directive)

1. **Block 2 — Pruning behavioral test (GPU).** Designed in `invariants_preregistration_v2_1p4b.md` §Block 2. Ablate bottom-20 QK_PR heads vs top-20 vs random, measure perplexity. Hardens Finding B. Est. 30–60 min A100, ~$2.

2. **Block 3 — Hessian diagonal (GPU).** Compute diagonal of Hessian per head on validation batch; correlate with |Δ|. Test whether curvature-based invariant outperforms spectral-based. Est. 2–4 h A100.

3. **Phase 5 — Biology analog.** Per-gene effective rank on gene-expression covariance matrix; predict DepMap Achilles DFE tail weight. User is designing this plan separately as of 2026-04-24.

4. **Section 5 writeup.** Deferred. Canonical language in `phase1a_findings_framing.md` and `phase1b_1p4b_verdict_report.md` prevents drift.

5. **Remaining Tier 2.** T2.2 (Pile cross-dataset), T2.3 (inverse-meta), T2.4 (self-modeling 1.4B). Notebooks built, awaiting GPU to run.

6. **Fix in T2.1 analyze()** — loose threshold gave 90 emergent / 40 born (ratio 2.25). Strict Paper 2 criteria gives 56 emergent / 6 born (ratio 9.33). The fix is to use the classification from `per_class_dfe_robustness.py`. Low-priority cleanup.

## 7. Conceptual frame to not lose

The investigation started from a simple question — does a static head-level invariant predict DFE tail weight better than the training-dynamics category "emergent". Pre-reg v1 was negative: OV_PR at the **final** checkpoint explains little. Dense scan revealed a two-phase structure: pre-step-2000 the spectral–criticality correlation is **negative** (concentrated circuits → heavy tails); post-step-4000 it **flips** (spread circuits → heavy tails). Step 1000 sits in the negative-correlation regime. The primary predictive window is therefore the first ~0.7 % of training. This replicates across three parameter scales, is training-driven (not lottery ticket), and the sign flip is universal with the same temporal window.

The scale dependence is informative, not damning: at 1.4B the within-class gradation disappears (classes become more homogeneous in OV_PR), but the aggregate signal stays (ρ = −0.48) because the between-class contrast strengthens (QK gap doubles to +41). The invariant operates through between-class discrimination at larger scales and via within-and-between-class combined at smaller scales.

VWKK framework context: heavy-tail concentration is consistent with their prediction E2 (frustrated landscape). The two-phase structure is novel and not in their framework. The biology analog (Phase 5) would be direct substrate-independence test.

## 8. Restoration instructions for future session

To continue from this state:
1. Read this file first.
2. Read `analyses/phase1a_findings_framing.md` — locked canonical language.
3. Read `analyses/phase1b_1p4b_verdict_report.md` — three-scale summary.
4. Read `analyses/invariants_preregistration_v2_1p4b.md` — pre-reg details + four-tier rules.
5. Check memory file `~/.claude/projects/-Users-teo/memory/project_dfe_pilot.md`.
6. Ask user for next directive. Do not start new experiments without explicit green light.

---

*Session ended 2026-04-24 with 1.4B validation complete, 4/5 pre-registered tests PASS, paper writeup deferred, biology validation plan to be drafted by user.*
