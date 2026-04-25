# DFE Research — Master Index

**Last updated:** 2026-04-25
**GitHub:** https://github.com/mool32/functional-differentiation-dfe
**Author:** Theodor Spiro (theospirin@gmail.com), Paper 1 arXiv 2604.10571

This is the single index across the entire project. Three companion state docs go deeper on specific tracks:
- `SESSION_STATE_2026_04_24.md` — ML Phase 1 invariant search
- `SESSION_STATE_BIOLOGY_2026_04_24.md` — Biology cross-substrate attempts
- `analyses/phase1a_findings_framing.md` — Locked canonical language

---

## 1. The story in three sentences

We tested whether a structural early-training invariant predicts head ablation effects in transformer models, and whether the same invariant crosses to biological substrates. Across five language model checkpoints from three independent teams, the invariant replicates with consistent magnitude (ρ ∈ [−0.55, −0.42]). Five biological pre-registered tests under strict rules failed; the ML invariant is now characterized as universal within transformer training but not extending to the biology operationalizations attempted.

## 2. Active project state

**ML track (5 scales × 3 teams):**
- Pythia 160M / 410M / 1.4B (EleutherAI, GPT-NeoX, Pile): PRE-REG PASS
- OLMo-2 1B (AllenAI, OLMo-2, Dolma): PRE-REG PASS
- TinyLlama-1.1B (Singapore U, Llama, SlimPajama+StarCoder): **RUNNING NOW**

**Biology track:** CLOSED honestly under strict pre-registration. Five operationalizations all FAIL or HOLD.

**Paper scope decision:** ML-only paper. Awaiting TinyLlama verdict before writeup.

## 3. ML findings — locked

### 3.1 Headline numerical result

| model | scale | architecture | training data | ρ(OV_PR @early, |Δ| @same) | n |
|-------|-------|--------------|---------------|-----------------------------|---|
| Pythia 160M-deduped | 160M | GPT-NeoX | Pile | **−0.416** | 144 |
| Pythia 410M-deduped | 410M | GPT-NeoX | Pile | **−0.555** | 144 |
| Pythia 1.4B-deduped | 1.4B | GPT-NeoX | Pile | **−0.484** | 144 |
| OLMo-2 1B-early | 1B | OLMo-2 | Dolma | **−0.487** | 256 |
| TinyLlama-1.1B | 1.1B | Llama | SlimPajama | (running) | 176 |

All values from same-checkpoint correlation at step ≈ 0.7% of training (Pythia step 1000 / OLMo step 1000 / TinyLlama step 10k).

### 3.2 Two-phase trajectory replicates

In all four completed scales, ρ(OV_PR_t, |Δ|_t) goes:
- step 0–2k: peak negative ρ (−0.5 to −0.7)
- sign flip between step 4k and step 8k (uniformly, all four scales)
- step 8k+: positive ρ (+0.1 to +0.5)

Universal training-step-localized phase transition.

### 3.3 QK active-vs-dead discrimination scales with model

QK_PR gap (active classes − never-critical class) at final checkpoint:
- 160M: +17
- 410M: +20
- 1.4B: +41 (doubles)

Discrimination is training-driven (step-0 gap = 0 in 160M control).

### 3.4 Pre-registration commit chain

| pre-reg | scope | locked | verdict |
|---------|-------|--------|---------|
| v1 (`f99769b`) | Phase 1a primary OV_PR @ step143k 410M | 2026-04-24 | NULL |
| v2 (`8be0f11`) | 1.4B 5-test joint | 2026-04-24 | 4/5 PASS |
| v3 (`171aacd`) | OLMo-2 1B cross-architecture | 2026-04-24 | PASS |
| v6 (`7523931`) | TinyLlama-1.1B cross-team | 2026-04-25 | running |

## 4. Biology findings — closed under strict rules

All five attempts pre-registered and locked before data access. Six binding rules accepted (Direction absolute, Single primary, Numeric thresholds locked, Null legitimate, No post-hoc rescue, Pre-reg commit hash baked).

| pre-reg | substrate | unit | predicted | observed | verdict |
|---------|-----------|------|-----------|----------|---------|
| v1 (`1b01020`) | Schiebinger reprogramming cells | per-cell gene-PR | ρ < 0 | ρ = −0.34 fragile, methodology-null marginal | MARGINAL |
| v3 (`a8e7ea3`) | Bastidas-Ponce pancreas cells | per-cell module-PC1 | ρ < 0 | ρ = −0.25, gate FAIL | HOLD |
| v4 (`7f4c638`) | SARS-CoV-2 RBD viral protein | per-residue contact-PR | ρ < 0 | **ρ = +0.49** strong wrong-direction | **FAIL_WRONG_DIRECTION** |
| v5 (`5308efc`) | ProteinGym viral subset | per-protein spectral-PR | ρ < 0 | ρ = +0.09, gate FAIL | HOLD |
| pre-reg v2 (`467e792`) | Schiebinger module entropy | per-cell entropy | ρ < 0 | ρ ≈ 0 | NULL |

### Cross-substrate claim status

**Under Rules 1–6: not supported.** Five operationalizations across cell-level and protein-level units, weight-space-style and population-distribution-style measurements, all failed pre-registered direction.

### Independent biology observations (separate from cross-substrate claim)

These were observed but require their own pre-registration to validate:
- Per-module PR × K562 perturbability: ρ = +0.6 across 4 dataset combinations, developmental-specific (HPA control = 0.15), driven by stress/growth/sensor modules.
- RBD non-RBM scaffold residues: ρ = +0.78 between contact-PR and fitness impact.
- Schiebinger sign-flip trajectory across reprogramming days.

Each is a future-work direction, not validation of ML invariant.

## 5. Where everything lives

### 5.1 Repository structure (paper/)

```
paper/
├── PROJECT_MASTER_INDEX.md            ← this file
├── SESSION_STATE_2026_04_24.md        ← ML phase 1 deep history
├── SESSION_STATE_BIOLOGY_2026_04_24.md  ← biology deep history
├── main.tex                            ← Paper 2 LaTeX (Tier 1 integrated)
├── replication_plan.md                 ← Tier 1/2/3 plan (locked 2026-04-20)
│
├── analyses/                           ← all analysis scripts + summaries
│   ├── invariants_preregistration.md         (v1)
│   ├── invariants_preregistration_v2_1p4b.md
│   ├── invariants_preregistration_v3_olmo.md
│   ├── invariants_preregistration_v4_rbd.md
│   ├── invariants_preregistration_v5_proteingym_viral.md
│   ├── invariants_preregistration_v6_tinyllama.md
│   ├── biology_preregistration_v1_schiebinger.md
│   ├── biology_preregistration_v2_module_entropy.md
│   ├── biology_preregistration_bastidas_ponce.md
│   ├── biology_validation_plan.md            (user-supplied original plan)
│   ├── phase1a_findings_framing.md           (locked canonical language)
│   ├── phase1b_1p4b_verdict_report.md
│   ├── per_class_dfe.py + summary.json
│   ├── per_class_dfe_robustness.py + summary.json
│   ├── phase1a_spectral.py + summary.json
│   ├── phase1a_spectral_logtransform.py + summary.json
│   ├── phase1b_B_stability.py + summary.json
│   ├── phase1b_C_160m_invariants.py + summary.json
│   ├── phase1b_D_dense_scan.py + summary.json
│   ├── phase1b_E_lottery_and_intraclass.py + summary.json
│   ├── biology_schiebinger_*.py (10 scripts) + summaries
│   ├── biology_bastidas_ponce_primary.py + summary.json
│   ├── biology_per_module_analog.py + summary.json
│   ├── biology_steady_state_control.py + summary.json
│   ├── biology_mechanism_probe.py + summary.json
│   ├── biology_rbd_primary.py + summary.json
│   ├── biology_proteingym_viral.py + summary.json
│   ├── tier2_olmo_verdict.json
│   └── ...
│
├── tables/                             ← per-head/per-cell/per-protein output
│   ├── all_ablations.csv               (Pythia 410M, Paper 2 main pilot, 1584 rows)
│   ├── tier2_t21_scaling_160m.csv      (160M 1488 rows)
│   ├── tier2_pre1p4b_ablations.csv     (1.4B 1008 rows)
│   ├── phase1a_spectral_invariants.csv (410M per-head, step143k)
│   ├── phase1b_*.csv                   (per-checkpoint per-head)
│   ├── biology_schiebinger_*.csv       (per-cell tables)
│   ├── biology_rbd_per_residue.csv
│   ├── biology_proteingym_viral.csv
│   ├── per_class_dfe_*.csv
│   ├── robust_*.csv
│   └── ...
│
├── data/
│   ├── all_ablations.csv               (Paper 2 410M canonical)
│   ├── tier2_t21_scaling_160m.csv
│   ├── tier2_pre1p4b_*.csv             (Pythia 1.4B raw)
│   ├── tier2_olmo_*.csv                (OLMo-2 1B raw)
│   ├── micropilot/question_bank.json   (Paper 3 self-modeling 29 questions)
│   ├── tier1_summary.json              (Paper 2 Tier 1 4 checks)
│   ├── rbd_cache/                      (PDB 6M0J + Bloom DMS)
│   ├── proteingym_cache/               (ProteinGym v1 viral)
│   └── ...
│
├── figures/                            ← all PDF figures
│   ├── fig1-7 + figS1-S4 (Paper 2 main figures)
│   ├── per_class_dfe_*.pdf
│   ├── robust_*.pdf
│   ├── phase1a_*.pdf, phase1b_*.pdf
│   ├── biology_*.pdf
│   └── ...
│
└── notebooks (Colab-ready)
    ├── main_pilot_colab.ipynb          (Paper 2 reproduction)
    ├── tier1_replication.ipynb         (T1.1-T1.4 in one session)
    ├── micropilot_ablation_sweep.ipynb (Paper 3 self-modeling)
    ├── tier2_t21_scaling.ipynb         (160M + 1.4B)
    ├── tier2_t22_crossdataset.ipynb    (Pile, not run yet)
    ├── tier2_t23_inverse_meta.ipynb    (Paper 3 inverse-meta)
    ├── tier2_t24_self_14b.ipynb        (Paper 3 1.4B self-modeling)
    ├── tier2_pre1p4b_validation.ipynb  (1.4B pre-reg v2)
    ├── tier2_olmo_validation.ipynb     (OLMo-2 1B pre-reg v3)
    └── tier2_tinyllama_validation.ipynb (TinyLlama pre-reg v6)
```

### 5.2 Key commit chain

| commit | content |
|--------|---------|
| `b69cfbf` | Paper 2 Tier 1 integrated, arXiv-ready pending compile |
| `056913f` | Tier 2 notebooks T2.1–T2.4 |
| `bd0e643` | Block 1.2 per-class DFE + robustness pipeline |
| `f99769b` | Pre-registration v1 (Phase 1a) |
| `51d5341` | Phase 1a result: primary NULL |
| `0b4a7b3` | **Locked framing doc** (canonical language) |
| `c1d3382` | Phase 1b-B/C: stability + 160M |
| `d341c99` | **Phase 1b-D dense scan** (sign-flip discovery) |
| `10316a4` | Phase 1b-E + pre-reg v2 |
| `8be0f11` | Pre-reg v2 amendment (weak tier) |
| `3ddf146` | 1.4B Colab notebook |
| `a23562a` | **1.4B 4/5 PASS** |
| `13bc88d` | ML state snapshot |
| `1b010202` | Biology pre-reg v1 |
| `07c4020` | Schiebinger H1 PASS, H0 FAIL |
| `c8b398f` | Schiebinger null controls |
| `fe000cf` | Schiebinger robustness FAIL + binary outcome |
| `467e792` | Biology pre-reg v2 (module entropy) |
| `dbc2aea` | Module entropy null |
| `ed7b044` | Module variants V2/V3/V4a |
| `612c013` | V4a robustness + null gate fail |
| `a8e7ea3` | Biology pre-reg Bastidas-Ponce |
| `ecc23a1` | BP one-shot HOLD per pre-reg |
| `5a23abe` | Per-module biology test (4-way replication) |
| `b3f28df` | Steady-state HPA control |
| `9cda415` | Mechanism probe (stress/growth driver) |
| `219e1bf` | **Biology state snapshot** |
| `7f4c638` | Biology pre-reg v4 (RBD) |
| `fdef01e` | RBD FAIL_WRONG_DIRECTION |
| `5308efc` | Biology pre-reg v5 (ProteinGym viral) |
| `fffe45d` | ProteinGym v5 HOLD |
| `171aacd` | Pre-reg v3 (OLMo) |
| `bea4ffc` | OLMo PASS |
| `7523931` | Pre-reg v6 (TinyLlama) |
| `8e9c16b` | TinyLlama notebook hash baked |

### 5.3 External resources used

- **Pythia weights:** `EleutherAI/pythia-{160m,410m,1.4b}-deduped` on HuggingFace
- **OLMo-2 weights:** `allenai/OLMo-2-0425-1B-early-training`
- **TinyLlama weights:** `TinyLlama/tinyLlama-intermediate-checkpoints`
- **wikitext-103:** standard streaming via `datasets`
- **Schiebinger reprogramming:** local at `~/Desktop/research/perceptual_modules/paper6/results/schiebinger_scored.h5ad`
- **Bastidas-Ponce pancreas:** local at same path
- **HPA module activity:** local at `~/Desktop/research/perceptual_modules/power_law_test/results/phase0_baseline/activity_hpa_43.csv`
- **Dixit / Replogle K562 perturbability:** local at `~/Desktop/research/perceptual_modules/paper6/results/`
- **SARS-CoV-2 RBD DMS:** Bloom lab public GitHub `jbloomlab/SARS-CoV-2-RBD_DMS`
- **PDB 6M0J:** RCSB public
- **ProteinGym v1 DMS:** HuggingFace `OATML-Markslab/ProteinGym_v1` (parquet) + Marks lab AlphaFold structures zip

## 6. Outstanding work — what's left

### Tier 2 from replication plan (pending)
- **T2.2 Pile cross-dataset** on 410M (notebook ready, ~$3 GPU)
- **T2.3 Inverse meta-heads analysis** for Paper 3 (notebook ready, ~$1)
- **T2.4 Self-modeling replication on 1.4B** (notebook ready, ~$15 A100 80GB)

### Pythia 1.4B step-37000 missing
The 1.4B run completed 7 of 7 ablation checkpoints; OLMo run completed 5 of 7 (missing step 37000 final). Primary verdicts both PASS regardless. S-4 final-checkpoint gap remains incomplete on OLMo.

### Currently running
- TinyLlama-1.1B per pre-reg v6 (Colab one-click). Expected 3 hours.

### Future work (separate pre-registrations needed)
- Cross-architecture extension to Mamba / RWKV (state-space, no traditional OV)
- Biology RBD scaffold ρ=+0.78 follow-up (own pre-reg)
- Biology per-module developmental-stress observation (own pre-reg)
- Pythia 6.9B addition (one more scale)

## 7. Paper writeup status

**Decision:** ML-only paper. Awaiting TinyLlama verdict.

**Outline locked already:**
- Section 1 — Intro: early-window invariant question, prior work
- Section 2 — Methods: per-head OV_PR computation, ablation methodology, eval protocol
- Section 3 — Results:
  - 3.1 Cross-scale Pythia (160M/410M/1.4B)
  - 3.2 Cross-architecture OLMo + TinyLlama
  - 3.3 Two-phase trajectory replicates universally
  - 3.4 QK active-vs-dead discrimination scales with model
  - 3.5 Lottery ticket rejected (step-0 controls)
- Section 4 — Discussion + limitations:
  - Within-ML universality established
  - Biology cross-substrate not supported under tested operationalizations
  - Future work directions
- Section 5 (short) — Connection to evolution-learning framework (deferred or merged)

**Not yet written.** Awaiting verification.

---

*Master index updated 2026-04-25. State of work captured at this snapshot. For real-time state, check git log on the main branch.*
