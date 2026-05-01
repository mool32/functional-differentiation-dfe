[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC-BY 4.0](https://img.shields.io/badge/Data%20%26%20Manuscript-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Preprint](https://img.shields.io/badge/Preprint-Manuscript-orange)](main.tex)
[![Companion arXiv](https://img.shields.io/badge/Paper%201-arXiv%3A2604.10571-b31b1b.svg)](https://arxiv.org/abs/2604.10571)

# Functional Differentiation Generates Universal Fitness-Effect Distributions in Neural Networks

**1,584 ablations × 8 checkpoints × Pythia 410M: the DFE shape evolves from delta-peak-with-outliers to heavy-tailed Student's *t* through functional differentiation of attention heads — and a single head (L8H9) undergoes phase-transition-like specialization between steps 4k and 8k**

Theodor Spiro | [ORCID 0009-0004-5382-9346](https://orcid.org/0009-0004-5382-9346) | tspiro@vaika.org

📄 **Manuscript:** [`main.tex`](main.tex) (NeurIPS-style LaTeX source) · 11 figures + 1 table in [`figures/`](figures/) and [`tables/`](tables/)
🧮 **Main reproduction notebook:** [`main_pilot_colab.ipynb`](main_pilot_colab.ipynb) — full ablation sweep, ~82 min on A100 40 GB
🔁 **Replication suite:** [`tier1_replication.ipynb`](tier1_replication.ipynb) (seed / cross-dataset / bootstrap / drift) plus 6 Tier 2 notebooks (`tier2_*.ipynb`) covering scaling (Pythia 1.4B), cross-architecture (OLMo, TinyLlama), and inverse-meta validation
🧬 **Companion paper:** [Universal statistical signatures of evolution in AI architectures, Spiro 2026, arXiv:2604.10571](https://arxiv.org/abs/2604.10571) — the broader DFE-universality framework that this paper provides a generative mechanism for at the component level

---

## Brief Summary

Ablation studies in trained neural networks reveal a universal statistical form — heavy-tailed, negatively skewed distributions of fitness effects (DFE) — previously observed across biological and engineered adaptive systems. Whether this universality is an emergent property of training, or what generates it component by component, has not been characterized. We ablate **144 identity-tracked attention heads, 24 transformer layers, and 30 distributed-noise perturbations at each of 8 checkpoints** spanning the full training trajectory of Pythia 410M-deduped (step 512 → 143,000; 1,584 ablations total) and demonstrate that:

1. **The DFE shape evolves from a delta-peak-with-outliers regime at early training to a continuous heavy-tailed form.** By step 143,000 the median Gamma shape **β ≈ 0.6** falls within the biological range reported for *E. coli* and yeast.
2. **The evolution is driven by functional differentiation of attention heads.** **39%** of eventually-critical heads are *emergent* (grew out of noise during training); only **4%** were *born-critical*; the remaining 57% are *growing* or *never-critical*. The emergent class dominates born-critical 10:1.
3. **Differentiation has a dual structure.** The number of functional components grows (Spearman *ρ* = +1.00) while importance concentration grows simultaneously (*ρ* = −0.88 on effective *N*), with inequality (Gini) approximately invariant across training. We name this "dual differentiation": broaden + concentrate, with conserved inequality.
4. **Substrate-independence of the DFE form is conditional on discrete modularity.** Distributed parametric noise produces Gaussian DFE by CLT; head ablations produce Student's *t* with **df ≈ 2.3**; layer ablations produce super-Cauchy tails with **df ≈ 0.85**. The three perturbation granularities are qualitatively distinct regimes — not a continuous family.
5. **L8H9 is a concrete phase transition.** A single attention head undergoes phase-transition-like specialization between step 4,000 and 8,000 — observable as a Gini-/-eff_N break in the population statistics *and* as a nameable single-head event. Specialization is reproducible across seeds and persists through end of training; provides a concrete target for mechanistic interpretability.

These results characterize a single model (Pythia 410M); scaling and cross-architecture replication are the natural follow-up. The Tier 2 replication suite addresses this within the repository (Pythia 1.4B, OLMo-2 1B, TinyLlama-1.1B).

## Inputs

| Source | Use | Access |
|---|---|---|
| **Pythia 410M-deduped** (EleutherAI) | Primary model — 8 checkpoints from step 512 to 143,000 | [HuggingFace EleutherAI/pythia-410m-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped) |
| **Pythia 1.4B / 160M** | Scale-invariance replication (Tier 2) | HuggingFace |
| **OLMo-2 1B** (AllenAI, Dolma corpus) | Cross-architecture replication (Tier 2) | HuggingFace |
| **TinyLlama-1.1B** (SlimPajama + StarCoder) | Cross-architecture replication (Tier 2) | HuggingFace |
| **wikitext-103** | Loss evaluation corpus | HuggingFace `wikitext-103-raw-v1` |
| **C4 / OpenWebText** | Cross-dataset robustness | HuggingFace |

## Repository structure

```
├── main.tex                           # NeurIPS-style manuscript (520 lines)
├── references.bib                     # BibTeX bibliography
├── sections/                          # Manuscript sections (1_introduction.md ... 4_discussion.md)
├── abstract_and_outline.md            # Initial abstract + outline (kept for provenance)
├── figure_captions.md                 # All figure captions (single source of truth)
├── replication_plan.md                # The Tier 1 + Tier 2 replication plan
├── submission_checklist.md            # Living checklist for arXiv / NeurIPS submission
│
├── main_pilot_colab.ipynb             # Main ablation sweep (Colab A100, ~82 min)
├── tier1_replication.ipynb            # Tier 1: seed / cross-dataset / bootstrap / drift
├── tier2_t21_scaling.ipynb            # Tier 2.1: Pythia 1.4B scaling
├── tier2_pre1p4b_validation.ipynb     # Tier 2: pre-1.4B sanity checks
├── tier2_t22_crossdataset.ipynb       # Tier 2.2: cross-dataset robustness
├── tier2_t23_inverse_meta.ipynb       # Tier 2.3: inverse-meta head validation
├── tier2_t24_self_14b.ipynb           # Tier 2.4: self-modeling at 1.4B (Paper 3 connection)
├── tier2_olmo_validation.ipynb        # Tier 2: OLMo-2 1B
├── tier2_tinyllama_validation.ipynb   # Tier 2: TinyLlama-1.1B
├── micropilot_ablation_sweep.ipynb    # Initial micropilot (kept for provenance)
├── build_question_bank.ipynb          # Auxiliary
│
├── build_olmo_notebook.py             # Notebook builders (parametric)
├── build_1p4b_notebook.py
├── build_tinyllama_notebook.py
├── build_tier2_notebooks.py
│
├── analyses/                          # Pre-registrations and analysis scripts
│                                      # (invariants_preregistration v1-v6, biology
│                                      # validation suite, per-class DFE robustness)
├── data/                              # Ablation outputs and intermediates
├── figures/                           # 11 publication figures (PDF + PNG)
├── tables/                            # Manuscript tables
├── paper3/                            # Paper 3 (Self-Specific Attention Heads) outline —
│                                      # work-in-progress for a separate paper, will move
│                                      # to its own repo when ready
│
├── HANDOFF.md, RESUME_HERE.md, SESSION_STATE_*.md, PROJECT_MASTER_INDEX.md
│                                      # Process documentation — kept deliberately as a
│                                      # record of how the work unfolded session by session
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Figures

| Reference | File | Topic |
|---|---|---|
| Fig. 1 | `figures/fig1_head_trajectories` | 144 head trajectories, 4-class classification |
| Fig. 2 | `figures/fig2_differentiation_metrics` | Gini, effective *N*, differentiation count across training |
| Fig. 3 | `figures/fig3_dfe_emergence` | DFE histograms at 4 representative checkpoints |
| Fig. 4 | `figures/fig4_distribution_fits` | ΔAIC + β with bootstrap CI |
| Fig. 5 | `figures/fig5_layer_heatmap` | Layer ablation magnitude heatmap |
| Fig. 6 | `figures/fig6_hierarchy` | Three perturbation granularities at step 143,000 |
| Fig. 7 | `figures/fig7_L8H9_crystallization` | L8H9 single-head emergence |
| Fig. S1 | `figures/figS1_lorenz_curves` | Lorenz curve overlap across checkpoints |
| Fig. S2 | `figures/figS2_threshold_robustness` | Differentiation count sensitivity to threshold |
| Fig. S3 | `figures/figS3_regime_sensitivity` | Head df stable vs layer df collapses under outlier removal |
| Fig. S4 | `figures/figS4_gini_triviality` | Empirical Gini vs Student's *t* fit prediction |
| Table S1 | `tables/tableS1_top10_heads` | Top-10 most critical heads at step 143,000 |

## Reproducing the analysis

### Option A — Google Colab (recommended)

```bash
# Upload main_pilot_colab.ipynb to Colab, connect A100 40 GB runtime, Run all.
# Expected runtime: ~82 minutes for the main 1,584-ablation sweep.
# Tier 1 replication adds ~40 minutes; Tier 2 notebooks each ~3-5 hours.
```

### Option B — local (GPU required)

```bash
git clone https://github.com/mool32/functional-differentiation-dfe.git
cd functional-differentiation-dfe
pip install -r requirements.txt
jupyter notebook main_pilot_colab.ipynb
```

All random seeds are fixed (seed 42 for bootstrap, 42000+*i* for distributed-noise perturbations). Output is deterministic to float32 precision; commit history includes a 500-cycle drift verification.

## Data format

`data/all_ablations.csv` — 1,584 rows, 10 columns:

| Column | Type | Description |
|---|---|---|
| `checkpoint` | int | Training step (512, 1000, 2000, 4000, 8000, 16000, 64000, 143000) |
| `perturbation_type` | str | `head` / `layer` / `type_a` (distributed noise) |
| `subtype` | str | `output_zeroing` / `full_zeroing` / `noise_alpha1.0` |
| `layer_idx` | int | Transformer layer (0–23) |
| `head_idx` | int | Attention head (0–15), or −1 for layer / type_a |
| `seed` | int | PRNG seed (type_a only, else −1) |
| `baseline_loss` | float | Cross-entropy on wikitext-103 (25 batches × 4 × 2048 tokens) |
| `perturbed_loss` | float | Cross-entropy after ablation |
| `delta` | float | Fitness effect: `−(perturbed − baseline) / |baseline|` |
| `elapsed_sec` | float | Wall-clock time |

## Compiling the paper

Requires `pdflatex`, `bibtex`, and the NeurIPS 2024 style file (`neurips_2024.sty` from the [NeurIPS website](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles)):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

```bibtex
@article{spiro2026functionaldifferentiation,
  author = {Spiro, Theodor},
  title  = {Functional differentiation generates universal fitness-effect distributions in neural networks},
  year   = {2026},
  note   = {Manuscript in preparation. Companion paper: arXiv:2604.10571}
}
```

Companion (Paper 1):

```bibtex
@article{spiro2026aievolution,
  author        = {Spiro, Theodor},
  title         = {Universal statistical signatures of evolution in artificial intelligence architectures},
  year          = {2026},
  eprint        = {2604.10571},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2604.10571},
  url           = {https://arxiv.org/abs/2604.10571}
}
```

## Contact

Theodor Spiro — tspiro@vaika.org

## License

- **Code** (`*.ipynb`, `*.py`, `analyses/*.py`): MIT (see [LICENSE](LICENSE))
- **Data** (`data/*`): CC-BY 4.0
- **Figures** (`figures/*`, `tables/*`): CC-BY 4.0
- **Manuscript** (`main.tex`, `references.bib`, `sections/*`, `abstract_and_outline.md`, `figure_captions.md`): CC-BY 4.0
