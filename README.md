# Functional Differentiation Generates Universal Fitness-Effect Distributions in Neural Networks

Paper repository: code, data, and reproduction notebook for the study of DFE emergence across the full training trajectory of Pythia 410M.

## What this is

Ablation studies of 144 identity-tracked attention heads, 24 transformer layers, and 30 distributed-noise perturbations at each of 8 checkpoints (step 512 → step 143,000) of [Pythia 410M-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped). 1,584 ablations total.

Results summarized in the paper (`main.pdf` after compile) and in figures under `figures/`. Raw data in `data/all_ablations.csv`.

## Reproduce

```bash
# Clone
git clone https://github.com/mool32/functional-differentiation-dfe
cd functional-differentiation-dfe

# Option A: Run on Google Colab (recommended)
# Upload main_pilot_colab.ipynb to Colab, connect A100 runtime, Run all.
# Expected runtime: ~82 minutes on A100 40 GB. Cost: ~$4 in Colab Pro.

# Option B: Run locally (requires GPU)
pip install -r requirements.txt
python -c "from reproduce import run_all; run_all()"
```

All random seeds are fixed (seed 42 for bootstrap, 42000+i for Type A perturbations). Deterministic to float32 precision.

## Data format

`data/all_ablations.csv` — 1,584 rows, 10 columns:

| Column | Type | Description |
|--------|------|-------------|
| `checkpoint` | int | Training step (512, 1000, 2000, 4000, 8000, 16000, 64000, 143000) |
| `perturbation_type` | str | `head`, `layer`, or `type_a` |
| `subtype` | str | `output_zeroing`, `full_zeroing`, `noise_alpha1.0` |
| `layer_idx` | int | Transformer layer (0–23) |
| `head_idx` | int | Attention head (0–15), or -1 for layer/type_a |
| `seed` | int | PRNG seed for type_a only (else -1) |
| `baseline_loss` | float | Cross-entropy on wikitext-103 (25 batches × 4 × 2048 tokens) |
| `perturbed_loss` | float | Cross-entropy after ablation |
| `delta` | float | Fitness effect: `-(perturbed - baseline) / |baseline|` |
| `elapsed_sec` | float | Wall-clock time for this ablation |

## Figures

All 7 main-text figures and 4 supplementary figures plus one table are in `figures/` as PDF and PNG:

- `fig1_head_trajectories` — 144 head trajectories, 4-class classification (born-critical / emergent / growing / never-critical)
- `fig2_differentiation_metrics` — Gini, Effective N, differentiation count across training
- `fig3_dfe_emergence` — DFE histograms at 4 representative checkpoints
- `fig4_distribution_fits` — ΔAIC + β with bootstrap CI
- `fig5_layer_heatmap` — Layer ablation magnitude heatmap
- `fig6_hierarchy` — Three perturbation granularities at step 143,000
- `fig7_L8H9_crystallization` — L8H9 single-head emergence
- `figS1_lorenz_curves` — Lorenz curve overlap across checkpoints
- `figS2_threshold_robustness` — Differentiation count sensitivity to threshold choice
- `figS3_regime_sensitivity` — Head df stable vs layer df collapses under outlier removal
- `figS4_gini_triviality` — Empirical Gini vs Student's t-fit-predicted Gini
- `tableS1_top10_heads` — Top-10 most critical heads at step 143,000

## Compile the paper

Requires `pdflatex` and `bibtex`, plus the NeurIPS 2024 style file (`neurips_2024.sty`) downloadable from [NeurIPS website](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles).

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`.

## Author

Theodor Spiro (independent researcher). Contact: theospirin@gmail.com.

## Citing

If this work is useful to you, please cite:

```bibtex
@article{spiro2026functional,
  author    = {Spiro, Theodor},
  title     = {Functional Differentiation Generates Universal Fitness-Effect Distributions in Neural Networks},
  journal   = {arXiv preprint},
  year      = {2026},
}
```

Paper 1 that this work extends:

```bibtex
@article{spiro2026universal,
  author    = {Spiro, Theodor},
  title     = {Universal Statistical Signatures of Evolution in Artificial Intelligence Architectures},
  journal   = {arXiv preprint arXiv:2604.10571},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.10571},
}
```

## Acknowledgements

Analysis was conducted with extensive use of Claude (Anthropic) as a thinking and implementation partner. Computation was performed on Google Colab Pro (NVIDIA A100 40 GB), total runtime 82 minutes, cost ≈ $4.

## License

Code: MIT. Data (`all_ablations.csv`) and figures: CC-BY 4.0.
