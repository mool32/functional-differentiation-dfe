## 2 Methods

### 2.1 Model and checkpoints

All experiments use Pythia 410M-deduped \citep{biderman2023pythia}, retrieved from the HuggingFace repository \texttt{EleutherAI/pythia-410m-deduped}. The model has 24 transformer layers, 16 attention heads per layer, hidden size 1024, head dimension 64, and MLP intermediate size 4096. Pythia publishes 154 training checkpoints per model; we measure at 8 log-spaced checkpoints: step 512, 1,000, 2,000, 4,000, 8,000, 16,000, 64,000, and 143,000 (final). Each checkpoint is accessed via the \texttt{revision} parameter of HuggingFace \texttt{AutoModelForCausalLM.from\_pretrained()}, \texttt{revision=step\{N\}}.

### 2.2 Perturbation protocols

**Head ablation (primary protocol).** We zero the columns of the attention output projection (\texttt{attention.dense.weight}, shape $[d_\text{model}, d_\text{model}]$) corresponding to the head's output slice. For head $h$ in layer $\ell$, we set $W_\text{dense}^{(\ell)}[:,\, h \cdot d_h : (h+1) \cdot d_h] = 0$, where $d_h = 64$. This is equivalent to masking the contribution of head $h$ to the residual stream while leaving its internal QKV computation intact; we implement it as weight modification to integrate cleanly with save/restore. Heads are sampled systematically: $h \in \{0, 3, 6, 9, 12, 15\}$ at each of the 24 layers, for a total of 144 fixed heads. The same 144 heads are ablated at every checkpoint to enable identity-tracked per-head trajectories (cf. Fig.~\ref{fig:head_trajectories}).

**Layer ablation.** We zero all parameters in one transformer block: attention QKV projection, attention output projection, MLP up/down projections, and layer-norm weights and biases. All 24 layers are ablated exhaustively at each checkpoint.

**Distributed noise (Type A).** Gaussian noise of standard deviation $\alpha \cdot \sigma_p$ is added to every parameter tensor $p$ in one transformer block, where $\sigma_p$ is the per-tensor standard deviation of the original weights and $\alpha = 1.0$ in the main experiment. At each checkpoint we apply 30 independent noise samples targeting block index $i \bmod 24$ with deterministic PRNG seed $42{,}000 + i$ for $i \in \{0, \ldots, 29\}$.

### 2.3 Fitness metric and baseline evaluation

The fitness effect of a perturbation is
$$\Delta = -\frac{\ell_\text{perturbed} - \ell_\text{baseline}}{|\ell_\text{baseline}|},$$
with the sign chosen so that positive $\Delta$ corresponds to reduced loss (beneficial) and negative $\Delta$ to increased loss (deleterious), matching the convention of Paper~1.

Losses are mean cross-entropy over 25 validation batches of 4 sequences × 2,048 tokens = 204,800 tokens total. The validation set is streamed from \texttt{wikitext-103-raw-v1} (HuggingFace \texttt{wikitext/wikitext-103-raw-v1}, \texttt{train} split), tokenized with the Pythia tokenizer, and concatenated before reshaping into the batched tensor. We use the \texttt{train} split rather than \texttt{validation} because the latter (~250K tokens after filtering short examples) is tight for stable 204K-token baselines; Pythia was trained on the Pile, not on wikitext, so train/test leakage from the perspective of our measurement is minimal. Our analysis depends on loss *shifts* induced by ablations, not on absolute loss values, and the relative shifts are what the results in Sec.~\ref{sec:results} rely on. The identical token tensor is saved to disk once and loaded for every model load, guaranteeing that all $\Delta$ values are computed against the same token stream. **Baseline loss is recomputed for each checkpoint** on this identical batch set; no cross-checkpoint normalization is applied beyond the per-checkpoint division by $|\ell_\text{baseline}|$ in the definition of $\Delta$. This ensures that $\Delta$ values within a checkpoint are directly comparable, and that $\Delta$ values across checkpoints are comparable *as relative shifts*, since the $|\ell_\text{baseline}|$ factor (which varies from 6.83 at step 512 to 2.89 at step 143,000) is absorbed into the normalization.

### 2.4 Precision and correctness verification

All forward passes are in float32 with TF32 matmul enabled on A100 (\texttt{torch.backends.cuda.matmul.allow\_tf32 = True}). Earlier pilot experiments in float16 showed substantial numerical noise at early checkpoints where baseline loss is large; float32 is essential for clean signal at step 512.

Save/restore correctness is verified by SHA-256 checksum of the affected weight tensor before and after each restoration. Prior to the main sweep we verified bitwise identity for all three perturbation types (head, layer, Type~A) on a random reference layer. During the sweep, a cheaper drift monitor computes the absolute sum of a reference tensor every 30 ablations and asserts drift $< 10^{-3}$ against its initial value. The $10^{-3}$ threshold is generous --- approximately $1000\times$ above expected float32 numerical drift in tensor-sum operations --- chosen to catch genuine restore bugs while tolerating natural floating-point non-determinism. All 1,584 ablations of the main sweep completed with zero observed drift to float32 precision.

### 2.5 Metrics and threshold justification

**Primary metrics (threshold-free).** Gini coefficient of $|\Delta|$, computed on the sorted absolute values. Effective number of important heads, $N_\text{eff} = 2^H$ where $H = -\sum_i p_i \log_2 p_i$ is the Shannon entropy of $p_i = |\Delta_i| / \sum_j |\Delta_j|$ (the Hill number of order 1; \citealp{hill1973diversity, jost2006entropy}).

**Secondary metrics.** Differentiation count $= \#\{h : |\Delta_h| > \tau\}$ with $\tau = 5 \times 10^{-4}$. Distribution fits: Normal, Laplace, Student's $t$ via \texttt{scipy.stats.\{norm, laplace, t\}.fit()}, compared by $\Delta\text{AIC}$ relative to Normal. Gamma shape $\beta$ fit to the deleterious tail $\{-\Delta : \Delta < -10^{-4}\}$ with \texttt{scipy.stats.gamma.fit(neg, floc=0)}; the lower-bound filter at $10^{-4}$ excludes values below the numerical floor.

**Threshold justification.** The threshold $\tau = 5 \times 10^{-4}$ for differentiation count was chosen to exclude measurements below approximately $3\sigma$ of the numerical precision floor in float32 evaluation of the baseline loss across 25 batches (empirically $\sigma_\text{eval} \approx 1.5 \times 10^{-4}$, measured by re-evaluating the same model at identical seed across independent batch shufflings). All threshold-based claims in Secs.~\ref{subsec:emergence}--\ref{subsec:dual} are robust to threshold choice in the range $[10^{-4}, 10^{-3}]$. Fig.~S2 reproduces per-checkpoint differentiation counts at threshold values $\{10^{-4}, 5 \times 10^{-4}, 10^{-3}\}$; the class-count claims in Sec.~\ref{subsec:emergence} (4.2\% born-critical, 38.9\% emergent, 22.9\% growing, 34.0\% never-critical) shift by at most $\pm 2$ percentage points across this range, with qualitative conclusions preserved.

### 2.6 Statistical procedures

Bootstrap confidence intervals use 2,000 resamples with fixed seed 42; 95\% intervals are reported as the (2.5, 97.5) percentiles of the bootstrap distribution. Sensitivity analyses (Sec.~\ref{subsec:hierarchy}) refit the distribution after removing the $k$ most extreme $|\Delta|$ observations for $k \in \{1, 3, 5, 10\}$ (heads) or by specific identified outliers L0 and L5 (layers); we report the Student's $t$ $df$ as the shape-stability indicator, since the AIC advantage over Normal is inherently sample-size-dependent while the fitted $df$ is not. All random seeds (42 for bootstrap, $42{,}000 + i$ for Type~A noise samples) are arbitrary but fixed; deterministic reproduction of all numerical results is enabled.

### 2.7 Code and data availability

All 1,584 ablations are released as CSV with the following fields: \texttt{checkpoint}, \texttt{perturbation\_type}, \texttt{subtype}, \texttt{layer\_idx}, \texttt{head\_idx}, \texttt{seed}, \texttt{baseline\_loss}, \texttt{perturbed\_loss}, \texttt{delta}, \texttt{elapsed\_sec}. Code, data, and a self-contained Colab notebook reproducing the main sweep (expected runtime: 82 minutes on A100 40 GB) are available at \url{https://github.com/mool32/functional-differentiation-dfe}.
