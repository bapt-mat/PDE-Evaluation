# PDE-Evaluation

Code associated with the paper:

> **On the post-hoc Evaluation of PDE Discovery: A Multifaceted Challenge of Scientific
> Advancement**
> Baptiste Mathevon, Farah Cherfaoui, Amaury Habrard, Marc Sebban

<details>
<summary><b>Abstract</b></summary>
<br>

Partial differential equation (PDE) discovery aims to identify from data the
governing law of a physical system. Constituting a cornerstone of scientific advancement, it
has become during the past decade a major line of research in the rapidly evolving field of
Physics-informed Machine Learning (PiML). Among the remaining open problems to address in this
domain, the post-hoc evaluation of discovered PDEs raises the particular difficulty of being
multifaceted. Indeed, it requires jointly considering predictive accuracy, physical
consistency, interpretability, and out-of-distribution generalization capacity. Given that some
of these properties are conflicting, it is worth noting that the wide range of existing
evaluation metrics only partially address the overall problem, potentially leading to overly
interpreted conclusions about the validity of a presumed new physical theory. From an abundant
literature spanning machine learning, numerical analysis, information theory or symbolic
regression, we propose, to our knowledge, the first taxonomy of PDE evaluation metrics, and
discuss their advantages and limitations in depth. Based on the observation that evaluation is
often achieved on a case-by-case basis and that a universally accepted methodology remains
elusive, we further provide recommendations with the aim of promoting standardized and reliable
practices, before sketching promising future lines of research in this field. We argue that
this paper is intended both for ML experts who design new PDE discovery algorithms and for
users of these methods aiming, in real applications, to discover and validate well-founded
scientific laws.

</details>

Post-hoc evaluation of PDE discovery: a suite of metrics and diagnostic plots for assessing
the quality of a *discovered* PDE against a known ground-truth PDE.

Given a candidate equation, e.g. $u_t = -u u_x + 0.05 u_{xx}$, this repository simulates it,
compares it against the reference dynamics, and reports quantitative metrics (accuracy,
sparsity, coefficient recovery, generalization, trade-off scores) together with diagnostic
figures (solution fields, error heatmaps, spectral error, convergence curves).

Three reference systems are provided: **Burgers' equation**, the **Korteweg-de Vries (KdV)
equation**, and the **Kuramoto-Sivashinsky (KS) equation**.

### Open questions in the post-hoc evaluation of PDE discovery

![Open questions in the post-hoc evaluation of PDE discovery](assets/Questions.png)

## Repository structure

```
utils.py                       ETD1 spectral solver and PDE-file parsing, shared by all entry points
simulation.py                  entry point for figure generation
evaluation/evaluation.py       entry point for metric computation
evaluation/errors.py           accuracy metrics (MSE, nMAE, rollout error, ...)
evaluation/coef.py             coefficient-recovery metrics (NDCG, Tanimoto, ...)
evaluation/sparsity.py         sparsity and structural-complexity metrics
evaluation/tradeoff.py         accuracy-sparsity trade-off metrics (AICc, BIC, MDL, ...)
evaluation/generalization.py   out-of-distribution and numerical-convergence metrics
evaluation/shared.py           helper routines shared across metric modules
evaluation/sensitivity.py      term-sensitivity weighting used by the wL2coef metric
```

## PDE file format

A PDE is specified as a plain-text file listing its terms and coefficients:

```
u_xx: 0.05
u·u_x: -1.0
```

which corresponds to $u_t = 0.05 u_{xx} - u u_x$. Supported terms are `u`, `u_x`, `u_xx`,
`u_xxx`, `u_xxxx` (linear terms), products of two such terms (quadratic terms, e.g. `u·u_x`),
and `x·u_xx`-style terms (a linear term multiplied by the spatial coordinate). A constant
offset may be specified with `b: <value>`.

Files are expected under `pde_files/<system>/`, named `<system>_pde_<tag>.txt`, e.g.
`pde_files/ks/ks_pde_true.txt` for the ground truth and `pde_files/ks/ks_pde_A.txt` for a
candidate. This convention is not required — files may be given by full path — but it is what
enables the `--file <system>` shortcut described below.

## Computing metrics

```
cd evaluation
python3 evaluation.py --file ks_pde_A.txt --metric all
```

This simulates the ground truth (`ks_pde_true.txt`, resolved automatically from the same
directory) and the candidate `ks_pde_A.txt`, then reports every registered metric.

Multiple candidates may be evaluated against the same ground truth in one call:

```
python3 evaluation.py --file ks_pde_A.txt ks_pde_B.txt --metric all
```

Individual metrics can be selected in place of `all`, either by name or by module (which
expands to all metrics defined in that module):

```
python3 evaluation.py --file ks_pde_A.txt --metric MSE rollout Sterms
python3 evaluation.py --file ks_pde_A.txt --metric errors coef
```

Some metrics require additional arguments; the script reports which one is missing if omitted:

```
python3 evaluation.py --file ks_pde_A.txt ks_pde_B.txt --metric Sterms --theta_size 5
python3 evaluation.py --file ks_pde_A.txt ks_pde_B.txt --metric Score --baseline_file ks_pde_A.txt
```

If a single system name is given instead of a filename, every candidate found for that system
is evaluated against its ground truth:

```
python3 evaluation.py --file ks --metric all --theta_size 5 --baseline_file ks_pde_A.txt
```

Additional options include `--dt_factor` (finer internal time-stepping), `--T_max`
(simulation horizon), `--k_max` (spectral cutoff for fMSE), `--ic` (choice of out-of-distribution
initial condition for `IC_nMAE`), and `--n_refine` (number of refinement steps for
`Conv_t`/`Conv_x`). See `python3 evaluation.py --help` for the complete list.

## Generating figures

```
python3 simulation.py --file ks_pde_A.txt
```

Produces a side-by-side comparison of the ground truth and candidate solution fields, a
solution heatmap, and an error heatmap, written to `simulations/` and `errors/`.

Example rollout-error (`--rollout`) and Fourier-space error (`--fourier`) plots for two KS candidates:

<p>
<img src="assets/rollout_ks_pde_A.png" width="25%">
<img src="assets/rollout_ks_pde_B.png" width="25%">
<img src="assets/fourier_ks_pde_A.png" width="22%">
<img src="assets/fourier_ks_pde_B.png" width="22%">
</p>

Additional options:

```
python3 simulation.py --file ks_pde_A.txt ks_pde_B.txt --eq_title      # annotate each panel with its equation
python3 simulation.py --file ks_pde_A.txt --rollout --rollout_t 300    # rollout-error plot, reported at t=300
python3 simulation.py --file ks_pde_A.txt --fourier --fourier_log      # spectral (Fourier-space) error heatmap
python3 simulation.py --file ks_pde_A.txt --conv --n_refine 6          # numerical convergence plots
python3 simulation.py --file ks --clean                                # all candidates for a system, unlabeled axes
```

## Requirements

`numpy`, `matplotlib`. `evaluation/sensitivity.py` additionally requires `jax`, used only by
the `wL2coef` metric.

## List of metrics

Notation follows the paper: $u$/$\hat u$ are the ground-truth/candidate solution fields,
$\alpha$/$\hat\alpha$ their coefficient vectors, $\Theta$ the candidate term dictionary, and
$\mathcal{S}(\hat\alpha) = \{\Theta_i : \hat\alpha_i \neq 0\}$ the candidate's active terms.

| Metric | Module | Formula | Description |
|---|---|---|---|
| `MSE` | `errors` | $\frac{1}{n}\sum_i (\hat u_i - u_i)^2$ | Mean squared error between candidate and ground-truth solutions |
| `nMSE` | `errors` | $\frac{\sum_i (\hat u_i - u_i)^2}{\sum_i u_i^2}$ | MSE normalized by the ground truth's squared magnitude |
| `nMAE` | `errors` | $\frac{\sum_i \lvert \hat u_i - u_i \rvert}{\sum_i \lvert u_i \rvert}$ | Mean absolute error normalized by the ground truth's magnitude |
| `n_utMSE` | `errors` | $\frac{\sum_i (\hat u_{t,i} - u_{t,i})^2}{\sum_i u_{t,i}^2}$ | Normalized MSE on the residual $u_t$ (RHS) instead of $u$ |
| `fMSE` | `errors` | $\frac{1}{N_T}\sum_t \frac{\sum_{k} \lvert \mathcal{F}(\hat u)(t,k) - \mathcal{F}(u)(t,k) \rvert^2}{k_{max}-k_{min}+1}$ | Mean squared error in Fourier space, over the meaningful wavenumber band |
| `rollout` | `errors` | $\frac{1}{N_T}\sum_{i=1}^{N_T} \lVert \hat u(i\delta_t,\cdot) - u(i\delta_t,\cdot) \rVert_{L^2(\Omega)}^2$ | Time-averaged L2 error between candidate and ground-truth trajectories |
| `nL2coef` | `coef` | $\frac{\lVert \alpha - \hat\alpha \rVert_2^2}{\lVert \alpha \rVert_2^2}$ | Normalized L2 error between candidate and ground-truth coefficient vectors |
| `nL1coef` | `coef` | $\frac{\lVert \alpha - \hat\alpha \rVert_1}{\lVert \alpha \rVert_1}$ | Normalized L1 error between candidate and ground-truth coefficient vectors |
| `maxerror` | `coef` | $\max_{i:\alpha_i \neq 0} \frac{\lvert \alpha_i - \hat\alpha_i \rvert}{\lvert \alpha_i \rvert}$ | Largest relative coefficient error over the ground truth's nonzero terms |
| `NDCG` | `coef` | $\frac{DCG(rank(\hat\alpha))}{DCG(rank(\alpha))}$, with $DCG(r) = \sum_i \frac{r_i}{\log_2(i+1)}$ | Ranking similarity of coefficient magnitudes (discounted cumulative gain ratio) |
| `wL2coef` | `coef` | $\frac{\sum_i w_i(\alpha_i - \hat\alpha_i)^2}{\sum_i w_i}$, with $w_i \propto \lVert \partial u/\partial \alpha_i \rVert_2^2$ | L2 coefficient error weighted by each term's sensitivity |
| `Tanimoto` | `coef` | $\frac{\alpha^\top \hat\alpha}{\lVert \alpha \rVert_2^2 + \lVert \hat\alpha \rVert_2^2 - \alpha^\top \hat\alpha}$ | Tanimoto/Jaccard-style similarity between coefficient vectors |
| `Sterms` | `sparsity` | $1 - \frac{\lvert \{ \hat\alpha_i = 0 \} \rvert}{\lvert \Theta \rvert}$ (or $\infty$ if $\hat\alpha = 0$) | Fraction of nonzero coefficients relative to the full candidate dictionary |
| `ExpTree` | `sparsity` | $size\left(ExpTree(\hat\alpha^\top \Theta)\right)$ | Size (node count) of the equation's expression tree |
| `Score` | `tradeoff` | $\frac{-\Delta \log(nMAE)}{\Delta C}$ | Accuracy-per-complexity gain of the candidate relative to a baseline PDE |
| `Reward1` | `tradeoff` | $\left(1 - c_0 \log_{10}\lvert \mathcal{S}(\hat\alpha) \rvert\right) \times \left(1 - \frac{\sum_i (u_{t,i} - \Theta_i^\top \hat\alpha)^2}{\sum_i (u_{t,i} - \bar u_t)^2}\right)$ | Sparsity-weighted R² fit on the $u_t$ residual |
| `Reward2` | `tradeoff` | $\frac{1 - \xi_1 \lvert \mathcal{S}(\hat\alpha) \rvert - \xi_2 \, depth(ExpTree(\hat\alpha^\top \Theta))}{1 + \lVert u_t - \hat\alpha^\top \Theta \rVert_2^2}$ | Sparsity/depth-penalized inverse residual error |
| `AICc` | `tradeoff` | $n \log \frac{\lVert u_t - \Theta^\top \hat\alpha \rVert_2^2}{n} + 2\lvert \mathcal{S}(\hat\alpha) \rvert$ | Corrected Akaike Information Criterion (fit vs. sparsity trade-off) |
| `BIC` | `tradeoff` | $\log(n)\lvert \mathcal{S}(\hat\alpha) \rvert - 2\log(L(\mathcal{T},\hat\alpha))$ | Bayesian Information Criterion (fit vs. sparsity trade-off) |
| `MDL_Fey` | `tradeoff` | $\log_2 N(\hat\alpha) + \lambda \log_2\left[\max\left(1, \frac{err(\hat\alpha)}{\epsilon_d}\right)\right]$ | AI-Feynman-style minimum description length score |
| `MDL_Sym` | `tradeoff` | $-\log p(\mathcal{T} \mid e(\Theta^\top \hat\alpha)) + \lambda \, len\left(ExpTree(\Theta^\top \hat\alpha)\right)$ | SymLang-style minimum description length score |
| `IC_nMAE` | `generalization` | $nMAE(\hat u, u)$ resimulated under an out-of-distribution initial condition | nMAE between candidate and ground truth under an unseen initial condition |
| `rollout_OOD` | `generalization` | $\frac{1}{N_T'}\sum_{i=1}^{N_T'} \lVert \hat u(i\delta_t,\cdot) - u(i\delta_t,\cdot) \rVert_{L^2(\Omega)}^2$, with $N_T' \gg N_T$ | Rollout error evaluated over a longer (out-of-distribution) horizon |
| `Conv_t` | `generalization` | $\lim_{\delta_t \rightarrow 0} \lVert \hat u - \hat u_{\delta_t} \rVert$ | Numerical convergence of the candidate PDE as the time step shrinks |
| `Conv_x` | `generalization` | $\lim_{h \rightarrow 0} \lVert \hat u - \hat u_h \rVert$ | Numerical convergence of the candidate PDE as the spatial grid is refined |

Each module name above can be passed directly to `--metric` to select all of its metrics at
once (e.g. `--metric errors`), as described in [Computing metrics](#computing-metrics).
