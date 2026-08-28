# Plot gallery

The gallery below is generated from `gp3bayespy` workflows during the site-polish step. Plotting functions return ordinary Matplotlib `Figure` objects and do not save files unless the caller explicitly does so.

<div class="gp-gallery">

<figure>
  <img src="../assets/gallery/binary-roc.png" alt="Binary ROC curve generated from gp3bayespy predictive diagnostics">
  <figcaption><strong>Binary ROC</strong><br>Threshold-free discrimination diagnostics from <code>binary_roc_curve()</code> and <code>plot_binary_roc()</code>.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/binary-pr.png" alt="Precision recall curve generated from gp3bayespy predictive diagnostics">
  <figcaption><strong>Precision–recall</strong><br>Class-performance diagnostics from <code>binary_precision_recall_curve()</code>.</figcaption>
</figure>

<figure class="gp-gallery__wide">
  <img src="../assets/gallery/pupil-simulation.png" alt="Simulated dynamic pupil trajectories generated with gp3bayespy">
  <figcaption><strong>Dynamic pupil simulation</strong><br>Condition-specific trajectories from an advanced synthetic pupil time-course workflow.</figcaption>
</figure>

</div>

## Reproduce the predictive plots

```python
import matplotlib.pyplot as plt
import gp3bayespy as gp

probability = [0.05, 0.20, 0.75, 0.90]
observed = [0, 0, 1, 1]

roc = gp.binary_roc_curve(probability, observed)
fig = gp.plot_binary_roc(roc)
fig.savefig("roc.png", dpi=200, bbox_inches="tight")
plt.close(fig)
```

## Plot families

<div class="grid cards" markdown>

-   **Sampling diagnostics**

    Traces, energy, divergences, treedepth, and chain-level evidence.

-   **Predictive diagnostics**

    ROC, precision–recall, calibration, posterior-predictive statistics, Q-Q checks, and tail checks.

-   **Prediction atlases**

    Profiles, gradients, two-dimensional surfaces, contrast profiles, interval width, rank probabilities, and predictive distributions.

-   **PSIS-LOO influence**

    Pointwise and grouped ELPD, Pareto-*k*, influence rank, and comparison graphics.

-   **Prior/posterior evidence**

    Density overlays, contraction, interval shifts, distance summaries, and hierarchical-effect graphics.

-   **Sensitivity & recovery**

    Sensitivity suites, parameter recovery, SBC, and robustness atlases.

-   **Dynamic pupillometry**

    Observed/posterior trajectories, temporal diagnostics, GP hyperparameters, derivatives, dynamic contrasts, binocular trajectories, response parameters, and model comparison.

-   **Publication systems**

    Dashboards, model cards, evidence inventories, and registry-ready figures.

</div>

For the functions behind these graphics, use the [API reference](reference/index.md).
