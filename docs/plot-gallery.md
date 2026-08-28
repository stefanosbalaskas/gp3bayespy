# Plot gallery

Plotting functions return ordinary Matplotlib `Figure` objects. They do not save files unless a dedicated publication/registry API is explicitly given a destination.

Major plot families include:

- sampling diagnostics: traces, energy, divergences, treedepth;
- predictive diagnostics: ROC, precision-recall, calibration, posterior-predictive statistics, Q-Q/tail checks;
- prediction profiles, gradients, surfaces, contrast profiles, predictive-distribution atlases;
- PSIS-LOO pointwise/group influence and Pareto-k views;
- prior/posterior shift, contraction, distance, and hierarchical-effect graphics;
- sensitivity, recovery, and SBC graphics;
- pupil observed/posterior trajectories, PPC, ACF/spectra, GP/residual scale, derivatives, dynamic contrasts, binocular trajectories, response parameters, model comparison and calibration;
- publication dashboards, model cards, evidence inventories, and registries.

Example:

```python
import matplotlib.pyplot as plt
import gp3bayespy as gp

roc = gp.binary_roc_curve([0.05, 0.2, 0.75, 0.9], [0, 0, 1, 1])
fig = gp.plot_binary_roc(roc)
fig.savefig("roc.png", dpi=200, bbox_inches="tight")
plt.close(fig)
```
