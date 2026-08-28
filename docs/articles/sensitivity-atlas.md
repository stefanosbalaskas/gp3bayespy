# Sensitivity Atlases

> Python-facing port of `sensitivity-atlas.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Existing sensitivity workflows now share a publication-oriented table-and-plot
layer.

Local stability under a perturbation is not promoted to a universal robustness
claim.

## Python API mapping

- `gp3bayespy.estimand_sensitivity_table`
- `gp3bayespy.group_deletion_sensitivity_table`
- `gp3bayespy.plot_estimand_sensitivity_gg`
- `gp3bayespy.plot_group_deletion_sensitivity`
- `gp3bayespy.plot_powerscale_sensitivity_gg`
- `gp3bayespy.plot_prior_sensitivity`
- `gp3bayespy.plot_random_slope_sensitivity`
- `gp3bayespy.powerscale_sensitivity_table`
- `gp3bayespy.prior_sensitivity_table`
- `gp3bayespy.random_slope_sensitivity_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/sensitivity_workflow.py`.
