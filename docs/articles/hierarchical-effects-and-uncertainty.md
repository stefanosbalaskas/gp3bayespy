# Hierarchical Effects and Predictive Uncertainty

> Python-facing port of `hierarchical-effects-and-uncertainty.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

`gp3bayespy` treats group-level estimates as posterior quantities to inspect, not
as automatic rankings of participants or items.

## Grouped posterior predictive checks

The check compares observed group summaries with their posterior predictive
distribution. No group is automatically excluded.

## Descriptive uncertainty decomposition

The expected-response component and remaining predictive component are Monte
Carlo variance summaries under the fitted model. They should not be interpreted
as a causal variance decomposition.

## Python API mapping

- `gp3bayespy.group_effect_table`
- `gp3bayespy.grouped_prediction_check`
- `gp3bayespy.plot_group_effects`
- `gp3bayespy.plot_grouped_prediction_check`
- `gp3bayespy.plot_uncertainty_decomposition`
- `gp3bayespy.plot_variance_components`
- `gp3bayespy.prediction_uncertainty_decomposition`
- `gp3bayespy.variance_component_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
