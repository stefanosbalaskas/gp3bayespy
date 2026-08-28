# Prediction Profiles, Surfaces, and Contrast Profiles

> Python-facing port of `prediction-profiles-surfaces-and-contrasts.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Governed prediction grids can be explored as one-dimensional profiles,
finite-difference predictive gradients, two-dimensional surfaces, and
two-level contrast profiles.

These are fitted predictive descriptions, not causal response curves or
automatic interaction tests.

## Python API mapping

- `gp3bayespy.create_prediction_contrast_profile`
- `gp3bayespy.create_prediction_profile`
- `gp3bayespy.create_prediction_surface`
- `gp3bayespy.plot_prediction_contrast_profile`
- `gp3bayespy.plot_prediction_gradient`
- `gp3bayespy.plot_prediction_profile`
- `gp3bayespy.plot_prediction_surface`
- `gp3bayespy.plot_prediction_surface_uncertainty`
- `gp3bayespy.prediction_gradient_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
