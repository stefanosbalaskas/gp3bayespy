# Predictive Distribution and Calibration Uncertainty

> Python-facing port of `predictive-distribution-and-calibration-uncertainty.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Posterior predictive diagnostics can retain uncertainty in entire outcome
distributions as well as in predictive scores.

For binary models:

These summaries do not automatically establish calibration, predictive
adequacy, or model superiority.

## Python API mapping

- `gp3bayespy.binary_calibration_uncertainty`
- `gp3bayespy.binary_calibration_uncertainty_table`
- `gp3bayespy.create_predictive_distribution_atlas`
- `gp3bayespy.plot_binary_calibration_uncertainty`
- `gp3bayespy.plot_prediction_score_uncertainty`
- `gp3bayespy.plot_predictive_atlas_statistics`
- `gp3bayespy.plot_predictive_quantile_envelope`
- `gp3bayespy.prediction_score_uncertainty`
- `gp3bayespy.prediction_score_uncertainty_table`
- `gp3bayespy.predictive_quantile_envelope`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
