# Prediction, Calibration, and Scoring

> Python-facing port of `prediction-calibration-and-scoring.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The prediction API distinguishes conditional expected responses from new
posterior predictive outcomes. This distinction is retained in the returned
`gp3bayes_prediction` object and in downstream calibration and scoring tools.

## Fitted predictions

For binary fits, expected predictions are event probabilities and can be used
for calibration and threshold summaries. For duration fits, the API separately
exposes the arithmetic expected response, conditional median, and new-outcome
posterior predictive distribution.

All reported metrics are descriptive. The package does not choose a threshold
or certify predictive adequacy automatically.

## Python API mapping

- `gp3bayespy.audit_prediction_support`
- `gp3bayespy.binary_calibration_table`
- `gp3bayespy.binary_prediction_scores`
- `gp3bayespy.binary_threshold_metrics`
- `gp3bayespy.create_prediction_grid`
- `gp3bayespy.duration_pit_table`
- `gp3bayespy.duration_prediction_scores`
- `gp3bayespy.duration_quantile_calibration`
- `gp3bayespy.plot_binary_calibration`
- `gp3bayespy.plot_prediction_intervals`
- `gp3bayespy.plot_prediction_support`
- `gp3bayespy.predict_binary_probability`
- `gp3bayespy.predict_duration`
- `gp3bayespy.predict_model`
- `gp3bayespy.prediction_table`
- `gp3bayespy.predictive_coverage_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
