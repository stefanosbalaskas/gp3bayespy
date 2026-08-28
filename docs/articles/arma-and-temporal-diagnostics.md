# Bounded ARMA and Temporal Diagnostics

> Python-facing port of `arma-and-temporal-diagnostics.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Audit first; do not select an order from a plot

The observed ACF can reflect residual dependence, misspecified mean trajectories, design structure, preprocessing, or combinations of these. gp3bayes therefore treats the audit as descriptive evidence rather than an order-selection algorithm.

# Governed ARMA orders

Orders are bounded to AR(3)/MA(2). The common shortcuts are available directly in the model specification.

# Post-fit residual comparison

Residual spectra are deliberately descriptive. Peaks are not interpreted as cognitive or physiological rhythms by gp3bayes.

## Python API mapping

- `gp3bayespy.audit_pupil_temporal_dependence`
- `gp3bayespy.compare_pupil_autocorrelation`
- `gp3bayespy.create_pupil_arma_spec`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.plot_pupil_autocorrelation_comparison`
- `gp3bayespy.plot_pupil_residual_spectrum`
- `gp3bayespy.plot_pupil_temporal_dependence`
- `gp3bayespy.pupil_autocorrelation_table`
- `gp3bayespy.pupil_residual_spectrum`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
