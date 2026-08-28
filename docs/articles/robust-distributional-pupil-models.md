# Robust and Distributional Pupil Models

> Python-facing port of `robust-distributional-pupil-models.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Why separate location and residual scale?

A Gaussian pupil model with constant residual standard deviation assumes the unexplained variability is similar across the full time course and conditions. gp3bayespy (R reference 0.5) can instead declare the residual scale as constant, condition-dependent, time-dependent, or condition-by-time dependent. The Student-t option provides a robust observation distribution without automatically labelling individual observations as invalid outliers.

# Candidate specifications are explicit

Student-t robustness and residual ARMA are deliberately not combined in the governed 0.5 interface. They should be treated as distinct modelling hypotheses and compared against the declared predictive target.

# Posterior residual-scale trajectory

## Python API mapping

- `gp3bayespy.estimate_pupil_residual_scale`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.plot_advanced_pupil_simulation`
- `gp3bayespy.plot_pupil_residual_scale`
- `gp3bayespy.pupil_distribution_table`
- `gp3bayespy.pupil_residual_scale_table`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`
- `gp3bayespy.specify_pupil_distribution`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
