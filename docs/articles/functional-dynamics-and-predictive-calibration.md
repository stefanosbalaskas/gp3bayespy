# Functional Dynamics and Predictive Calibration

> Python-facing port of `functional-dynamics-and-predictive-calibration.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Purpose

Version 0.5 treats the posterior trajectory itself as an object from which
predeclared functional estimands can be derived. This avoids using a single
peak or selected window as the only description of temporal change. The
functions in this article remain descriptive: they do not infer a physiological
onset, changepoint, attention state, or cognitive event.

## Backend-free simulation

## Fit declaration

The Student-t and ARMA layers are deliberately not combined by the governed
0.5 interface. Robust observation tails and residual serial dependence should
first be assessed as separately declared candidate explanations.

## Posterior derivatives

The following fit is intentionally not executed while building the vignette.

A derivative summarizes rate of posterior trajectory change. It is not an
automatic response-onset detector. Likewise, duration above a threshold is
only meaningful when that threshold was scientifically prespecified.

## Predictive calibration on held-out data

The reported RMSE, MAE, bias, interval coverage, interval width, and draw-based
CRPS describe the supplied prediction task. They become out-of-sample evidence
only when `newdata` was genuinely withheld from fitting.

## Python API mapping

- `gp3bayespy.audit_advanced_pupil_identifiability`
- `gp3bayespy.audit_pupil_predictive_calibration`
- `gp3bayespy.create_pupil_gp_spec`
- `gp3bayespy.estimate_pupil_dynamic_contrast`
- `gp3bayespy.estimate_pupil_threshold_duration`
- `gp3bayespy.estimate_pupil_trajectory_derivative`
- `gp3bayespy.fit_advanced_pupil_model_cmdstanr`
- `gp3bayespy.plot_advanced_pupil_simulation`
- `gp3bayespy.plot_pupil_dynamic_contrast`
- `gp3bayespy.plot_pupil_predictive_calibration`
- `gp3bayespy.plot_pupil_trajectory_derivative`
- `gp3bayespy.predict_advanced_pupil_trajectory`
- `gp3bayespy.pupil_model_card`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
